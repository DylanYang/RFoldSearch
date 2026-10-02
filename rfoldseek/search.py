"""The search pipeline: encode -> prefilter -> rerank.

Two reranking backends share the token prefilter. ``atm`` (default) aligns
candidates by affine-gap token Smith-Waterman, superposes the aligned C4'
atoms and ranks by R_str = S_SW * aTM, the Smith-Waterman score times the alignment
TM-score (structure_score.py); it reports the cluster-maximum Gumbel E-value
of S_SW. ``tm`` (RFoldseek-TM) hands the candidate pool to US-align and ranks by
the mean of its two length-normalised TM-scores; that backend reports no
E-value. Both need coordinates for the query and the database.
"""
import json
import os
import warnings

import numpy as np
import torch

from .database import load_database, short_id
from .prefilter import KMER_K, KmerPrefilter
from . import structure_score as _ss
from . import evalue as _evalue
from . import usalign as _usalign
from . import tiebreak as _tiebreak

GAP_OPEN = -2.0
GAP_EXTEND = -0.5
DEFAULT_TOP_N = 200      # candidate-pool size after the prefilter (validation-selected)
DEFAULT_MIN_DIAG = 2     # Foldseek double-match rule
DEFAULT_RERANK = "atm"
RERANKERS = ("atm", "tm")
_NO_CUT = 1 << 62        # ask the prefilter for every candidate, uncut

_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def asset(name):
    return os.path.join(_ASSETS, name)


def load_matrix(path=None):
    m = torch.load(path or asset("sub_matrix.pt"), map_location="cpu",
                   weights_only=False)
    return np.asarray(m["matrix"].numpy() if isinstance(m, dict) else m,
                      dtype=np.float32)


class Searcher:
    """Load a database once, search many queries."""

    def __init__(self, db_path, matrix_path=None, evalue_params_path=None):
        (self.db, self.ents, self.toks, self.ids, self.n_codes,
         self.clusters) = load_database(db_path)
        self.id2i = {cid: i for i, cid in enumerate(self.ids)}
        self.sub = load_matrix(matrix_path)
        self.prefilter = KmerPrefilter(self.toks, self.n_codes, self.sub)
        # E-values count the search space in sequence clusters, so a family
        # held four hundred times poses one question of the database.
        self.n_clusters = len(set(self.clusters))
        ev_path = evalue_params_path or asset("evalue_params.json")
        if evalue_params_path is None and self.db.get("evalue_params") is not None:
            self.ev_par = self.db["evalue_params"]
        else:
            self.ev_par = json.load(open(ev_path)) if os.path.exists(ev_path) else None
        if (self.ev_par is not None
                and self.ev_par.get("query_parameter_model") is not None
                and "training_domain" not in self.ev_par["query_parameter_model"]):
            raise ValueError("query-conditional E-value model needs a training_domain")
        self.empirical_ev = (self.ev_par or {}).get("empirical_calibration")
        if self.empirical_ev is not None:
            fitted_n = self.empirical_ev.get(
                "n_clusters", self.ev_par.get("n_clusters"))
            if fitted_n is None:
                raise ValueError("empirical E-value calibration needs a cluster count")
            if fitted_n is not None and int(fitted_n) != self.n_clusters:
                raise ValueError(
                    "empirical E-value calibration was fitted for "
                    f"{int(fitted_n)} clusters, but the loaded database has "
                    f"{self.n_clusters}; recalibrate for this database"
                )
        elif self.ev_par is not None:
            # Parametric-only params: the cluster-max null was fitted on one
            # database. Reused on a database of a different size or structural
            # makeup (e.g. predicted structures, which are more homogeneous and
            # drive unrelated scores up) the raw E-value can be badly
            # anticonservative. Warn if the calibration database differs.
            fitted_n = self.ev_par.get("n_clusters")
            if (fitted_n and self.n_clusters
                    and not 0.5 < int(fitted_n) / self.n_clusters < 2.0):
                warnings.warn(
                    f"E-value parameters were calibrated on {int(fitted_n)} "
                    f"clusters but this database has {self.n_clusters} "
                    "(>2x different); the parametric E-value may be "
                    "miscalibrated on a database of this size or structural "
                    "makeup. Fit a database-specific empirical_calibration "
                    "(scripts/build_empirical_evalue_background.py + "
                    "scripts/fit_empirical_evalue_calibration.py).",
                    stacklevel=2,
                )
        self._tie_cache = {}
        self._flat = None

    def _flat_structures(self):
        """Flattened tokens and C4' coordinates for aTM, built on first use.
        ``None`` when any chain lacks coordinates for its tokens."""
        if self._flat is None:
            xyz = [_ss.c4_coords(e, len(t)) for e, t in zip(self.ents, self.toks)]
            if any(x is None for x in xyz):
                self._flat = False
            else:
                offsets = np.zeros(len(self.toks) + 1, np.int64)
                offsets[1:] = np.cumsum([len(t) for t in self.toks])
                self._flat = (np.concatenate(self.toks).astype(np.int64), offsets,
                              np.concatenate(xyz))
        return self._flat or None

    def tie_keys(self, seed):
        """Hash tie-break keys for every chain, cached per seed."""
        if seed not in self._tie_cache:
            self._tie_cache[seed] = _tiebreak.tie_keys(self.ids, seed)
        return self._tie_cache[seed]

    def search(self, qtok, top_k=10, top_n=DEFAULT_TOP_N,
               min_diag=DEFAULT_MIN_DIAG, self_idx=-1, rerank=DEFAULT_RERANK,
               usalign=None, query_entry=None, tie_seed=None):
        """-> list of hit dicts, ranked by the chosen backend's score.

        ``rerank="atm"`` (default) ranks by ``structure_score`` (R_str = S_SW * aTM) and
        reports ``atm``, ``aligned`` (aligned nucleotide pairs), an E-value,
        and ``sw_score`` (S_SW) and ``bits`` from which the E-value is computed.
        ``rerank="tm"`` ranks by US-align TM-score and reports
        ``tm_mean``/``tm_query``/``tm_target`` instead. Both need the query
        coordinates: pass ``query_entry``, or search a chain in the database.

        ``tie_seed`` makes equal scores break by a hash of the chain id
        rather than by database index, at both the prefilter's ``top_n``
        cut and the reranking. ``None`` keeps the historical index-order
        behaviour.
        """
        if rerank not in RERANKERS:
            raise ValueError(f"unknown rerank backend {rerank!r}; use one of {RERANKERS}")
        qtok = np.asarray(qtok, dtype=np.int64)
        if len(qtok) < KMER_K:
            raise ValueError(f"query has {len(qtok)} tokens; need >= {KMER_K}")
        if tie_seed is None:
            cand, _ = self.prefilter.candidates(qtok, top_n, min_diag, self_idx)
        else:
            cand, pre = self.prefilter.candidates(qtok, _NO_CUT, min_diag, self_idx)
            if len(cand):
                keys = self.tie_keys(tie_seed)[cand]
                cand = cand[np.lexsort((keys, -pre))][:top_n]
        if len(cand) == 0:
            return []
        if rerank == "tm":
            if query_entry is None:
                if self_idx < 0:
                    raise ValueError(
                        "TM reranking needs query coordinates; search a chain in "
                        "the database or pass query_entry"
                    )
                query_entry = self.ents[self_idx]
            return self._rerank_tm(query_entry, cand, top_k, usalign, tie_seed)
        (toks, offsets, xyz), qxyz = self._atm_inputs(qtok, query_entry, self_idx)
        sw, atm, aligned = _ss.score_candidates(
            qtok, qxyz, np.asarray(cand, np.int64), toks, offsets, xyz,
            self.sub, np.float32(GAP_OPEN), np.float32(GAP_EXTEND))
        key = sw * atm
        if tie_seed is None:
            order = np.argsort(-key, kind="stable")
        else:
            order = np.lexsort((self.tie_keys(tie_seed)[cand], -key))
        hits = []
        if self.ev_par is not None:
            mu, beta = _evalue.clustermax_params_for_query(self.ev_par, qtok)
        for rank, oi in enumerate(order[:top_k], 1):
            t, sc = int(cand[oi]), float(sw[oi])
            hit = {"rank": rank, "target": self.ids[t], "sw_score": sc,
                   "length": len(self.toks[t])}
            hit["structure_score"] = float(key[oi])
            hit["atm"] = float(atm[oi])
            hit["aligned"] = int(aligned[oi])
            if self.ev_par is not None:
                hit["bits"] = float(_evalue.bit_score_clustermax(sc, mu, beta))
                raw_evalue = float(_evalue.evalue_clustermax(
                    sc, mu, beta, self.n_clusters))
                if self.empirical_ev is not None:
                    hit["raw_evalue"] = raw_evalue
                    hit["evalue"] = float(_evalue.apply_empirical_calibration(
                        raw_evalue, self.empirical_ev))
                else:
                    hit["evalue"] = raw_evalue
            hits.append(hit)
        return hits

    def _atm_inputs(self, qtok, query_entry, self_idx):
        """(flattened database, query C4') for aTM."""
        flat = self._flat_structures()
        if flat is None:
            raise ValueError(
                "structure-aware ranking needs C4' coordinates for every "
                "database chain; rebuild the database from structures")
        entry = query_entry if query_entry is not None else (
            self.ents[self_idx] if self_idx >= 0 else None)
        qxyz = _ss.c4_coords(entry, len(qtok)) if entry is not None else None
        if qxyz is None:
            raise ValueError(
                "structure-aware ranking needs the query's C4' coordinates; "
                "pass query_entry, or search a chain in the database")
        return flat, qxyz

    def _rerank_tm(self, query_entry, cand, top_k, usalign, tie_seed=None):
        """Rank the candidate pool by US-align TM-score (RFoldseek-TM)."""
        binary = _usalign.find_usalign(usalign)
        pool = [(self.ids[int(t)], self.ents[int(t)]["coords"],
                 self.ents[int(t)]["mask"]) for t in cand]
        scores, skipped = _usalign.score_pool(
            binary, query_entry["coords"], query_entry["mask"], pool)
        self.last_tm_skipped = skipped
        keys = self.tie_keys(tie_seed) if tie_seed is not None else None

        def _key(r):
            mean = -0.5 * (r[1] + r[2])
            return (mean, keys[self.id2i[r[0]]]) if keys is not None else (mean,)

        ranked = sorted(
            ((cid, tm_q, tm_t) for cid, (tm_q, tm_t) in scores.items()), key=_key)
        hits = []
        for rank, (cid, tm_q, tm_t) in enumerate(ranked[:top_k], 1):
            hits.append({
                "rank": rank,
                "target": cid,
                "tm_mean": 0.5 * (tm_q + tm_t),
                "tm_query": float(tm_q),
                "tm_target": float(tm_t),
                "length": len(self.toks[self.id2i[cid]]),
            })
        return hits

    def search_id(self, chain_id, **kw):
        """Search a chain already in the database (self-hit removed)."""
        i = self.id2i[chain_id]
        return self.search(self.toks[i], self_idx=i, **kw)
