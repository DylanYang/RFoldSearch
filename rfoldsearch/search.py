"""The search pipeline: encode -> prefilter -> rerank.

Three reranking backends share the token prefilter. ``tm_a`` (default) aligns
candidates by affine-gap token Smith-Waterman, superposes the aligned C4'
atoms and ranks by R_str = S_SW * TM_a. ``sw`` keeps the token-only
Smith-Waterman ranking. Both report the E-value of S_SW. ``tm``
(RFoldSearch-TM) hands the candidate pool to US-align and ranks by the mean of
its two length-normalised TM-scores; that backend reports no E-value.
"""
import json
import os
import warnings

import numpy as np
import torch

from .database import (calibration_context_fingerprint, database_fingerprint,
                       load_database, short_id)
from .prefilter import KMER_K, KmerPrefilter
from .smith_waterman import smith_waterman_numba
from . import structure_score as _ss
from . import evalue as _evalue
from . import experimental_evalue as _experimental_evalue
from . import usalign as _usalign
from . import tiebreak as _tiebreak

GAP_OPEN = -2.0
GAP_EXTEND = -0.5
DEFAULT_TOP_N = 200      # candidate-pool size after the prefilter (validation-selected)
DEFAULT_MIN_DIAG = 2     # Foldseek double-match rule
DEFAULT_RERANK = "tm_a"
RERANKERS = ("tm_a", "sw", "tm")
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
        embedded_params = self.db.get("evalue_params") is not None
        using_packaged_default = evalue_params_path is None and not embedded_params
        if evalue_params_path is None and embedded_params:
            self.ev_par = self.db["evalue_params"]
        else:
            if os.path.exists(ev_path):
                with open(ev_path) as handle:
                    self.ev_par = json.load(handle)
            else:
                self.ev_par = None
        target_fingerprint = database_fingerprint(
            self.ents, self.clusters, self.n_codes)
        stored_fingerprint = self.db.get("database_fingerprint")
        if (stored_fingerprint is not None
                and stored_fingerprint != target_fingerprint):
            raise ValueError("stored database fingerprint does not match its contents")
        score_fingerprint = calibration_context_fingerprint(
            target_fingerprint, self.sub, GAP_OPEN, GAP_EXTEND)
        if using_packaged_default:
            experimental_path = asset("evalue_params_experimental_empirical.json")
            if os.path.exists(experimental_path):
                with open(experimental_path) as handle:
                    experimental_params = json.load(handle)
                if (experimental_params.get("target_database_fingerprint") ==
                        target_fingerprint and
                        experimental_params.get("calibration_context_fingerprint") ==
                        score_fingerprint):
                    self.ev_par = experimental_params
        if (self.ev_par is not None
                and self.ev_par.get("validation_status") ==
                "provisional-incomplete-structural-audit"):
            warnings.warn(
                "The E-value parameters are provisional because their "
                "structural-background audit is incomplete; this search "
                "will not report an E-value.",
                stacklevel=2,
            )
            self.ev_par = None
        if self.ev_par is not None:
            fitted_target = self.ev_par.get("target_database_fingerprint")
            fitted_context = self.ev_par.get("calibration_context_fingerprint")
            mismatch = ((fitted_target is not None
                         and fitted_target != target_fingerprint)
                        or (fitted_context is not None
                            and fitted_context != score_fingerprint))
            if mismatch and using_packaged_default:
                warnings.warn(
                    "The packaged E-value parameters belong to a different target "
                    "database or scoring context; this search will not report an "
                    "E-value. Attach parameters fitted for the loaded database.",
                    stacklevel=2,
                )
                self.ev_par = None
            elif mismatch:
                raise ValueError(
                    "E-value parameters were fitted for a different target database "
                    "or scoring context"
                )
        if (self.ev_par is not None
                and self.ev_par.get("query_parameter_model") is not None
                and "training_domain" not in self.ev_par["query_parameter_model"]):
            raise ValueError("query-conditional E-value model needs a training_domain")
        fitted_clusters = (self.ev_par or {}).get("n_clusters")
        if (fitted_clusters is not None
                and self.db.get("n_clusters") is not None
                and int(fitted_clusters) != self.n_clusters):
            raise ValueError(
                "E-value parameters were fitted for "
                f"{int(fitted_clusters)} clusters, but the loaded database has "
                f"{self.n_clusters}; attach parameters fitted for this database"
            )
        self.experimental_ev = (self.ev_par if self.ev_par is not None
                                and self.ev_par.get("scheme") ==
                                _experimental_evalue.SCHEME else None)
        if self.experimental_ev is not None:
            _experimental_evalue.validate_parameters(
                self.experimental_ev, self.n_clusters, target_fingerprint,
                score_fingerprint)
        self.power_ev = ((self.ev_par or {}).get("power_calibration")
                         if self.experimental_ev is None else None)
        if self.power_ev is not None:
            _evalue.apply_power_calibration(0.0, self.power_ev)
            validation_status = self.power_ev.get("validation_status")
            if validation_status in {
                    "provisional-incomplete-structural-audit",
                    "structural-audit-not-recorded"}:
                warnings.warn(
                    "The power-law E-value calibration is provisional because "
                    "a complete structural-background audit is not recorded; "
                    "reporting the initial parametric E-value only.",
                    stacklevel=2,
                )
                self.power_ev = None
        if self.power_ev is not None:
            fitted_n = self.power_ev.get(
                "n_clusters", self.ev_par.get("n_clusters"))
            if fitted_n is None:
                raise ValueError("power E-value calibration needs a cluster count")
            if int(fitted_n) != self.n_clusters:
                if using_packaged_default:
                    warnings.warn(
                        "The packaged power E-value calibration is for the "
                        f"{int(fitted_n)}-cluster experimental database, but the "
                        f"loaded database has {self.n_clusters} clusters; reporting "
                        "the initial parametric E-value only. Attach this database's "
                        "own calibration to report a calibrated E-value.",
                        stacklevel=2,
                    )
                    self.power_ev = None
                else:
                    raise ValueError(
                        "power E-value calibration was fitted for "
                        f"{int(fitted_n)} clusters, but the loaded database has "
                        f"{self.n_clusters}; recalibrate for this database"
                    )
        elif self.ev_par is not None and self.experimental_ev is None:
            fitted_n = self.ev_par.get("n_clusters")
            if (fitted_n and self.n_clusters
                    and not 0.5 < int(fitted_n) / self.n_clusters < 2.0):
                warnings.warn(
                    f"E-value parameters were calibrated on {int(fitted_n)} "
                    f"clusters but this database has {self.n_clusters} "
                    "(>2x different); the parametric E-value may be "
                    "miscalibrated. Fit a database-specific power_calibration "
                    "(scripts/build_evalue_calibration_background.py + "
                    "scripts/fit_power_evalue_calibration.py).",
                    stacklevel=2,
                )
        self._tie_cache = {}
        self._flat = None

    def _flat_structures(self):
        """Flattened tokens and C4' coordinates for TM_a, built on first use.

        Return ``None`` when any database chain lacks coordinates matching its
        token sequence.
        """
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

        ``rerank="tm_a"`` (default) ranks by ``structure_score``
        (R_str = S_SW * TM_a) and reports ``tm_a``, ``aligned`` (aligned
        nucleotide pairs), and the S_SW-derived fields ``sw_score``, ``bits``,
        and E-value. ``rerank="sw"`` retains the token-only Smith-Waterman
        ranking. ``rerank="tm"`` ranks by US-align TM-score and reports
        ``tm_mean``/``tm_query``/``tm_target`` instead.

        ``tie_seed`` makes equal scores break by a hash of the chain id
        rather than by database index, at both the prefilter's ``top_n``
        cut and the reranking. ``None`` keeps the historical index-order
        behaviour.
        """
        if rerank not in RERANKERS:
            raise ValueError(
                f"unknown rerank backend {rerank!r}; use one of {RERANKERS}")
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
        tm_a = aligned = None
        if rerank == "tm_a":
            (toks, offsets, xyz), qxyz = self._tm_a_inputs(
                qtok, query_entry, self_idx)
            sw, tm_a, aligned = _ss.score_candidates(
                qtok, qxyz, np.asarray(cand, np.int64), toks, offsets, xyz,
                self.sub, np.float32(GAP_OPEN), np.float32(GAP_EXTEND))
            score = sw * tm_a
        else:
            sw = np.array([
                smith_waterman_numba(qtok, self.toks[t], self.sub,
                                     GAP_OPEN, GAP_EXTEND)
                for t in cand
            ])
            score = sw
        if tie_seed is None:
            order = np.argsort(-score, kind="stable")
        else:
            order = np.lexsort((self.tie_keys(tie_seed)[cand], -score))
        hits = []
        if self.ev_par is not None and self.experimental_ev is None:
            mu, beta = _evalue.clustermax_params_for_query(self.ev_par, qtok)
        if self.experimental_ev is not None:
            entropy = _experimental_evalue.token_entropy(qtok)
        for rank, oi in enumerate(order[:top_k], 1):
            t, sc = int(cand[oi]), float(sw[oi])
            hit = {"rank": rank, "target": self.ids[t], "sw_score": sc,
                   "length": len(self.toks[t])}
            if tm_a is not None:
                hit["structure_score"] = float(score[oi])
                hit["tm_a"] = float(tm_a[oi])
                hit["aligned"] = int(aligned[oi])
            if self.ev_par is not None:
                if self.experimental_ev is not None:
                    raw_evalue = _experimental_evalue.initial_evalue(
                        sc, len(qtok), entropy,
                        self.experimental_ev["target_aware_gumbel"])
                    hit["bits"] = float(-np.log2(max(
                        raw_evalue / self.n_clusters, np.finfo(float).tiny)))
                    hit["raw_evalue"] = raw_evalue
                    hit["evalue"] = _experimental_evalue.apply_empirical(
                        raw_evalue, self.experimental_ev["empirical_calibration"])
                else:
                    hit["bits"] = float(_evalue.bit_score_clustermax(sc, mu, beta))
                    raw_evalue = float(_evalue.evalue_clustermax(
                        sc, mu, beta, self.n_clusters))
                    if self.power_ev is not None:
                        hit["raw_evalue"] = raw_evalue
                        hit["evalue"] = float(_evalue.apply_power_calibration(
                            raw_evalue, self.power_ev))
                    else:
                        hit["evalue"] = raw_evalue
            hits.append(hit)
        return hits

    def _tm_a_inputs(self, qtok, query_entry, self_idx):
        """Return (flattened database, query C4') for TM_a ranking."""
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
        """Rank the candidate pool by US-align TM-score (RFoldSearch-TM)."""
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
