"""The search pipeline: encode -> prefilter -> rerank.

Three reranking backends share the token prefilter. ``tm_a`` (default) aligns
candidates by affine-gap token Smith-Waterman, superposes the aligned C4'
atoms and ranks by R_str = S_SW * TM_a. ``sw`` keeps the token-only
Smith-Waterman ranking. Both report the E-value of S_SW. ``tm``
(RFoldseek-TM) hands the candidate pool to US-align and ranks by the mean of
its two length-normalised TM-scores; that backend reports no E-value.
"""
import json
import os

import numpy as np
import torch

from .database import (calibration_context_fingerprint, database_fingerprint,
                       cluster_median_lengths, load_database, short_id)
from .prefilter import KMER_K, KmerPrefilter
from .smith_waterman import smith_waterman_numba
from . import structure_score as _ss
from . import evalue as _evalue
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
        # E-value parameters are inseparable from their target database. Do
        # not silently fall back to a bundled calibration: it would mix a
        # different database's null distribution into this search.
        if evalue_params_path is None and self.db.get("evalue_params") is not None:
            self.ev_par = self.db["evalue_params"]
        elif evalue_params_path is not None:
            with open(evalue_params_path) as handle:
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
        if self.ev_par is not None:
            _evalue.validate_target_aware_params(
                self.ev_par, self.n_clusters, target_fingerprint,
                score_fingerprint,
                cluster_median_lengths(self.ents, self.clusters))
        self.power_ev = self.ev_par["power_calibration"] if self.ev_par is not None else None
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
        nucleotide pairs), and the S_SW-derived fields ``sw_score``,
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
        for rank, oi in enumerate(order[:top_k], 1):
            t, sc = int(cand[oi]), float(sw[oi])
            hit = {"rank": rank, "target": self.ids[t], "sw_score": sc,
                   "length": len(self.toks[t])}
            if tm_a is not None:
                hit["structure_score"] = float(score[oi])
                hit["tm_a"] = float(tm_a[oi])
                hit["aligned"] = int(aligned[oi])
            if self.ev_par is not None:
                target_length = len(self.toks[t])
                hit["normalized_sw_score"] = sc / min(len(qtok), target_length)
                raw_evalue = float(_evalue.target_aware_evalue(
                    self.ev_par, qtok, sc, target_length))
                hit["raw_evalue"] = raw_evalue
                hit["evalue"] = float(_evalue.apply_power_calibration(
                    raw_evalue, self.power_ev))
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
