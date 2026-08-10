"""The search pipeline: encode -> prefilter -> token Smith-Waterman -> E-value."""
import json
import os

import numpy as np
import torch

from .database import load_database, short_id
from .prefilter import KMER_K, KmerPrefilter
from .smith_waterman import smith_waterman_numba
from . import evalue as _evalue

GAP_OPEN = -2.0
GAP_EXTEND = -0.5
DEFAULT_TOP_N = 200      # candidate-pool size after the prefilter (validation-selected)
DEFAULT_MIN_DIAG = 2     # Foldseek double-match rule

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
        self.db, self.ents, self.toks, self.ids, self.n_codes = load_database(db_path)
        self.id2i = {cid: i for i, cid in enumerate(self.ids)}
        self.sub = load_matrix(matrix_path)
        self.prefilter = KmerPrefilter(self.toks, self.n_codes, self.sub)
        self.n_db_tokens = int(sum(len(t) for t in self.toks))
        ev_path = evalue_params_path or asset("evalue_params.json")
        self.ev_par = json.load(open(ev_path)) if os.path.exists(ev_path) else None

    def search(self, qtok, top_k=10, top_n=DEFAULT_TOP_N,
               min_diag=DEFAULT_MIN_DIAG, self_idx=-1):
        """-> list of hit dicts, ranked by Smith-Waterman score."""
        qtok = np.asarray(qtok, dtype=np.int64)
        if len(qtok) < KMER_K:
            raise ValueError(f"query has {len(qtok)} tokens; need >= {KMER_K}")
        cand, _ = self.prefilter.candidates(qtok, top_n, min_diag, self_idx)
        if len(cand) == 0:
            return []
        sw = np.array([smith_waterman_numba(qtok, self.toks[t], self.sub,
                                            GAP_OPEN, GAP_EXTEND) for t in cand])
        order = np.argsort(-sw, kind="stable")
        hits = []
        for rank, oi in enumerate(order[:top_k], 1):
            t, sc = int(cand[oi]), float(sw[oi])
            hit = {"rank": rank, "target": self.ids[t], "sw_score": sc,
                   "length": len(self.toks[t])}
            if self.ev_par is not None:
                lam_m, k_m = _evalue.params_for_length(self.ev_par, len(qtok))
                hit["bits"] = float(_evalue.bit_score(sc, len(qtok), lam_m, k_m))
                hit["evalue"] = float(_evalue.evalue(sc, len(qtok),
                                                     self.n_db_tokens, lam_m, k_m))
            hits.append(hit)
        return hits

    def search_id(self, chain_id, **kw):
        """Search a chain already in the database (self-hit removed)."""
        i = self.id2i[chain_id]
        return self.search(self.toks[i], self_idx=i, **kw)
