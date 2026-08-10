"""Reference prefilter: the paper pipeline's implementation, kept verbatim.

Ported from the RFoldseek CLI in RFoldseek-paper-code. It is the semantic
ground truth the compiled kernel in `prefilter.py` must match; the
equivalence test runs both and demands identical output.
"""
from collections import defaultdict

import numpy as np

from .prefilter import KMER_K


def build_index(toks, K):
    powers = np.array([K ** (KMER_K - 1 - i) for i in range(KMER_K)], dtype=np.int64)
    idx = defaultdict(list)
    for ti, tt in enumerate(toks):
        for p in range(len(tt) - KMER_K + 1):
            idx[int(np.dot(tt[p:p + KMER_K], powers))].append((ti, p))
    return idx, powers


def make_expander(sub, K, powers, thr):
    diag = np.array([sub[t, t] for t in range(K)], dtype=np.float64)

    def expand(qk):
        base = int(np.dot(qk, powers))
        out = [base]
        self_score = float(sum(diag[int(qk[i])] for i in range(KMER_K)))
        for pos in range(KMER_K):
            d = int(powers[pos])
            cur = int(qk[pos])
            base_wo = self_score - diag[cur]
            row = sub[cur]
            for tok in range(K):
                if tok != cur and base_wo + row[tok] >= thr:
                    out.append(base + d * (tok - cur))
        return out

    return expand


def candidates(qtok, toks, idx, expand, min_diag, self_idx, top_n):
    per_target = defaultdict(dict)
    for qp in range(len(qtok) - KMER_K + 1):
        for kid in expand(qtok[qp:qp + KMER_K]):
            for ti, tp in idx.get(kid, ()):
                if ti == self_idx:
                    continue
                d = qp - tp
                per_target[ti][d] = per_target[ti].get(d, 0) + 1
    scored = [(ti, max((c for c in dm.values() if c >= min_diag), default=-1))
              for ti, dm in per_target.items()]
    scored = [x for x in scored if x[1] > 0]
    scored.sort(key=lambda x: (-x[1], x[0]))
    return (np.array([ti for ti, _ in scored[:top_n]], dtype=np.int64),
            np.array([s for _, s in scored[:top_n]], dtype=np.int64))
