"""
Smith-Waterman local alignment for VQ-VAE token sequences.

Given two token sequences and a substitution matrix (the deployed
Structure-BLOSUM in `files/models/sub_matrix.pt`), compute the optimal
local alignment score. Used as the scoring function for structure search.

Usage
-----
    from smith_waterman import smith_waterman
    score = smith_waterman(query_tokens, target_tokens, sub_matrix, gap_open=-2)
"""

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover - optional acceleration
    njit = None


def smith_waterman(
    query: np.ndarray,
    target: np.ndarray,
    sub_matrix: np.ndarray,
    gap_open: float = -2.0,
    gap_extend: float = -0.5,
) -> float:
    """Smith-Waterman local alignment score (affine gap).

    Args:
        query:      (Lq,) int token sequence
        target:     (Lt,) int token sequence
        sub_matrix: (N, N) substitution scores indexed by token id
        gap_open:   cost of opening a gap (negative)
        gap_extend: cost of extending a gap (negative)

    Returns:
        Best local alignment score (0 = no alignment).
    """
    Lq, Lt = len(query), len(target)
    if Lq == 0 or Lt == 0:
        return 0.0

    # Three DP tables for affine gap: H (match), E (gap in query), F (gap in target)
    H = np.zeros((Lq + 1, Lt + 1), dtype=np.float32)
    E = np.full((Lq + 1, Lt + 1), -np.inf, dtype=np.float32)
    F = np.full((Lq + 1, Lt + 1), -np.inf, dtype=np.float32)

    # Lookup substitution scores in a vectorized way per-row
    best = 0.0
    for i in range(1, Lq + 1):
        qi = query[i - 1]
        # E[i, j] = max(H[i, j-1] + gap_open, E[i, j-1] + gap_extend)  — build column-wise
        for j in range(1, Lt + 1):
            tj = target[j - 1]
            E[i, j] = max(H[i, j - 1] + gap_open, E[i, j - 1] + gap_extend)
            F[i, j] = max(H[i - 1, j] + gap_open, F[i - 1, j] + gap_extend)
            diag = H[i - 1, j - 1] + float(sub_matrix[qi, tj])
            H[i, j] = max(0.0, diag, E[i, j], F[i, j])
            if H[i, j] > best:
                best = H[i, j]
    return float(best)


def smith_waterman_fast(
    query: np.ndarray,
    target: np.ndarray,
    sub_matrix: np.ndarray,
    gap_open: float = -2.0,
    gap_extend: float = -0.5,
) -> float:
    """Vectorized SW with affine gap (row-wise numpy, ~5-10x faster)."""
    Lq, Lt = len(query), len(target)
    if Lq == 0 or Lt == 0:
        return 0.0

    # Precompute score rows: score_row[i] = sub_matrix[query[i], target] — (Lq, Lt)
    score_rows = sub_matrix[query[:, None], target[None, :]].astype(np.float32)

    H_prev = np.zeros(Lt + 1, dtype=np.float32)
    H_curr = np.zeros(Lt + 1, dtype=np.float32)
    F = np.full(Lt + 1, -np.inf, dtype=np.float32)
    best = 0.0

    for i in range(1, Lq + 1):
        E = -np.inf
        # F column: F[j] = max(H_prev[j] + gap_open, F[j] + gap_extend)
        F = np.maximum(H_prev + gap_open, F + gap_extend)

        row_scores = score_rows[i - 1]  # (Lt,)
        for j in range(1, Lt + 1):
            E = max(H_curr[j - 1] + gap_open, E + gap_extend)
            diag = H_prev[j - 1] + row_scores[j - 1]
            v = max(0.0, diag, E, F[j])
            H_curr[j] = v
            if v > best:
                best = v

        H_prev, H_curr = H_curr, H_prev
        H_curr[:] = 0.0
    return float(best)


def smith_waterman_batch(
    query: np.ndarray,
    targets_padded: np.ndarray,
    target_lens: np.ndarray,
    sub_matrix: np.ndarray,
    gap_open: float = -2.0,
    gap_extend: float = -0.5,
) -> np.ndarray:
    """Batched SW: one query vs N targets, returning (N,) scores.

    Args:
        query:           (Lq,) int tokens
        targets_padded:  (N, Lt_max) int tokens, padded with any valid id
        target_lens:     (N,) int lengths; cells with j > target_lens[n]
                         are masked out of the 'best' accumulator.
        sub_matrix:      (n_codes, n_codes) float scores
        gap_open, gap_extend: affine gap penalties

    The inner j-loop is sequential (DP in j), but every numpy op inside
    is vectorized across the batch of N targets — ~100x faster than
    calling smith_waterman_fast in a Python loop when N is large.
    """
    Lq = int(len(query))
    N, Lt_max = targets_padded.shape
    if Lq == 0 or Lt_max == 0 or N == 0:
        return np.zeros(N, dtype=np.float32)

    # Valid-cell mask: True iff j <= target_lens[n], j in [1..Lt_max]
    j_range = np.arange(1, Lt_max + 1)
    valid_j = j_range[None, :] <= target_lens[:, None]  # (N, Lt_max)

    # H[:, 0] and column 0 stay zero (SW local)
    H_prev = np.zeros((N, Lt_max + 1), dtype=np.float32)
    H_curr = np.zeros((N, Lt_max + 1), dtype=np.float32)
    F = np.full((N, Lt_max + 1), -np.inf, dtype=np.float32)
    best = np.zeros(N, dtype=np.float32)

    for i in range(1, Lq + 1):
        # F column update: F[:, j] = max(H_prev[:, j] + go, F[:, j] + ge)
        F = np.maximum(H_prev + gap_open, F + gap_extend)

        # row_scores[n, j-1] = sub_matrix[query[i-1], targets_padded[n, j-1]]
        row_scores = sub_matrix[query[i - 1], targets_padded]  # (N, Lt_max)

        E = np.full(N, -np.inf, dtype=np.float32)
        for j in range(1, Lt_max + 1):
            # E[j] = max(H_curr[:, j-1] + go, E + ge)
            E = np.maximum(H_curr[:, j - 1] + gap_open, E + gap_extend)
            diag = H_prev[:, j - 1] + row_scores[:, j - 1]
            v = np.maximum(0.0, diag)
            v = np.maximum(v, E)
            v = np.maximum(v, F[:, j])
            H_curr[:, j] = v
            # only update best where this cell is inside the target length
            mask = valid_j[:, j - 1]
            # best = max(best, v * mask)  — but 0 won't hurt since best init 0
            best = np.where(mask & (v > best), v, best)

        # swap rolling rows
        H_prev, H_curr = H_curr, H_prev
        H_curr.fill(0.0)

    return best


if njit is not None:

    @njit(cache=True)
    def _smith_waterman_numba(query, target, sub_matrix, gap_open, gap_extend):
        Lq = len(query)
        Lt = len(target)
        if Lq == 0 or Lt == 0:
            return 0.0

        H_prev = np.zeros(Lt + 1, dtype=np.float32)
        H_curr = np.zeros(Lt + 1, dtype=np.float32)
        F = np.empty(Lt + 1, dtype=np.float32)
        for j in range(Lt + 1):
            F[j] = -np.inf

        best = 0.0
        for i in range(1, Lq + 1):
            E = -np.inf
            qi = query[i - 1]
            for j in range(1, Lt + 1):
                f_open = H_prev[j] + gap_open
                f_ext = F[j] + gap_extend
                F[j] = f_open if f_open > f_ext else f_ext

                e_open = H_curr[j - 1] + gap_open
                e_ext = E + gap_extend
                E = e_open if e_open > e_ext else e_ext

                diag = H_prev[j - 1] + sub_matrix[qi, target[j - 1]]
                v = diag
                if E > v:
                    v = E
                if F[j] > v:
                    v = F[j]
                if v < 0.0:
                    v = 0.0
                H_curr[j] = v
                if v > best:
                    best = v

            tmp = H_prev
            H_prev = H_curr
            H_curr = tmp
            for j in range(Lt + 1):
                H_curr[j] = 0.0

        return float(best)


def smith_waterman_numba(
    query: np.ndarray,
    target: np.ndarray,
    sub_matrix: np.ndarray,
    gap_open: float = -2.0,
    gap_extend: float = -0.5,
) -> float:
    """Numba-JIT affine-gap Smith-Waterman."""
    if njit is None:
        return smith_waterman_fast(query, target, sub_matrix, gap_open, gap_extend)
    return float(
        _smith_waterman_numba(
            query.astype(np.int64, copy=False),
            target.astype(np.int64, copy=False),
            sub_matrix.astype(np.float32, copy=False),
            float(gap_open),
            float(gap_extend),
        )
    )


if __name__ == "__main__":
    # Sanity test: identical sequences should score high
    import torch

    rng = np.random.RandomState(0)
    n_codes = 64
    # Synthetic sub matrix: identity-like, mean 0
    sub = rng.randn(n_codes, n_codes).astype(np.float32) * 0.1
    sub = (sub + sub.T) / 2
    np.fill_diagonal(sub, 1.0)

    a = rng.randint(0, n_codes, size=50)
    b = a.copy()  # identical
    c = rng.randint(0, n_codes, size=50)  # random

    s_same = smith_waterman_fast(a, b, sub)
    s_rand = smith_waterman_fast(a, c, sub)
    print("identical:  %.2f" % s_same)
    print("random:     %.2f" % s_rand)
    print("sanity: identical >> random ?", s_same > 5 * max(s_rand, 0.01))

    # Batched parity check: smith_waterman_batch should match smith_waterman_fast
    targets = np.stack([b, c])
    lens = np.array([len(b), len(c)])
    sb = smith_waterman_batch(a, targets, lens, sub)
    sf = np.array([smith_waterman_fast(a, b, sub), smith_waterman_fast(a, c, sub)])
    print("batched:    ", sb, "  fast:", sf, "  match:", np.allclose(sb, sf, atol=1e-3))
