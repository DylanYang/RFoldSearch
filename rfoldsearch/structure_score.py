"""Structure-aware ranking: TM_a of a token Smith-Waterman alignment."""
import numpy as np
from numba import njit, prange

C4_INDEX = 1        # coords are (C3', C4', O4', C1', base anchor) per residue
N_REFINE = 5        # superposition refinements per seed
MIN_PAIRS = 3


@njit(cache=True)
def smith_waterman_traceback(q, t, sub, go, ge):
    """Affine-gap SW -> (score, query indices, target indices) of aligned pairs.

    Pointer byte per cell: bits 0-1 the H source (0 start, 1 diagonal,
    2 gap in query, 3 gap in target); bit 2 the query-gap state was opened
    here; bit 3 the target-gap state was opened here.
    """
    Lq, Lt = len(q), len(t)
    W = Lt + 1
    # Flat row-major pointer matrix; row 0 and column 0 are never read, so
    # it is left uninitialised.
    ptr = np.empty((Lq + 1) * W, np.uint8)
    H_prev = np.zeros(W, np.float32)
    H_curr = np.zeros(W, np.float32)
    F = np.full(W, -np.inf, np.float32)
    tt = t.astype(np.int32)
    best, bi, bj = np.float32(0.0), 0, 0
    for i in range(1, Lq + 1):
        E = np.float32(-np.inf)
        srow = sub[q[i - 1]]
        base = i * W
        h_left = np.float32(0.0)          # H(i, j - 1); H(i, 0) = 0
        for j in range(1, W):
            p = 0
            f_open = H_prev[j] + go
            f_ext = F[j] + ge
            if f_open > f_ext:
                f = f_open
                p |= 8
            else:
                f = f_ext
            F[j] = f
            e_open = h_left + go
            e_ext = E + ge
            if e_open > e_ext:
                E = e_open
                p |= 4
            else:
                E = e_ext
            v = H_prev[j - 1] + srow[tt[j - 1]]
            src = 1
            if E > v:
                v = E
                src = 2
            if f > v:
                v = f
                src = 3
            if v < 0.0:
                v = np.float32(0.0)
            if v == 0.0:
                src = 0
            H_curr[j] = v
            h_left = v
            ptr[base + j] = p | src
            if v > best:
                best, bi, bj = v, i, j
        # Every H_curr[j >= 1] is rewritten in the next row and H_curr[0]
        # stays 0, so swapping the rows replaces copying and zeroing.
        H_prev, H_curr = H_curr, H_prev
    qa = np.empty(min(Lq, Lt), np.int64)
    ta = np.empty(min(Lq, Lt), np.int64)
    k, i, j, state = 0, bi, bj, 0
    while i > 0 and j > 0:
        c = ptr[i * W + j]
        if state == 0:
            src = c & 3
            if src == 0:
                break
            if src == 1:
                qa[k], ta[k] = i - 1, j - 1
                k += 1
                i, j = i - 1, j - 1
            elif src == 2:
                state = 1
            else:
                state = 2
        elif state == 1:                      # gap in query: move left
            if c & 4:
                state = 0
            j -= 1
        else:                                 # gap in target: move up
            if c & 8:
                state = 0
            i -= 1
    return best, qa[:k][::-1].copy(), ta[:k][::-1].copy()


@njit(cache=True)
def d0(L):
    """TM-score distance scale for RNA, in Angstrom."""
    return max(0.5, 0.6 * np.sqrt(max(L - 0.5, 0.0)) - 2.5)


@njit(cache=True)
def _kabsch(a, b):
    """Rotation r and translation tr minimising |a @ r.T + tr - b|."""
    ca = a.sum(0) / a.shape[0]
    cb = b.sum(0) / b.shape[0]
    h = (a - ca).T @ (b - cb)
    u, s, vt = np.linalg.svd(h)
    dm = np.eye(3)
    dm[2, 2] = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ dm @ u.T
    return r, cb - ca @ r.T


@njit(cache=True)
def _best_block(q, t, qa, ta, sub):
    """Highest-scoring gapless stretch of the alignment, as [start, end)."""
    best, bs, be = -np.inf, 0, 0
    cur, cs = 0.0, 0
    for k in range(len(qa)):
        if k > 0 and not (qa[k] == qa[k - 1] + 1 and ta[k] == ta[k - 1] + 1):
            cur, cs = 0.0, k
        v = sub[q[qa[k]], t[ta[k]]]
        if cur <= 0.0:
            cur, cs = v, k
        else:
            cur += v
        if cur > best:
            best, bs, be = cur, cs, k + 1
    return bs, be


@njit(cache=True)
def _refine(a, b, d0q, d0t, Lq, Lt, sel, seen, n_seen, record):
    """Best TM_a over up to N_REFINE superposition refinements from ``sel``.

    With ``record`` the selection used at each step k is stored in
    ``seen[k]``; returns (best, number of stored steps). Otherwise the run
    stops once its selection at step k equals ``seen[k1]`` for some
    k1 <= k: the rest of the run then repeats the recorded run from step k1
    with no more steps left than it had, so it cannot raise the maximum.
    """
    cut = min(d0q, d0t) + 1.0
    best = 0.0
    for k in range(N_REFINE):
        if sel.sum() < MIN_PAIRS:
            break
        if record:
            seen[k, :] = sel
            n_seen = k + 1
        else:
            for k1 in range(min(k + 1, n_seen)):
                if np.all(seen[k1] == sel):
                    return best, n_seen
        r, tr = _kabsch(a[sel], b[sel])
        d = np.sqrt((((a @ r.T + tr) - b) ** 2).sum(1))
        s = 0.5 * ((1.0 / (1.0 + (d / d0q) ** 2)).sum() / Lq
                   + (1.0 / (1.0 + (d / d0t) ** 2)).sum() / Lt)
        if s > best:
            best = s
        new = d < cut
        if new.sum() < MIN_PAIRS or np.all(new == sel):
            break
        sel = new
    return best, n_seen


@njit(cache=True)
def alignment_tm_a(a, b, Lq, Lt, block_start=0, block_end=0):
    """TM_a of aligned C4' pairs a[k] <-> b[k].

    Return the best mean of the query- and target-normalised TM-scores over
    two seeds (all pairs and ``block_start:block_end``) and every refinement
    of each, refitting on pairs within min(d0) + 1 A. Refinements of the
    block seed that repeat a step of the all-pairs run are skipped (always
    the case when the alignment is gapless and the block is every pair).
    """
    n = a.shape[0]
    if n < MIN_PAIRS:
        return 0.0
    d0q, d0t = d0(Lq), d0(Lt)
    seen = np.empty((N_REFINE, n), np.bool_)
    best, n_seen = _refine(a, b, d0q, d0t, Lq, Lt, np.ones(n, np.bool_),
                           seen, 0, True)
    if block_end - block_start >= MIN_PAIRS:
        sel = np.zeros(n, np.bool_)
        sel[block_start:block_end] = True
        v, _ = _refine(a, b, d0q, d0t, Lq, Lt, sel, seen, n_seen, False)
        if v > best:
            best = v
    return best


@njit(cache=True)
def score_pair(q, qxyz, t, txyz, sub, go, ge):
    """Return (S_SW, TM_a, number of aligned pairs) for one pair."""
    s, qa, ta = smith_waterman_traceback(q, t, sub, go, ge)
    bs, be = _best_block(q, t, qa, ta, sub)
    return s, alignment_tm_a(qxyz[qa], txyz[ta], len(q), len(t), bs, be), len(qa)


@njit(parallel=True, cache=True)
def score_candidates(q, qxyz, cand, toks, offsets, xyz, sub, go, ge):
    """Return (S_SW, TM_a, aligned pairs) for every candidate in parallel.

    Chain c occupies rows ``offsets[c]:offsets[c + 1]`` of flattened ``toks``
    and ``xyz``.
    """
    n = len(cand)
    S = np.zeros(n)
    T = np.zeros(n)
    A = np.zeros(n, np.int64)
    for k in prange(n):
        c = cand[k]
        lo, hi = offsets[c], offsets[c + 1]
        S[k], T[k], A[k] = score_pair(q, qxyz, toks[lo:hi], xyz[lo:hi], sub, go, ge)
    return S, T, A


def c4_coords(entry, n_tokens):
    """C4' coordinates of the tokenised residues, or None if unavailable."""
    coords, mask = entry.get("coords"), entry.get("mask")
    if coords is None or mask is None:
        return None
    coords = np.asarray(coords, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if coords.ndim != 3 or coords.shape[1] <= C4_INDEX or len(mask) != len(coords):
        return None
    xyz = coords[mask, C4_INDEX, :]
    if len(xyz) != n_tokens or not np.all(np.isfinite(xyz)):
        return None
    return np.ascontiguousarray(xyz)
