"""Structure-aware ranking: TM_a of a token Smith-Waterman alignment."""
import numpy as np
from numba import get_num_threads, njit, prange

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
    # interior pointer cells are all written before the traceback reads them
    ptr = np.empty((Lq + 1, Lt + 1), np.uint8)
    H_prev = np.zeros(Lt + 1, np.float32)
    H_curr = np.empty(Lt + 1, np.float32)
    H_curr[0] = 0.0
    F = np.full(Lt + 1, -np.inf, np.float32)
    best, bi, bj = np.float32(0.0), 0, 0
    for i in range(1, Lq + 1):
        E = np.float32(-np.inf)
        qi = q[i - 1]
        for j in range(1, Lt + 1):
            p = 0
            f_open = H_prev[j] + go
            f_ext = F[j] + ge
            if f_open > f_ext:
                F[j] = f_open
                p |= 8
            else:
                F[j] = f_ext
            e_open = H_curr[j - 1] + go
            e_ext = E + ge
            if e_open > e_ext:
                E = e_open
                p |= 4
            else:
                E = e_ext
            v = H_prev[j - 1] + sub[qi, t[j - 1]]
            src = 1
            if E > v:
                v = E
                src = 2
            if F[j] > v:
                v = F[j]
                src = 3
            if v < 0.0:
                v = np.float32(0.0)
            if v == 0.0:
                src = 0
            H_curr[j] = v
            ptr[i, j] = p | src
            if v > best:
                best, bi, bj = v, i, j
        # swap rows: every H_curr[j >= 1] is rewritten in the next row
        tmp = H_prev
        H_prev = H_curr
        H_curr = tmp
        H_curr[0] = 0.0
    qa = np.empty(min(Lq, Lt), np.int64)
    ta = np.empty(min(Lq, Lt), np.int64)
    k, i, j, state = 0, bi, bj, 0
    while i > 0 and j > 0:
        c = ptr[i, j]
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
def _refine(a, b, d0q, d0t, Lq, Lt, sel):
    cut = min(d0q, d0t) + 1.0
    best = 0.0
    for _ in range(N_REFINE):
        if sel.sum() < MIN_PAIRS:
            break
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
    return best


@njit(cache=True)
def alignment_tm_a(a, b, Lq, Lt, block_start=0, block_end=0):
    """TM_a of aligned C4' pairs a[k] <-> b[k].

    Return the best mean of the query- and target-normalised TM-scores over
    two seeds (all pairs and ``block_start:block_end``) and every refinement
    of each, refitting on pairs within min(d0) + 1 A.
    """
    n = a.shape[0]
    if n < MIN_PAIRS:
        return 0.0
    d0q, d0t = d0(Lq), d0(Lt)
    best = _refine(a, b, d0q, d0t, Lq, Lt, np.ones(n, np.bool_))
    if block_end - block_start >= MIN_PAIRS and (block_start != 0 or block_end != n):
        sel = np.zeros(n, np.bool_)
        sel[block_start:block_end] = True
        v = _refine(a, b, d0q, d0t, Lq, Lt, sel)
        if v > best:
            best = v
    return best


@njit(cache=True)
def score_pair(q, qxyz, t, txyz, sub, go, ge):
    """Return (S_SW, TM_a, number of aligned pairs) for one pair."""
    s, qa, ta = smith_waterman_traceback(q, t, sub, go, ge)
    bs, be = _best_block(q, t, qa, ta, sub)
    return s, alignment_tm_a(qxyz[qa], txyz[ta], len(q), len(t), bs, be), len(qa)


LANES = 8                    # targets aligned together by the batched SW
_PAD = np.float32(-1e30)     # substitution score of the padding token
PTR_BUDGET = 1 << 25         # traceback pointer bytes per batch (32 MiB); a
                             # longer batch gets fewer lanes, down to one pair


@njit(cache=True)
def _sw_lanes(qc, prof, go, ge, ptr, H_prev, H_curr, F, E, best, bi, bj):
    """Affine-gap SW of one query against LANES targets at once.

    The same recurrence, float32 operations and tie rules as
    `smith_waterman_traceback`, cell for cell; only independent alignments
    are interleaved (lane loop innermost). ``prof[c, j, l]`` is the score of
    query code ``c`` against position ``j`` of lane ``l``; lanes shorter than
    the batch are padded with ``_PAD``. A padded cell can never exceed the
    running best of its lane (its value is below an earlier cell of the same
    lane, or 0), and the traceback only moves up or left, so padding changes
    neither the score, the end cell nor the path.
    """
    Lq = qc.shape[0]
    T = prof.shape[1]
    L = prof.shape[2]
    ninf = np.float32(-np.inf)
    zero = np.float32(0.0)
    for j in range(T + 1):
        for l in range(L):
            H_prev[j, l] = zero
            F[j, l] = ninf
    for l in range(L):
        H_curr[0, l] = zero
        best[l] = zero
        bi[l] = 0
        bj[l] = 0
    for i in range(1, Lq + 1):
        row = prof[qc[i - 1]]
        for l in range(L):
            E[l] = ninf
        for j in range(1, T + 1):
            for l in range(L):
                f_open = H_prev[j, l] + go
                f_ext = F[j, l] + ge
                fo = f_open > f_ext
                fv = f_open if fo else f_ext
                F[j, l] = fv
                e_open = H_curr[j - 1, l] + go
                e_ext = E[l] + ge
                eo = e_open > e_ext
                ev = e_open if eo else e_ext
                E[l] = ev
                v0 = H_prev[j - 1, l] + row[j - 1, l]
                c2 = ev > v0
                v1 = ev if c2 else v0
                c3 = fv > v1
                v2 = fv if c3 else v1
                v = zero if v2 < zero else v2
                src = np.uint8(3) if c3 else (np.uint8(2) if c2 else np.uint8(1))
                src = np.uint8(0) if v == zero else src
                H_curr[j, l] = v
                ptr[i, j, l] = (src | (np.uint8(8) if fo else np.uint8(0))
                                | (np.uint8(4) if eo else np.uint8(0)))
                better = v > best[l]
                best[l] = v if better else best[l]
                bi[l] = i if better else bi[l]
                bj[l] = j if better else bj[l]
        tmp = H_prev
        H_prev = H_curr
        H_curr = tmp
        for l in range(L):
            H_curr[0, l] = zero


@njit(cache=True)
def _traceback_lane(ptr, l, i, j, Lq, Lt):
    """`smith_waterman_traceback`'s traceback, reading lane ``l``."""
    qa = np.empty(min(Lq, Lt), np.int64)
    ta = np.empty(min(Lq, Lt), np.int64)
    k, state = 0, 0
    while i > 0 and j > 0:
        c = ptr[i, j, l]
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
    return qa[:k][::-1].copy(), ta[:k][::-1].copy()


@njit(cache=True)
def _query_codes(q, n_codes):
    """Query recoded to 0..U-1 over the codes it uses, and those codes."""
    remap = np.full(n_codes, -1, np.int64)
    used = 0
    for i in range(q.shape[0]):
        if remap[q[i]] < 0:
            remap[q[i]] = used
            used += 1
    codes = np.empty(used, np.int64)
    for a in range(n_codes):
        if remap[a] >= 0:
            codes[remap[a]] = a
    qc = np.empty(q.shape[0], np.int64)
    for i in range(q.shape[0]):
        qc[i] = remap[q[i]]
    return qc, codes


@njit(cache=True)
def _score_batch(lo, hi, q, qc, qcodes, qxyz, cand, order, lens, toks, offsets,
                 xyz, sub, go, ge, S, T, A):
    """Score candidates ``order[lo:hi]`` (one lane each) into S, T, A."""
    nl = hi - lo
    width = 0
    for k in range(lo, hi):
        width = max(width, lens[order[k]])
    n_codes = sub.shape[0]
    tb = np.full((width, nl), n_codes, np.int64)
    for k in range(lo, hi):
        c = cand[order[k]]
        o = offsets[c]
        for j in range(lens[order[k]]):
            tb[j, k - lo] = toks[o + j]
    prof = np.empty((qcodes.shape[0], width, nl), np.float32)
    for u in range(qcodes.shape[0]):
        a = qcodes[u]
        for j in range(width):
            for l in range(nl):
                t = tb[j, l]
                prof[u, j, l] = sub[a, t] if t < n_codes else _PAD
    Lq = q.shape[0]
    ptr = np.empty((Lq + 1, width + 1, nl), np.uint8)
    H_prev = np.empty((width + 1, nl), np.float32)
    H_curr = np.empty((width + 1, nl), np.float32)
    F = np.empty((width + 1, nl), np.float32)
    E = np.empty(nl, np.float32)
    best = np.empty(nl, np.float32)
    bi = np.empty(nl, np.int64)
    bj = np.empty(nl, np.int64)
    _sw_lanes(qc, prof, go, ge, ptr, H_prev, H_curr, F, E, best, bi, bj)
    for k in range(lo, hi):
        l = k - lo
        idx = order[k]
        c = cand[idx]
        o = offsets[c]
        Lt = lens[idx]
        qa, ta = _traceback_lane(ptr, l, bi[l], bj[l], Lq, Lt)
        t = toks[o:o + Lt]
        bs, be = _best_block(q, t, qa, ta, sub)
        S[idx] = best[l]
        T[idx] = alignment_tm_a(qxyz[qa], xyz[o:o + Lt][ta], Lq, Lt, bs, be)
        A[idx] = len(qa)


@njit(cache=True)
def _batch_bounds(Lq, lens, order, lanes, ptr_budget):
    """Split the length-sorted candidates into batches of at most ``lanes``
    whose pointer matrix ((Lq + 1) x (width + 1) bytes per lane) stays within
    ``ptr_budget``; a single candidate always forms a batch on its own."""
    n = order.shape[0]
    bounds = np.empty(n + 1, np.int64)
    nb = 0
    bounds[0] = 0
    lo = 0
    while lo < n:
        hi = lo + 1
        while hi < n and hi - lo < lanes:
            # lengths ascend along ``order``, so the newcomer sets the width
            if (Lq + 1) * (lens[order[hi]] + 1) * (hi - lo + 1) > ptr_budget:
                break
            hi += 1
        nb += 1
        bounds[nb] = hi
        lo = hi
    return bounds[:nb + 1]


@njit(cache=True)
def _score_setup(q, cand, offsets, n_codes, lanes, ptr_budget):
    n = cand.shape[0]
    lens = offsets[cand + 1] - offsets[cand]
    order = np.argsort(lens, kind="mergesort")   # similar lengths share a batch
    qc, qcodes = _query_codes(q, n_codes)
    bounds = _batch_bounds(q.shape[0], lens, order, lanes, ptr_budget)
    return (lens, order, qc, qcodes, np.zeros(n), np.zeros(n),
            np.zeros(n, np.int64), bounds)


@njit(cache=True)
def _score_candidates_serial(q, qxyz, cand, toks, offsets, xyz, sub, go, ge,
                             lanes, ptr_budget):
    lens, order, qc, qcodes, S, T, A, bounds = _score_setup(
        q, cand, offsets, sub.shape[0], lanes, ptr_budget)
    for b in range(bounds.shape[0] - 1):
        _score_batch(bounds[b], bounds[b + 1], q, qc, qcodes, qxyz, cand,
                     order, lens, toks, offsets, xyz, sub, go, ge, S, T, A)
    return S, T, A


@njit(parallel=True, cache=True)
def _score_candidates_parallel(q, qxyz, cand, toks, offsets, xyz, sub, go, ge,
                               lanes, ptr_budget):
    lens, order, qc, qcodes, S, T, A, bounds = _score_setup(
        q, cand, offsets, sub.shape[0], lanes, ptr_budget)
    for b in prange(bounds.shape[0] - 1):
        _score_batch(bounds[b], bounds[b + 1], q, qc, qcodes, qxyz, cand,
                     order, lens, toks, offsets, xyz, sub, go, ge, S, T, A)
    return S, T, A


@njit(parallel=True, cache=True)
def _score_candidates_pairs(q, qxyz, cand, toks, offsets, xyz, sub, go, ge):
    """One `score_pair` per candidate: the path for non-float32 inputs."""
    n = len(cand)
    S = np.zeros(n)
    T = np.zeros(n)
    A = np.zeros(n, np.int64)
    for k in prange(n):
        c = cand[k]
        lo, hi = offsets[c], offsets[c + 1]
        S[k], T[k], A[k] = score_pair(q, qxyz, toks[lo:hi], xyz[lo:hi], sub, go, ge)
    return S, T, A


def score_candidates(q, qxyz, cand, toks, offsets, xyz, sub, go, ge):
    """Return (S_SW, TM_a, aligned pairs) for every candidate.

    Chain c occupies rows ``offsets[c]:offsets[c + 1]`` of flattened ``toks``
    and ``xyz``. Values equal `score_pair` on each candidate bit for bit.
    With a float32 matrix and float32 gap penalties (what `Searcher` passes)
    candidates are aligned up to LANES at a time, with the pointer matrix of
    a batch kept within PTR_BUDGET bytes; any other input dtype is scored
    pair by pair, since the batched recurrence is float32 throughout.
    Batches run in parallel unless Numba is limited to one thread.
    """
    cand = np.asarray(cand, np.int64)
    if not (sub.dtype == np.float32 and isinstance(go, np.float32)
            and isinstance(ge, np.float32)):
        return _score_candidates_pairs(q, qxyz, cand, toks, offsets, xyz,
                                       sub, go, ge)
    scorer = (_score_candidates_serial if get_num_threads() == 1
              else _score_candidates_parallel)
    return scorer(q, qxyz, cand, toks, offsets, xyz, sub, go, ge,
                  LANES, PTR_BUDGET)


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
