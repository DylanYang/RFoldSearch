"""The release prefilter: a compiled kernel over a CSR k-mer index.

Computes exactly what the reference implementation computes -- per-target
best-diagonal hit counts under similar-k-mer expansion, Foldseek's
double-match rule, ties broken toward the lower target index.

Each code's posting list is sorted by (target, position), so the kernel walks
the targets in cache-sized blocks with one cursor per expanded query code and
reads every posting once. Inside a block, a dense (targets x diagonals)
histogram that fits in cache is incremented and each target's best diagonal
updated on the fly; a block stamp in the upper 16 bits of each cell retires
the previous block's counts without a clearing pass. Postings are packed as
``(target << pbits) | position`` in one uint32 when the database allows.
``best[t] = max_d count(t, d)`` does not depend on the order postings are
visited, so the counts are identical to the counting-sort kernel kept below
as the fallback (very large databases or queries of 65,535+ tokens).
Equivalence to `prefilter_reference` is enforced by
tests/test_prefilter_equivalence.py.
"""
import numpy as np
from numba import njit

KMER_K = 4
THRESH_FRAC = 0.75
HIST_BYTES = 1 << 20     # diagonal histogram per target block; ~L2-sized
_COUNT_BITS = 16         # low bits of a histogram cell: the diagonal count
_COUNT_MASK = (1 << _COUNT_BITS) - 1


def build_csr_index(toks, n_codes):
    powers = np.array([n_codes ** (KMER_K - 1 - i) for i in range(KMER_K)],
                      dtype=np.int64)
    codes, targets, positions = [], [], []
    for ti, tt in enumerate(toks):
        if len(tt) < KMER_K:
            continue
        w = np.lib.stride_tricks.sliding_window_view(np.asarray(tt, np.int64), KMER_K)
        codes.append(w @ powers)
        targets.append(np.full(w.shape[0], ti, dtype=np.int64))
        positions.append(np.arange(w.shape[0], dtype=np.int64))
    codes = np.concatenate(codes)
    targets = np.concatenate(targets)
    positions = np.concatenate(positions)
    order = np.argsort(codes, kind="stable")
    codes, targets, positions = codes[order], targets[order], positions[order]
    offsets = np.zeros(n_codes ** KMER_K + 1, dtype=np.int64)
    np.add.at(offsets, codes + 1, 1)
    np.cumsum(offsets, out=offsets)
    return {"offsets": offsets, "targets": targets.astype(np.int32),
            "positions": positions.astype(np.int32), "powers": powers,
            "n_codes": n_codes, "n_targets": len(toks),
            "max_tlen": max(len(t) for t in toks)}


def expand_query(qtok, powers, sub, diag, thr):
    """Every admitted code for every query position, as (codes, qps)."""
    qtok = np.asarray(qtok, dtype=np.int64)
    n_q = len(qtok) - KMER_K + 1
    if n_q <= 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    win = np.lib.stride_tricks.sliding_window_view(qtok, KMER_K)
    base = win @ powers
    need = thr - diag[win].sum(axis=1)
    all_codes, all_qp = [base], [np.arange(n_q)]
    for slot in range(KMER_K):
        cur = win[:, slot]
        ok = (sub[cur] - diag[cur][:, None]) >= need[:, None]
        ok[np.arange(n_q), cur] = False
        qp_i, tok_i = np.nonzero(ok)
        if qp_i.size:
            all_codes.append(base[qp_i] + powers[slot] * (tok_i - cur[qp_i]))
            all_qp.append(qp_i)
    return np.concatenate(all_codes), np.concatenate(all_qp)


@njit(cache=True)
def _best_diagonal_counts(codes, qps, offsets, targets, positions,
                          self_idx, n_targets, span, seg_count, best, span_buf):
    for t in range(n_targets):
        seg_count[t] = 0
        best[t] = 0
    total = 0
    for i in range(codes.shape[0]):
        lo, hi = offsets[codes[i]], offsets[codes[i] + 1]
        for e in range(lo, hi):
            t = targets[e]
            if t != self_idx:
                seg_count[t] += 1
                total += 1
    seg_off = np.empty(n_targets + 1, np.int64)
    s = 0
    for t in range(n_targets):
        seg_off[t] = s
        s += seg_count[t]
    seg_off[n_targets] = s
    diags = np.empty(total, np.int32)
    cur = seg_off[:-1].copy()
    for i in range(codes.shape[0]):
        qp = qps[i]
        lo, hi = offsets[codes[i]], offsets[codes[i] + 1]
        for e in range(lo, hi):
            t = targets[e]
            if t != self_idx:
                diags[cur[t]] = qp - positions[e] + span
                cur[t] += 1
    for t in range(n_targets):
        lo, hi = seg_off[t], seg_off[t + 1]
        if lo == hi:
            continue
        m = 0
        for e in range(lo, hi):
            d = diags[e]
            span_buf[d] += 1
            if span_buf[d] > m:
                m = span_buf[d]
        best[t] = m
        for e in range(lo, hi):
            span_buf[diags[e]] = 0
    return best


def pack_postings(index):
    """``(target << pbits) | position`` per posting as uint32, or None if it
    does not fit in 32 bits."""
    pbits = max(1, int(index["max_tlen"] - 1).bit_length())
    tbits = max(1, int(index["n_targets"] - 1).bit_length())
    if pbits + tbits > 32:
        return None
    packed = ((index["targets"].astype(np.uint32) << np.uint32(pbits))
              | index["positions"].astype(np.uint32))
    return packed, pbits


@njit(cache=True)
def _best_diagonal_blocked(codes, qps, offsets, post, pbits, self_idx,
                           n_targets, lengths, n_q, bases, best, cursor, hist,
                           stamp0):
    """Per-target best-diagonal counts over packed postings, block by block.

    Fills ``best`` and returns the next free histogram stamp.
    """
    nc = codes.shape[0]
    pmask = np.uint32((1 << pbits) - 1)
    shift = np.uint32(pbits)
    high = np.uint32(0xFFFFFFFF ^ _COUNT_MASK)
    count_mask = np.uint32(_COUNT_MASK)
    for t in range(n_targets):
        best[t] = 0
    for i in range(nc):
        cursor[i] = offsets[codes[i]]
    stamp = stamp0
    t0 = 0
    while t0 < n_targets:
        t1 = t0
        used = 0
        while t1 < n_targets:
            width = lengths[t1] + n_q
            if used + width > hist.size:
                break
            bases[t1] = used + lengths[t1]
            used += width
            t1 += 1
        limit = np.uint64(t1) << np.uint64(pbits)
        tag = np.uint32(stamp) << np.uint32(_COUNT_BITS)
        for i in range(nc):
            e = cursor[i]
            hi = offsets[codes[i] + 1]
            qp = qps[i]
            while e < hi:
                w = post[e]
                if np.uint64(w) >= limit:
                    break
                t = np.int64(w >> shift)
                if t != self_idx:
                    d = bases[t] + qp - np.int64(w & pmask)
                    c = hist[d]
                    if (c & high) != tag:
                        c = tag
                    c += np.uint32(1)
                    hist[d] = c
                    v = np.int32(c & count_mask)
                    if v > best[t]:
                        best[t] = v
                e += 1
            cursor[i] = e
        t0 = t1
        stamp += 1
        if stamp > _COUNT_MASK:
            hist[:] = 0
            stamp = 1
    return stamp


@njit(cache=True)
def _best_diagonal_counts_packed(codes, qps, offsets, post, pbits, self_idx,
                                 n_targets, span, seg_count, best, span_buf):
    """`_best_diagonal_counts` reading packed postings (long-query fallback)."""
    pmask = np.uint32((1 << pbits) - 1)
    shift = np.uint32(pbits)
    for t in range(n_targets):
        seg_count[t] = 0
        best[t] = 0
    total = 0
    for i in range(codes.shape[0]):
        for e in range(offsets[codes[i]], offsets[codes[i] + 1]):
            t = np.int64(post[e] >> shift)
            if t != self_idx:
                seg_count[t] += 1
                total += 1
    seg_off = np.empty(n_targets + 1, np.int64)
    s = 0
    for t in range(n_targets):
        seg_off[t] = s
        s += seg_count[t]
    seg_off[n_targets] = s
    diags = np.empty(total, np.int32)
    cur = seg_off[:-1].copy()
    for i in range(codes.shape[0]):
        qp = qps[i]
        for e in range(offsets[codes[i]], offsets[codes[i] + 1]):
            w = post[e]
            t = np.int64(w >> shift)
            if t != self_idx:
                diags[cur[t]] = qp - np.int64(w & pmask) + span
                cur[t] += 1
    for t in range(n_targets):
        lo, hi = seg_off[t], seg_off[t + 1]
        if lo == hi:
            continue
        m = 0
        for e in range(lo, hi):
            d = diags[e]
            span_buf[d] += 1
            if span_buf[d] > m:
                m = span_buf[d]
        best[t] = m
        for e in range(lo, hi):
            span_buf[diags[e]] = 0
    return best


class KmerPrefilter:
    """Build once per database, then `candidates()` per query."""

    def __init__(self, toks, n_codes, sub, thr=None):
        self.index = build_csr_index(toks, n_codes)
        self.sub = np.asarray(sub, dtype=np.float32)
        self.diag = np.array([self.sub[t, t] for t in range(n_codes)],
                             dtype=np.float64)
        self.thr = THRESH_FRAC * KMER_K if thr is None else thr
        n_t = self.index["n_targets"]
        self._seg = np.zeros(n_t, np.int64)
        self._best = np.zeros(n_t, np.int32)
        self._span = self.index["max_tlen"]
        self._buf = np.zeros(self._span + 8192, np.int32)
        self.hist_bytes = HIST_BYTES
        self._cursor = np.zeros(0, np.int64)
        self._lengths = np.asarray([len(t) for t in toks], dtype=np.int64)
        self._bases = np.empty(n_t, np.int64)
        self._hist = np.zeros(0, np.uint32)
        self._stamp = 1
        packed = pack_postings(self.index)
        if packed is not None:
            # the packed array replaces the two int32 posting arrays
            self.index["packed"], self.index["pbits"] = packed
            del self.index["targets"], self.index["positions"]

    def _best_diagonals(self, codes, qps, self_idx, n_q):
        ix = self.index
        n_t = ix["n_targets"]
        if "packed" in ix and n_q < _COUNT_MASK:
            width = self._span + n_q + 8
            capacity = max(width, self.hist_bytes // 4)
            if self._hist.size < capacity:
                self._hist = np.zeros(capacity, np.uint32)
                self._stamp = 1
            if self._cursor.shape[0] < codes.shape[0]:
                self._cursor = np.zeros(codes.shape[0], np.int64)
            self._stamp = _best_diagonal_blocked(
                codes, qps, ix["offsets"], ix["packed"], ix["pbits"],
                self_idx, n_t, self._lengths, n_q, self._bases, self._best,
                self._cursor, self._hist, self._stamp)
            return self._best
        # queries longer than any target need a wider diagonal buffer
        need = self._span + n_q + 8
        if need > self._buf.shape[0]:
            self._buf = np.zeros(need, np.int32)
        if "packed" in ix:
            return _best_diagonal_counts_packed(
                codes, qps, ix["offsets"], ix["packed"], ix["pbits"],
                self_idx, n_t, self._span, self._seg, self._best, self._buf)
        return _best_diagonal_counts(
            codes, qps, ix["offsets"], ix["targets"], ix["positions"],
            self_idx, n_t, self._span, self._seg, self._best, self._buf)

    def candidates(self, qtok, top_n, min_diag, self_idx=-1):
        ix = self.index
        codes, qps = expand_query(qtok, ix["powers"], self.sub, self.diag, self.thr)
        if codes.size == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        best = self._best_diagonals(codes, qps.astype(np.int64), self_idx,
                                    len(qtok) - KMER_K + 1)
        cand = np.flatnonzero(best >= min_diag)
        if cand.size == 0 or top_n <= 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        scores = best[cand]
        # (-score, target index) as one unique int64 key: selecting top_n
        # before sorting avoids a full sort and keeps the tie rule exactly.
        keys = -scores.astype(np.int64) * (ix["n_targets"] + 1) + cand
        if cand.size > top_n:
            keep = np.argpartition(keys, top_n - 1)[:top_n]
            keep = keep[np.argsort(keys[keep], kind="stable")]
        else:
            keep = np.argsort(keys, kind="stable")
        return cand[keep].astype(np.int64), scores[keep].astype(np.int64)
