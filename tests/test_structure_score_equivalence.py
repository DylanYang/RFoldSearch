"""The rewritten SW traceback and TM_a must reproduce v1.0.0 bit for bit."""
import numpy as np

import structure_score_v100 as v100
from rfoldsearch import structure_score as ss
from rfoldsearch.search import GAP_EXTEND, GAP_OPEN, load_matrix

GO, GE = np.float32(GAP_OPEN), np.float32(GAP_EXTEND)


def _rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _pairs(rng, n):
    """Token pairs from near-identical (gapless alignments) to unrelated,
    with indels, homopolymer runs, and very short chains."""
    for k in range(n):
        L = int(rng.integers(4, 160))
        base = rng.integers(0, 80, size=L)
        if k % 5 == 0:
            base[: L // 2] = 62                       # helix-like run
        kind = k % 4
        if kind == 0:
            q, t = base.copy(), base.copy()
            t[rng.integers(0, L, L // 10 + 1)] = rng.integers(0, 80, L // 10 + 1)
        elif kind == 1:
            q = base.copy()
            t = np.delete(base, rng.integers(0, L, L // 8 + 1))
            t = np.insert(t, rng.integers(0, len(t) + 1, 3), rng.integers(0, 80, 3))
        elif kind == 2:
            q = base.copy()
            t = rng.integers(0, 80, size=int(rng.integers(4, 160)))
        else:
            q = base[: max(4, L // 3)].copy()
            t = np.concatenate([rng.integers(0, 80, 20), base, rng.integers(0, 80, 20)])
        yield q.astype(np.int64), t.astype(np.int64)


def test_traceback_matches_v100_bitwise():
    rng = np.random.default_rng(10)
    sub = load_matrix()
    for q, t in _pairs(rng, 400):
        s1, qa1, ta1 = v100.smith_waterman_traceback(q, t, sub, GO, GE)
        s2, qa2, ta2 = ss.smith_waterman_traceback(q, t, sub, GO, GE)
        assert s1 == s2
        assert np.array_equal(qa1, qa2) and np.array_equal(ta1, ta2)


def test_score_candidates_matches_v100_bitwise():
    rng = np.random.default_rng(11)
    sub = load_matrix()
    pairs = list(_pairs(rng, 300))
    q = pairs[0][0]
    fold = np.cumsum(rng.normal(size=(400, 3)) * 3.0, axis=0)
    qxyz = fold[: len(q)].copy()
    toks, xyz, offs = [], [], [0]
    for k, (_, t) in enumerate(pairs):
        L = len(t)
        if k % 3 == 0:          # same fold, rigidly moved, part perturbed
            x = fold[:L] @ _rotation(rng).T + rng.normal(size=3) * 10.0
            x[L // 2:] += rng.normal(size=(L - L // 2, 3)) * (k % 7)
        else:
            x = np.cumsum(rng.normal(size=(L, 3)) * 3.0, axis=0)
        toks.append(t)
        xyz.append(x)
        offs.append(offs[-1] + L)
    toks = np.concatenate(toks)
    xyz = np.ascontiguousarray(np.concatenate(xyz))
    offs = np.array(offs, np.int64)
    cand = np.arange(len(pairs), dtype=np.int64)
    r1 = v100.score_candidates(q, qxyz, cand, toks, offs, xyz, sub, GO, GE)
    r2 = ss.score_candidates(q, qxyz, cand, toks, offs, xyz, sub, GO, GE)
    for a, b in zip(r1, r2):
        assert np.array_equal(a, b)
    assert np.any(r1[1] > 0.5)      # some candidates really are superposable


def test_alignment_tm_a_matches_v100_bitwise():
    rng = np.random.default_rng(12)
    for k in range(200):
        n = int(rng.integers(3, 120))
        a = np.cumsum(rng.normal(size=(n, 3)) * 3.0, axis=0)
        b = a @ _rotation(rng).T + rng.normal(size=3) * 20.0
        b[rng.integers(0, n, n // 3)] += rng.normal(size=(n // 3, 3)) * (k % 10)
        bs = int(rng.integers(0, n))
        be = n if k % 3 == 0 else int(rng.integers(bs, n + 1))
        if k % 3 == 0:
            bs = 0                                    # block = every pair
        Lq, Lt = n + int(rng.integers(0, 50)), n + int(rng.integers(0, 50))
        assert v100.alignment_tm_a(a, b, Lq, Lt, bs, be) == \
            ss.alignment_tm_a(a, b, Lq, Lt, bs, be)
