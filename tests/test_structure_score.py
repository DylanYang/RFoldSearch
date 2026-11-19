"""Traceback SW, TM_a, and the default structure-score backend."""
import numpy as np
import pytest
import torch

from rfoldsearch import structure_score as ss
from rfoldsearch.search import GAP_EXTEND, GAP_OPEN, Searcher, load_matrix
from rfoldsearch.smith_waterman import smith_waterman_numba

GO, GE = np.float32(GAP_OPEN), np.float32(GAP_EXTEND)


def _rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _path_score(q, t, qa, ta, sub):
    score = float(sub[q[qa], t[ta]].sum())
    for k in range(1, len(qa)):
        for gap in (qa[k] - qa[k - 1] - 1, ta[k] - ta[k - 1] - 1):
            if gap > 0:
                score += GAP_OPEN + (gap - 1) * GAP_EXTEND
    return score


def test_traceback_matches_deployed_score_and_path():
    rng = np.random.default_rng(0)
    sub = load_matrix()
    base = rng.integers(0, 80, size=90)
    for _ in range(10):
        q = base.copy()
        q[rng.integers(0, 90, 30)] = rng.integers(0, 80, 30)
        t = base.copy()
        t[rng.integers(0, 90, 20)] = rng.integers(0, 80, 20)
        score, qa, ta = ss.smith_waterman_traceback(q, t, sub, GO, GE)
        expected = smith_waterman_numba(q, t, sub, GAP_OPEN, GAP_EXTEND)
        assert score == pytest.approx(expected, abs=1e-3)
        assert np.all(np.diff(qa) > 0) and np.all(np.diff(ta) > 0)
        assert _path_score(q, t, qa, ta, sub) == pytest.approx(score, abs=1e-3)


def test_alignment_tm_a_is_rigid_motion_invariant_and_coverage_aware():
    rng = np.random.default_rng(1)
    a = np.cumsum(rng.normal(size=(80, 3)) * 3.0, axis=0)
    b = a @ _rotation(rng).T + rng.normal(size=3) * 20.0
    assert ss.alignment_tm_a(a, b, 80, 80) == pytest.approx(1.0, abs=1e-9)
    assert ss.alignment_tm_a(a[:40], b[:40], 80, 80) == pytest.approx(
        0.5, abs=1e-9)
    assert ss.alignment_tm_a(a, b, 80, 160) == pytest.approx(0.75, abs=1e-9)


def test_alignment_tm_a_refines_around_a_misaligned_segment():
    rng = np.random.default_rng(2)
    a = np.cumsum(rng.normal(size=(100, 3)) * 3.0, axis=0)
    b = a @ _rotation(rng).T
    b[80:] += rng.normal(size=(20, 3)) * 30.0
    assert ss.alignment_tm_a(a, b, 100, 100) == pytest.approx(0.8, abs=0.02)


@pytest.fixture
def db_path(tmp_path):
    rng = np.random.default_rng(3)
    base = rng.integers(0, 80, size=120).astype(np.int16)
    fold = np.cumsum(rng.normal(size=(120, 3)) * 3.0, axis=0)
    entries = []
    for i in range(40):
        tokens = base.copy()
        flips = rng.integers(0, len(tokens), size=rng.integers(0, 40))
        tokens[flips] = rng.integers(0, 80, size=len(flips))
        xyz = (fold @ _rotation(rng).T if i < 20 else
               np.cumsum(rng.normal(size=(120, 3)) * 3.0, axis=0))
        coords = np.repeat(xyz[:, None, :], 5, axis=1).astype(np.float32)
        entries.append({
            "id": f"chain_{i:03d}.pdb",
            "tokens": tokens,
            "coords": coords,
            "mask": np.ones(len(tokens), dtype=bool),
            "n_valid": len(tokens),
        })
    path = tmp_path / "tiny_db.pt"
    torch.save({"n_codes": 80, "entries": entries,
                "checkpoint": "synthetic"}, path)
    return str(path)


def test_tm_a_is_default_and_ranks_by_product(db_path):
    hits = Searcher(db_path).search_id("chain_000", top_k=39, top_n=39)
    scores = [hit["structure_score"] for hit in hits]
    assert scores == sorted(scores, reverse=True)
    for hit in hits:
        assert hit["structure_score"] == pytest.approx(
            hit["sw_score"] * hit["tm_a"])
        assert 0 < hit["aligned"] <= 120
    same_fold = {f"chain_{i:03d}" for i in range(1, 20)}
    assert len({hit["target"] for hit in hits[:19]} & same_fold) >= 17


def test_sw_backend_is_gone(db_path):
    with pytest.raises(ValueError, match="unknown rerank backend"):
        Searcher(db_path).search_id("chain_000", rerank="sw")


def test_tm_a_requires_database_coordinates(tmp_path):
    tokens = np.arange(40, dtype=np.int16) % 10
    entries = [{"id": f"c{i}", "tokens": tokens.copy()} for i in range(5)]
    path = tmp_path / "token_only.pt"
    torch.save({"n_codes": 80, "entries": entries,
                "checkpoint": "synthetic"}, path)
    with pytest.raises(ValueError, match="every database chain"):
        Searcher(str(path)).search_id("c0", top_k=2)


def _batched_world(rng):
    lens = np.r_[1, 2, 3, 5, rng.integers(4, 260, size=60)]
    base = rng.integers(0, 80, size=300)
    toks, xyz = [], []
    for n in lens:
        t = base[:n].copy()
        k = max(1, n // 4)
        t[rng.integers(0, n, k)] = rng.integers(0, 80, k)
        toks.append(t)
        xyz.append(np.cumsum(rng.normal(scale=3.0, size=(n, 3)), axis=0))
    offsets = np.zeros(len(toks) + 1, np.int64)
    offsets[1:] = np.cumsum(lens)
    return base, np.concatenate(toks).astype(np.int64), offsets, np.concatenate(xyz)


def _query(rng, base, qn):
    q = base[:qn].copy()
    q[rng.integers(0, qn, qn // 5)] = rng.integers(0, 80, qn // 5)
    return q, np.cumsum(rng.normal(scale=3.0, size=(qn, 3)), axis=0)


@pytest.mark.parametrize("parallel", [False, True])
# shipped batching; one pair per batch; a budget that mixes lane counts
@pytest.mark.parametrize("lanes,budget", [(ss.LANES, ss.PTR_BUDGET), (ss.LANES, 1),
                                          (3, 150_000)])
def test_batched_scorer_matches_score_pair_bitwise(parallel, lanes, budget):
    """The lane-batched scorer equals score_pair per candidate, bit for bit,
    whatever the number of lanes per batch."""
    rng = np.random.default_rng(3)
    sub = load_matrix()
    base, flat_t, offsets, flat_x = _batched_world(rng)
    for qn in (4, 37, 150, 290):
        q, qx = _query(rng, base, qn)
        cand = rng.permutation(len(offsets) - 1).astype(np.int64)[:61]
        scorer = (ss._score_candidates_parallel if parallel
                  else ss._score_candidates_serial)
        S, T, A = scorer(q, qx, cand, flat_t, offsets, flat_x, sub, GO, GE,
                         lanes, budget)
        for k, c in enumerate(cand):
            lo, hi = offsets[c], offsets[c + 1]
            s, tm, a = ss.score_pair(q, qx, flat_t[lo:hi], flat_x[lo:hi], sub, GO, GE)
            assert S[k] == s and T[k] == tm and A[k] == a


def test_batch_bounds_respect_lanes_and_budget():
    rng = np.random.default_rng(5)
    lens = rng.integers(1, 3000, size=500).astype(np.int64)
    order = np.argsort(lens, kind="mergesort")
    for Lq, lanes, budget in ((50, 8, 1 << 25), (2800, 8, 1 << 25), (4000, 8, 1),
                              (300, 5, 1_000_000)):
        b = ss._batch_bounds(Lq, lens, order, lanes, budget)
        assert b[0] == 0 and b[-1] == len(lens) and np.all(np.diff(b) >= 1)
        for lo, hi in zip(b[:-1], b[1:]):
            assert hi - lo <= lanes
            width = lens[order[lo:hi]].max()
            assert hi - lo == 1 or (Lq + 1) * (width + 1) * (hi - lo) <= budget


@pytest.mark.parametrize("dtype,gaps", [(np.float64, (GO, GE)),
                                        (np.float32, (float(GAP_OPEN), float(GAP_EXTEND)))])
def test_non_float32_inputs_match_score_pair(dtype, gaps):
    """score_candidates with a float64 matrix or Python-float gaps scores pair
    by pair, so it still equals score_pair given the same inputs."""
    rng = np.random.default_rng(4)
    sub = load_matrix().astype(dtype)
    go, ge = gaps
    base, flat_t, offsets, flat_x = _batched_world(rng)
    q, qx = _query(rng, base, 120)
    cand = rng.permutation(len(offsets) - 1).astype(np.int64)[:40]
    S, T, A = ss.score_candidates(q, qx, cand, flat_t, offsets, flat_x, sub, go, ge)
    for k, c in enumerate(cand):
        lo, hi = offsets[c], offsets[c + 1]
        s, tm, a = ss.score_pair(q, qx, flat_t[lo:hi], flat_x[lo:hi], sub, go, ge)
        assert S[k] == s and T[k] == tm and A[k] == a
