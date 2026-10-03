"""Structure-aware ranking: traceback SW, alignment TM-score, and the atm backend."""
import numpy as np
import pytest
import torch

from rfoldseek import structure_score as ss
from rfoldseek import evalue as EV
from rfoldseek.search import GAP_EXTEND, GAP_OPEN, Searcher, load_matrix
from rfoldseek.smith_waterman import smith_waterman_numba
from helpers import synthetic_evalue_params

pytestmark = pytest.mark.filterwarnings(
    "ignore:E-value parameters were calibrated on .* clusters:UserWarning")

GO, GE = np.float32(GAP_OPEN), np.float32(GAP_EXTEND)


def _rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _path_score(q, t, qa, ta, sub):
    """Score of an alignment path under the affine gap model."""
    s = float(sub[q[qa], t[ta]].sum())
    for k in range(1, len(qa)):
        for g in (qa[k] - qa[k - 1] - 1, ta[k] - ta[k - 1] - 1):
            if g > 0:
                s += GAP_OPEN + (g - 1) * GAP_EXTEND
    return s


def test_traceback_matches_deployed_score_and_path():
    rng = np.random.default_rng(0)
    sub = load_matrix()
    base = rng.integers(0, 80, size=90)
    for _ in range(50):
        q = base.copy()
        q[rng.integers(0, 90, 30)] = rng.integers(0, 80, 30)
        t = np.concatenate([rng.integers(0, 80, rng.integers(0, 20)), base[rng.integers(0, 20):],
                            rng.integers(0, 80, rng.integers(0, 20))])
        t[rng.integers(0, len(t), 20)] = rng.integers(0, 80, 20)
        score, qa, ta = ss.smith_waterman_traceback(q, t, sub, GO, GE)
        assert score == pytest.approx(smith_waterman_numba(q, t, sub, GAP_OPEN, GAP_EXTEND), abs=1e-3)
        assert np.all(np.diff(qa) > 0) and np.all(np.diff(ta) > 0)
        assert _path_score(q, t, qa, ta, sub) == pytest.approx(score, abs=1e-3)


def test_alignment_tm_is_invariant_to_rigid_motion():
    rng = np.random.default_rng(1)
    a = np.cumsum(rng.normal(size=(80, 3)) * 3.0, axis=0)
    b = a @ _rotation(rng).T + rng.normal(size=3) * 20.0
    assert ss.alignment_tm(a, b, 80, 80) == pytest.approx(1.0, abs=1e-9)
    # half of each chain aligned: each normalisation gives 1/2
    assert ss.alignment_tm(a[:40], b[:40], 80, 80) == pytest.approx(0.5, abs=1e-9)
    # a longer target lowers the target-normalised term only
    assert ss.alignment_tm(a, b, 80, 160) == pytest.approx(0.75, abs=1e-9)


def test_alignment_tm_is_not_distorted_by_a_misaligned_segment():
    rng = np.random.default_rng(2)
    a = np.cumsum(rng.normal(size=(100, 3)) * 3.0, axis=0)
    b = a @ _rotation(rng).T
    b[80:] += rng.normal(size=(20, 3)) * 30.0          # last fifth displaced
    assert ss.alignment_tm(a, b, 100, 100) == pytest.approx(0.8, abs=0.02)


@pytest.fixture
def db_path(tmp_path):
    """40 chains: token variants of one chain; coordinates of the first 20 are
    rigid copies of one fold, of the rest random walks."""
    rng = np.random.default_rng(3)
    base = rng.integers(0, 80, size=120).astype(np.int16)
    fold = np.cumsum(rng.normal(size=(120, 3)) * 3.0, axis=0)
    entries = []
    for i in range(40):
        t = base.copy()
        flips = rng.integers(0, len(t), size=rng.integers(0, 40))
        t[flips] = rng.integers(0, 80, size=len(flips))
        xyz = fold @ _rotation(rng).T if i < 20 else np.cumsum(rng.normal(size=(120, 3)) * 3.0, axis=0)
        coords = np.repeat(xyz[:, None, :], 5, axis=1).astype(np.float32)
        entries.append({"id": f"chain_{i:03d}.pdb", "tokens": t, "coords": coords,
                        "mask": np.ones(len(t), dtype=bool), "n_valid": len(t)})
    p = tmp_path / "tiny_db.pt"
    torch.save({"n_codes": 80, "entries": entries, "checkpoint": "synthetic",
                "evalue_params": synthetic_evalue_params()}, p)
    return str(p)


def test_atm_is_the_default_and_ranks_by_s_times_atm(db_path):
    s = Searcher(db_path)
    hits = s.search_id("chain_000", top_k=39, top_n=39)
    assert hits[0]["target"] != "chain_000"
    assert all({"atm", "structure_score", "aligned", "sw_score", "evalue"} <= set(h) for h in hits)
    assert [h["structure_score"] for h in hits] == sorted((h["structure_score"] for h in hits), reverse=True)
    for h in hits:
        assert h["structure_score"] == pytest.approx(h["sw_score"] * h["atm"])
        assert 0 < h["aligned"] <= 120
    # same fold ranks above random coordinates
    fold = {f"chain_{i:03d}" for i in range(1, 20)}
    top = {h["target"] for h in hits[:19]}
    assert len(top & fold) >= 17


def test_bits_and_evalue_are_those_of_the_smith_waterman_score(db_path):
    s = Searcher(db_path)
    q = np.asarray(s.toks[0], np.int64)
    mu, beta = EV.clustermax_params_for_query(s.ev_par, q)
    for h in s.search_id("chain_000", top_k=39, top_n=39):
        t = s.toks[s.id2i[h["target"]]]
        sw = smith_waterman_numba(q, t, s.sub, GAP_OPEN, GAP_EXTEND)
        assert h["sw_score"] == pytest.approx(sw, abs=1e-3)
        assert h["bits"] == pytest.approx(EV.bit_score_clustermax(sw, mu, beta), abs=1e-3)
        assert h["evalue"] == pytest.approx(EV.evalue_clustermax(sw, mu, beta, s.n_clusters), rel=1e-4)


def test_missing_query_coordinates_are_an_error(db_path):
    s = Searcher(db_path)
    with pytest.raises(ValueError, match="query's C4' coordinates"):
        s.search(np.asarray(s.toks[0], np.int64), top_k=5)


def test_missing_database_coordinates_are_an_error(tmp_path):
    rng = np.random.default_rng(5)
    tok = rng.integers(0, 80, size=50).astype(np.int16)        # identical chains: candidates exist
    entries = [{"id": f"c{i}", "tokens": tok.copy()} for i in range(5)]
    p = tmp_path / "nocoords.pt"
    torch.save({"n_codes": 80, "entries": entries, "checkpoint": "synthetic"}, p)
    s = Searcher(str(p))
    with pytest.raises(ValueError, match="every database chain"):
        s.search_id("c0", top_k=2)
