"""Composition-bias correction: off by default, and consistent wherever scores are made."""
import numpy as np
import pytest

from rfoldseek import structure_score as ss
from rfoldseek.compbias import background_frequencies, query_profile
from rfoldseek.database import calibration_context_fingerprint
from rfoldseek.search import GAP_EXTEND, GAP_OPEN, load_matrix
from rfoldseek.smith_waterman import smith_waterman_numba


def test_window_zero_reproduces_the_plain_scores():
    rng = np.random.default_rng(0)
    sub = load_matrix()
    query = rng.integers(0, 80, 60)
    targets = [rng.integers(0, 80, n) for n in (40, 60, 90)]
    pos, prof = query_profile(query, sub, window=0)
    for target in targets:
        assert smith_waterman_numba(pos, target, prof, GAP_OPEN, GAP_EXTEND) == \
            smith_waterman_numba(query, target, sub, GAP_OPEN, GAP_EXTEND)

    xyz = [np.cumsum(rng.normal(size=(len(t), 3)) * 3, axis=0) for t in targets]
    qxyz = np.cumsum(rng.normal(size=(len(query), 3)) * 3, axis=0)
    offsets = np.r_[0, np.cumsum([len(t) for t in targets])].astype(np.int64)
    flat_t = np.concatenate(targets).astype(np.int64)
    flat_x = np.concatenate(xyz)
    cand = np.arange(len(targets), dtype=np.int64)
    plain = ss.score_candidates(query.astype(np.int64), qxyz, cand, flat_t, offsets,
                                flat_x, sub, np.float32(GAP_OPEN), np.float32(GAP_EXTEND))
    profiled = ss.score_candidates(pos, qxyz, cand, flat_t, offsets, flat_x, prof,
                                   np.float32(GAP_OPEN), np.float32(GAP_EXTEND))
    for a, b in zip(plain, profiled):
        assert np.array_equal(a, b)


def test_correction_discounts_a_shared_low_complexity_run():
    rng = np.random.default_rng(1)
    sub = load_matrix()
    helix = int(np.argmax(np.diag(sub)))          # a state that matches itself well
    def runny(n):
        t = np.full(n, helix)
        t[rng.choice(n, n // 8, replace=False)] = rng.integers(0, 80, n // 8)
        return t
    query, other = runny(70), runny(70)
    database = [rng.integers(0, 80, 100) for _ in range(50)] + [query, other]
    background = background_frequencies(database, 80)

    def relative(window):
        pos, prof = query_profile(query, sub, background, window)
        own = smith_waterman_numba(pos, query, prof, GAP_OPEN, GAP_EXTEND)
        return smith_waterman_numba(pos, other, prof, GAP_OPEN, GAP_EXTEND) / own

    assert relative(20) < 0.6 * relative(0)


def test_window_enters_the_scoring_fingerprint_only_when_used():
    sub = load_matrix()
    base = calibration_context_fingerprint("db", sub, GAP_OPEN, GAP_EXTEND)
    assert calibration_context_fingerprint("db", sub, GAP_OPEN, GAP_EXTEND, 0) == base
    assert calibration_context_fingerprint("db", sub, GAP_OPEN, GAP_EXTEND, 20) != base


def test_profile_rejects_a_negative_window():
    with pytest.raises(ValueError):
        query_profile([1, 2, 3], load_matrix(), window=-1)
