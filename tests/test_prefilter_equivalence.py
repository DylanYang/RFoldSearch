"""The compiled prefilter must match the reference implementation exactly."""
import numpy as np
import pytest

from rfoldsearch import load_matrix
from rfoldsearch.prefilter import KMER_K, THRESH_FRAC, KmerPrefilter
from rfoldsearch import prefilter_reference as ref


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(7)
    K = 80
    toks = [rng.integers(0, K, size=rng.integers(20, 300)).astype(np.int64)
            for _ in range(250)]
    sub = load_matrix()
    return toks, K, sub


@pytest.mark.parametrize("qi", [0, 3, 17, 42, 99, 200])
@pytest.mark.parametrize("top_n", [50, 200])
def test_identical_candidates(world, qi, top_n):
    toks, K, sub = world
    thr = THRESH_FRAC * KMER_K
    kernel = KmerPrefilter(toks, K, sub)
    idx, powers = ref.build_index(toks, K)
    expand = ref.make_expander(sub, K, powers, thr)

    got_c, got_s = kernel.candidates(toks[qi], top_n, 2, self_idx=qi)
    exp_c, exp_s = ref.candidates(toks[qi], toks, idx, expand, 2, qi, top_n)
    assert np.array_equal(got_c, exp_c)
    assert np.array_equal(got_s, exp_s)


def test_query_longer_than_targets(world):
    toks, K, sub = world
    rng = np.random.default_rng(11)
    long_q = rng.integers(0, K, size=900).astype(np.int64)
    kernel = KmerPrefilter(toks, K, sub)
    idx, powers = ref.build_index(toks, K)
    expand = ref.make_expander(sub, K, powers, THRESH_FRAC * KMER_K)
    got_c, got_s = kernel.candidates(long_q, 100, 2, self_idx=-1)
    exp_c, exp_s = ref.candidates(long_q, toks, idx, expand, 2, -1, 100)
    assert np.array_equal(got_c, exp_c)
    assert np.array_equal(got_s, exp_s)


def test_repeated_codes_match_reference():
    """Helix-like chains: long homopolymer runs make the same 4-mer recur at
    many query positions, which the kernel groups."""
    rng = np.random.default_rng(23)
    K = 80
    helix = [62, 24, 30, 22]
    toks = []
    for _ in range(200):
        parts = []
        while sum(len(p) for p in parts) < rng.integers(30, 250):
            if rng.random() < 0.6:
                parts.append(np.full(rng.integers(4, 15), rng.choice(helix)))
            else:
                parts.append(rng.integers(0, K, size=rng.integers(1, 6)))
        toks.append(np.concatenate(parts).astype(np.int64))
    sub = load_matrix()
    thr = THRESH_FRAC * KMER_K
    kernel = KmerPrefilter(toks, K, sub)
    idx, powers = ref.build_index(toks, K)
    expand = ref.make_expander(sub, K, powers, thr)
    for qi in (0, 5, 50, 120, 199):
        got_c, got_s = kernel.candidates(toks[qi], 100, 2, self_idx=qi)
        exp_c, exp_s = ref.candidates(toks[qi], toks, idx, expand, 2, qi, 100)
        assert np.array_equal(got_c, exp_c)
        assert np.array_equal(got_s, exp_s)
