"""The compiled prefilter must match the reference implementation exactly."""
import numpy as np
import pytest

from rfoldseek import load_matrix
from rfoldseek.prefilter import KMER_K, THRESH_FRAC, KmerPrefilter
from rfoldseek import prefilter_reference as ref


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
