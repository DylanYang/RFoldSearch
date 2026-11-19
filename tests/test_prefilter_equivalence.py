"""The compiled prefilter must match the reference implementation exactly."""
import numpy as np
import pytest

from rfoldsearch import load_matrix
from rfoldsearch import prefilter as pf
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


@pytest.mark.parametrize("top_n", [0, 1, 7, 50, 200, 1000])
def test_partial_selection_matches_full_lexicographic_sort(world, top_n):
    """The top-N selection preserves the (score, index) ordering exactly."""
    toks, K, sub = world
    kernel = KmerPrefilter(toks, K, sub)
    idx, powers = ref.build_index(toks, K)
    expand = ref.make_expander(sub, K, powers, THRESH_FRAC * KMER_K)
    for qi in (1, 19, 77, 149):
        got_c, got_s = kernel.candidates(toks[qi], top_n, 1, self_idx=qi)
        exp_c, exp_s = ref.candidates(toks[qi], toks, idx, expand, 1, qi, top_n)
        assert np.array_equal(got_c, exp_c)
        assert np.array_equal(got_s, exp_s)


@pytest.mark.parametrize("hist_bytes", [1, 4096, 1 << 16])
def test_small_target_blocks(world, hist_bytes):
    """Many small target blocks (and stamp reuse across queries) change nothing."""
    toks, K, sub = world
    kernel = KmerPrefilter(toks, K, sub)
    kernel.hist_bytes = hist_bytes
    idx, powers = ref.build_index(toks, K)
    expand = ref.make_expander(sub, K, powers, THRESH_FRAC * KMER_K)
    for qi in range(0, 250, 13):
        got_c, got_s = kernel.candidates(toks[qi], 100, 2, self_idx=qi)
        exp_c, exp_s = ref.candidates(toks[qi], toks, idx, expand, 2, qi, 100)
        assert np.array_equal(got_c, exp_c)
        assert np.array_equal(got_s, exp_s)


def test_stamp_wraparound(world):
    """Histogram stamps wrapping past 16 bits are cleared, not misread."""
    toks, K, sub = world
    kernel = KmerPrefilter(toks, K, sub)
    kernel.hist_bytes = 1                      # one stamp per 8 targets
    base_c, base_s = kernel.candidates(toks[5], 100, 2, self_idx=5)
    kernel._stamp = pf._COUNT_MASK - 3         # wrap inside the next query
    got_c, got_s = kernel.candidates(toks[5], 100, 2, self_idx=5)
    assert np.array_equal(got_c, base_c) and np.array_equal(got_s, base_s)


def test_fallback_kernels_agree(world, monkeypatch):
    """Packed counting-sort fallback and the unpacked kernel give the same counts."""
    toks, K, sub = world
    packed = KmerPrefilter(toks, K, sub)
    monkeypatch.setattr(pf, "pack_postings", lambda index: None)
    unpacked = KmerPrefilter(toks, K, sub)
    assert "packed" in packed.index and "packed" not in unpacked.index
    for qi in (0, 42, 199):
        q = np.asarray(toks[qi], np.int64)
        codes, qps = pf.expand_query(q, packed.index["powers"], packed.sub,
                                     packed.diag, packed.thr)
        qps = qps.astype(np.int64)
        n_q = len(q) - KMER_K + 1
        a = packed._best_diagonals(codes, qps, qi, n_q).copy()
        b = unpacked._best_diagonals(codes, qps, qi, n_q).copy()
        ix = packed.index
        c = pf._best_diagonal_counts_packed(
            codes, qps, ix["offsets"], ix["packed"], ix["pbits"], qi,
            ix["n_targets"], packed._span, packed._seg, packed._best.copy(),
            packed._buf).copy()
        assert np.array_equal(a, b) and np.array_equal(a, c)
