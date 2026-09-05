"""The length table must not be flattened by default."""
import numpy as np
import pytest

from rfoldseek import evalue


@pytest.fixture
def toks():
    """Chains at a spread of lengths, drawn from one background."""
    rng = np.random.default_rng(0)
    out = []
    for L in (40, 80, 150, 300, 600):
        for _ in range(12):
            out.append(rng.integers(0, 20, size=L).astype(np.int64))
    return out


@pytest.fixture
def sub():
    rng = np.random.default_rng(1)
    m = rng.normal(-0.2, 0.3, size=(20, 20)).astype(np.float32)
    m = (m + m.T) / 2
    np.fill_diagonal(m, 1.0)
    return m


def test_lambda_is_not_forced_monotone_by_default(toks, sub):
    table = evalue.calibrate_length_dependent(
        sub, toks, lengths=(40, 80, 150, 300), n_pairs=120, long_n_pairs=120)

    lam = [r["lambda"] for r in table]
    assert "lambda_raw" not in table[0], "smoothing ran; monotone should default off"
    assert lam == [r["lambda"] for r in table]


def test_monotone_true_still_smooths_and_keeps_the_raw_values(toks, sub):
    table = evalue.calibrate_length_dependent(
        sub, toks, lengths=(40, 80, 150, 300), n_pairs=120, long_n_pairs=120,
        monotone=True)

    lam = [r["lambda"] for r in table]
    assert lam == sorted(lam, reverse=True), "monotone=True must give a non-increasing lambda"
    assert all("lambda_raw" in r for r in table)


def test_long_lengths_get_the_same_sample_size_by_default():
    import inspect
    p = inspect.signature(evalue.calibrate_length_dependent).parameters
    assert p["long_n_pairs"].default == p["n_pairs"].default
