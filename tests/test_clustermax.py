"""Cluster-max calibration: one draw per sequence cluster."""
import numpy as np
import pytest

from rfoldseek import evalue


@pytest.fixture
def toks():
    """Ten distinct chains, each duplicated a different number of times."""
    rng = np.random.default_rng(0)
    base = [rng.integers(0, 20, size=120).astype(np.int64) for _ in range(10)]
    out, clusters = [], []
    for i, b in enumerate(base):
        for _ in range(1 + i):                 # cluster i has i+1 near-identical members
            t = b.copy()
            t[rng.integers(0, len(t), size=3)] = rng.integers(0, 20, size=3)
            out.append(t)
            clusters.append(f"c{i}")
    return out, np.array(clusters, dtype=object)


@pytest.fixture
def sub():
    rng = np.random.default_rng(1)
    m = rng.normal(-0.3, 0.3, size=(20, 20)).astype(np.float32)
    m = (m + m.T) / 2
    np.fill_diagonal(m, 1.0)
    return m


def test_calibrate_clustermax_reports_the_cluster_count(toks, sub):
    chains, clusters = toks

    par = evalue.calibrate_clustermax(sub, chains, clusters, length=120, n_queries=8, seed=0)

    assert par["n_clusters"] == 10
    assert par["n_chains"] == len(chains)
    assert par["beta"] > 0


def test_a_cluster_contributes_one_draw_however_many_members(toks, sub):
    """Swamping one cluster must not move the fitted null.

    A per-pair fit would be dominated by the swamped cluster, since it would
    contribute one observation per member; taking the maximum per cluster
    leaves the sample size and the fit unchanged. The number of draws is the
    property that separates the two, so it is asserted directly.
    """
    chains, clusters = toks
    big = "c9"                                   # the largest cluster in the fixture
    swamp = [i for i, c in enumerate(clusters) if c == big]
    grown = list(chains) + [chains[i].copy() for i in swamp] * 40
    grown_cl = np.array(list(clusters) + [big] * (len(swamp) * 40), dtype=object)

    a = evalue.calibrate_clustermax(sub, chains, clusters, length=120, n_queries=8, seed=0)
    b = evalue.calibrate_clustermax(sub, grown, grown_cl, length=120, n_queries=8, seed=0)

    assert b["n_chains"] > 5 * a["n_chains"], "fixture did not actually swamp c0"
    assert b["n_clusters"] == a["n_clusters"]
    assert b["n_maxima"] == a["n_maxima"], "one draw per cluster, whatever its size"
    assert abs(b["beta"] - a["beta"]) < 0.05 * a["beta"]


def test_evalue_clustermax_scales_with_the_cluster_count(toks, sub):
    chains, clusters = toks
    par = evalue.calibrate_clustermax(sub, chains, clusters, length=120, n_queries=8, seed=0)

    e1 = evalue.evalue_clustermax(50.0, par["mu"], par["beta"], 10)
    e2 = evalue.evalue_clustermax(50.0, par["mu"], par["beta"], 100)

    assert e2 == pytest.approx(10 * e1, rel=1e-6)


def test_a_higher_score_is_more_significant(toks, sub):
    par = evalue.calibrate_clustermax(sub, *toks, length=120, n_queries=8, seed=0)

    lo = evalue.evalue_clustermax(20.0, par["mu"], par["beta"], 10)
    hi = evalue.evalue_clustermax(80.0, par["mu"], par["beta"], 10)

    assert hi < lo
