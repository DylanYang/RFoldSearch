"""Experimental raw-SW target-aware E-value deployment checks."""
import json
import numpy as np
import pytest
import torch

from rfoldsearch import experimental_evalue as EV
from rfoldsearch.database import (calibration_context_fingerprint,
                                  database_fingerprint)
from rfoldsearch.search import GAP_EXTEND, GAP_OPEN, Searcher, load_matrix
from rfoldsearch.experimental_calibrate import _leader_tables, fit_from_matrices


def test_structural_exclusion_backfills_within_cluster():
    scores = np.array([[50., 40., 30., 20.]])
    query_indices = np.array([3])
    clusters = np.array(["a", "a", "b", "q"])
    family = np.array(["RF2", "RF2", "RF3", "RF1"])
    clan = np.array(["", "", "", ""])
    tm = np.array([[.7, .2, .1, np.nan]])
    leaders, annotated, before = _leader_tables(
        scores, query_indices, clusters, [[0, 1], [2], [3]],
        family, clan, tm)
    assert leaders.tolist() == [[1, 2, -1]]
    assert annotated.tolist() == [[1, 2, -1]]
    assert before.tolist() == [2]


def test_empirical_map_is_monotone_and_uses_declared_tails():
    fitted = EV.fit_empirical([np.array([.1, .3, 1., 3., 10.]),
                               np.array([.2, .4, 2., 4., 20.])],
                              np.array(["RF1", "RF2"]), np.ones(2), 100)
    values = EV.apply_empirical(np.array([0., .01, .1, 1., 20., 21., 10000.]), fitted)
    assert np.all(np.diff(values) >= 0)
    assert values[0] == 0
    assert values[-1] == 100
    assert values[-2] == 100
    assert values[1] == pytest.approx(
        fitted["observed_counts"][0] * (.01 / .1) ** fitted["low_tail_power_b"])
    assert EV.apply_empirical(1e-300, fitted) > 0


def test_fitter_uses_clean_background_for_both_stages():
    rng = np.random.default_rng(22)
    tokens = [rng.integers(0, 20, size=30 + i).tolist() for i in range(24)]
    entries = [{"id": f"t{i}", "tokens": token} for i, token in enumerate(tokens)]
    ids = [entry["id"] for entry in entries]
    clusters = np.asarray([f"c{i}" for i in range(24)])
    family = np.asarray([f"RF{i % 4}" for i in range(24)])
    clan = np.asarray([""] * 24)
    scores = np.maximum(rng.gumbel(12, 3, (24, 24)), 0)
    np.fill_diagonal(scores, 100)
    tm = np.full((24, 24), .1)
    tm[0, 1] = .8
    model, calibration, audit = fit_from_matrices(
        entries, tokens, ids, clusters, family, clan, np.arange(24),
        scores, tm, log=lambda _: None)
    assert model["n_observations"] == 24 * 18 - 1
    assert calibration["n_queries"] == 24
    assert audit["n_retained_background"] == 24 * 18 - 1


def test_searcher_uses_matching_experimental_empirical_parameters(tmp_path):
    entries = [{"id": f"t{i}", "tokens": [i % 5, (i + 1) % 5] * 15,
                "cluster": f"c{i}"} for i in range(5)]
    target = database_fingerprint(entries, [e["cluster"] for e in entries], 80)
    context = calibration_context_fingerprint(
        target, load_matrix(), GAP_OPEN, GAP_EXTEND)
    params = {
        "scheme": EV.SCHEME, "score_definition": "raw-sw", "n_clusters": 5,
        "target_database_fingerprint": target,
        "calibration_context_fingerprint": context,
        "validation_status": "audited-e0-le-100",
        "target_aware_gumbel": {
            "coefficients_mu_logbeta": [[8., 0.], [0., 0.], [0., 0.],
                                         [0., 0.], [0., 0.]],
            "feature_mean": [3., 3., 0., 1.],
            "feature_scale": [1., 1., 1., 1.],
            "cluster_median_lengths": [30.], "cluster_multiplicities": [5],
        },
        "empirical_calibration": {
            "raw_evalues": [.01, 1.], "observed_counts": [.02, 1.],
            "low_tail_power_b": 1., "n_clusters": 5,
        },
    }
    db = tmp_path / "db.pt"
    torch.save({"n_codes": 80, "entries": entries, "n_clusters": 5,
                "evalue_params": params}, db)
    searcher = Searcher(str(db))
    hit = searcher.search(np.asarray(entries[0]["tokens"]),
                          rerank="sw", top_k=1)[0]
    assert hit["raw_evalue"] == pytest.approx(EV.initial_evalue(
        hit["sw_score"], 30, EV.token_entropy(entries[0]["tokens"]),
        params["target_aware_gumbel"]))
    assert hit["evalue"] == pytest.approx(EV.apply_empirical(
        hit["raw_evalue"], params["empirical_calibration"]))
    params["target_database_fingerprint"] = "wrong"
    bad = tmp_path / "wrong.json"
    bad.write_text(json.dumps(params))
    with pytest.raises(ValueError, match="different target database"):
        Searcher(str(db), evalue_params_path=str(bad))
