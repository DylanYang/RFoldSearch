import numpy as np
import pytest
from scipy.stats import gumbel_r

from scripts.fresh_experimental_evalue_compare import (
    apply_power, e0_for_scores, family_weights, features, fit_gumbel, fit_power,
    group_folds, log_gumbel_survival,
    select_leaders, positive_leaders,
)


def test_group_folds_keeps_family_and_clan_together():
    family = np.array(["A", "A", "B", "C", "D", "E", "F"])
    clan = np.array(["1", "", "1", "", "", "", ""])
    folds = group_folds(family, clan)
    assert folds[0] == folds[1] == folds[2]
    assert set(folds) == set(range(4))


def test_gumbel_tail_calculation_matches_scipy():
    z = np.array([-10., -1., 0., 1., 5., 20., 100.])
    np.testing.assert_allclose(log_gumbel_survival(z),
                               gumbel_r.logsf(z), atol=1e-7)
    model = {"coefficients": [[1., 0, 0, 0, 0], [0., 0, 0, 0, 0]],
             "mean": [0., 0, 0, 0], "scale": [1., 1., 1., 1.]}
    x = np.zeros((3, 4))
    np.testing.assert_allclose(e0_for_scores(np.array([1., 2.]), x, model),
                               3 * gumbel_r.sf(np.array([0., 1.])), rtol=1e-7)


def test_censored_conditional_fit_and_power_monotonic():
    rng = np.random.default_rng(27)
    n_query, n_cluster = 40, 40
    qlength = rng.integers(25, 250, n_query).astype(float)
    clength = rng.integers(25, 250, n_cluster).astype(float)
    h = rng.uniform(1, 3, n_query)
    x = features(qlength, h, clength)
    mu = 2 + .2 * (x[:, :, 0] - x[:, :, 1])
    y = rng.gumbel(mu, .5)
    leaders = np.broadcast_to(np.arange(n_cluster), y.shape).copy()
    families = np.asarray([str(i//2) for i in range(n_query)])
    model = fit_gumbel(y, leaders, x, families, qlength, clength,
                       np.arange(n_query), n_query, True)
    assert model["censored"] and model["n_tail_observations"] > 0
    assert model["max_abs_gradient"] < 2e-3
    e0 = e0_for_scores(np.array([1., 2., 3.]), x[0], model)
    assert np.all(np.diff(e0) < 0)
    power = {"a": .5, "b": 1.2, "fit_max": 100}
    assert np.all(np.diff(apply_power(e0, power, qlength[0], n_cluster)) < 0)
    assert np.isclose(family_weights(families).sum(), 1)


def test_global_and_four_bin_power_fit():
    families = np.asarray([str(i//4) for i in range(80)])
    lengths = np.linspace(25., 400., 80)
    values = [np.geomspace(.001 * (1 + i/80), 100, 100) for i in range(80)]
    global_model = fit_power(values, families, lengths, np.ones(80), False)
    bin_model = fit_power(values, families, lengths, np.ones(80), True)
    assert global_model["a"] > 0 and global_model["b"] > 0
    assert len(bin_model["bins"]) == 4
    assert all(item["a"] > 0 for item in bin_model["bins"])


def test_structural_exclusion_backfills_member_not_entire_cluster():
    ids = ["q", "related", "unrelated", "empty"]
    labels = {"q": ("Q", ""), "related": ("T", ""),
              "unrelated": ("T", ""), "empty": ("U", "")}
    scores = np.array([[100., 90., 30., 60.]])
    state = np.array([[0, 2, 1, 2]], dtype=np.uint8)
    result = select_leaders(scores, np.array([0]), ids, [0, 1, 1, 2],
                            [[0], [1, 2], [3]], labels, True, state)
    assert result.tolist() == [[-1, 2, -1]]
    before = select_leaders(scores, np.array([0]), ids, [0, 1, 1, 2],
                            [[0], [1, 2], [3]], labels, True)
    assert (before >= 0).sum() == 2  # Bq must remain 2, not 1 after cleaning.


def test_missing_audit_is_not_unrelated_and_threshold_is_inclusive():
    with pytest.raises(ValueError, match="missing structural"):
        select_leaders(np.array([[10., 5.]]), np.array([0]), ["q", "t"],
                        [0, 1], [[0], [1]], {"q": ("A", ""), "t": ("B", "")},
                        False, np.array([[0, 0]], dtype=np.uint8))


def test_structural_homolog_selection_filters_before_cluster_max():
    result = positive_leaders(np.array([[10., 9., 5.]]), np.array([0]),
        ["q", "a", "b"], [0, 1, 1], [[0], [1, 2]],
        {k: ("F", "") for k in ("q", "a", "b")}, np.array([[0, 1, 2]]))
    assert result.tolist() == [[-1, 2]]


def test_zero_background_queries_remain_in_family_average():
    from scripts.fresh_experimental_evalue_compare import mean_family_curve
    curve = mean_family_curve([{"family": "F", "count": [0.]},
                               {"family": "F", "count": [2.]}],
                               "count", np.array([1.]))
    assert curve["observed"] == [1.]


def test_audit_covers_auxiliary_and_all_fallback_members():
    from scripts.audit_clean_experimental_background import required_pairs, checked_output
    ids = ["q", "aux", "related", "fallback", "clan"]
    labels = {"q": ("Q", "C"), "aux": ("", ""), "related": ("T", ""),
              "fallback": ("T", ""), "clan": ("U", "C")}
    pairs = set(required_pairs(["q", "aux"], ids, [0, 1, 2, 2, 3], labels))
    assert ("q", "related") in pairs and ("q", "fallback") in pairs
    assert ("aux", "clan") in pairs and ("q", "clan") not in pairs
    with pytest.raises(RuntimeError, match="returned"):
        checked_output("", [2])


def test_audit_loader_requires_matching_database_and_inclusive_cutoff(tmp_path):
    import csv, json
    from scripts.fresh_experimental_evalue_compare import file_sha256, load_structural_state
    db, table = tmp_path / "db.pt", tmp_path / "audit.tsv"
    db.write_bytes(b"frozen-db")
    with table.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["query_id", "target_id", "tm_query", "tm_target", "status"])
        writer.writerow(["q", "t", .45, .45, "ok"])
    metadata = {"complete": True, "threshold": .45,
                "database_sha256": file_sha256(db), "table_sha256": file_sha256(table)}
    table.with_suffix(".json").write_text(json.dumps(metadata))
    assert load_structural_state(table, ["q"], ["q", "t"], db).tolist() == [[0, 2]]
    db.write_bytes(b"other-db")
    with pytest.raises(ValueError, match="different database"):
        load_structural_state(table, ["q"], ["q", "t"], db)
