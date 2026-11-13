import json

import numpy as np

from scripts.recheck_fresh_experimental_evalue_range import (
    audit_states, family_ci, family_mean, power_at_e0,
)


def test_power_fit_boundary_and_linear_tail():
    model = {"a": 2.0, "b": 0.5, "fit_max": 100.0}
    assert np.isclose(power_at_e0(model, 40, 100, 878), 20)
    assert np.isclose(power_at_e0(model, 40, 200, 878), 40)
    assert power_at_e0(model, 40, 10000, 878) == 878


def test_family_mean_balances_families_not_queries():
    values = np.array([[1, 2], [3, 4], [10, 20]], dtype=float)
    assert np.allclose(family_mean(values, np.array(["A", "A", "B"])), [6, 11.5])
    assert family_ci(np.array([1, 3, 10], dtype=float),
                     np.array(["A", "A", "B"]))["value"] == 6


def test_partial_audit_keeps_unknown_and_inclusive_boundary(tmp_path):
    table = tmp_path / "members.tsv"
    table.write_text("query_id\ttarget_id\ttm_query\ttm_target\tstatus\n"
                     "q\ta\t0.45\t0.45\tok\n"
                     "q\tb\t0.2\t0.3\tok\n", encoding="utf-8")
    table.with_suffix(".json").write_text(json.dumps({
        "database_sha256": "same-db", "threshold": .45,
        "completed_pairs": 1, "required_pairs": 3,
        "complete": False,
    }), encoding="utf-8")
    states, metadata = audit_states(table, ["q"], ["q", "a", "b", "missing"], "same-db")
    assert states.tolist() == [[0, 2, 1, 0]]
    assert metadata["observed_table_rows"] == 2
