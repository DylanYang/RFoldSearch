"""Target-aware E-value parameters are the only supported search calibration."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rfoldsearch import evalue as EV
from rfoldsearch.database import (assign_clusters, attach_evalue_calibration,
                                calibration_context_fingerprint,
                                database_fingerprint, read_clusters)
from rfoldsearch.search import Searcher, asset
from helpers import add_coords, synthetic_evalue_params


def test_read_clusters_parses_a_cdhit_clstr(tmp_path):
    path = tmp_path / "nr80.clstr"
    path.write_text(">Cluster 0\n0\t120nt, >chainA... *\n"
                    "1\t118nt, >chainB... at +/98.00%\n"
                    ">Cluster 1\n0\t90nt, >chainC... *\n")
    assert read_clusters(str(path)) == {"chainA": 0, "chainB": 0, "chainC": 1}


def test_unmapped_chains_keep_distinct_clusters():
    entries = [{"id": "/data/chainA.cif"}, {"id": "/data/orphan1.cif"},
               {"id": "/data/orphan2.cif"}]
    labels = assign_clusters(entries, {"chainA": 0})
    assert len(set(labels)) == 3


def test_fingerprint_tracks_tokens_membership_and_score_context():
    entries = [{"id": "a", "tokens": [1, 2, 3]},
               {"id": "b", "tokens": [3, 2, 1]}]
    first = database_fingerprint(entries, ["x", "x"], 80)
    assert first == database_fingerprint(entries, [10, 10], 80)
    assert first != database_fingerprint(entries, ["x", "y"], 80)
    assert first != database_fingerprint(
        [entries[0], {"id": "b", "tokens": [3, 2, 2]}], ["x", "x"], 80)
    matrix = np.eye(4, dtype=np.float32)
    assert calibration_context_fingerprint(first, matrix, -2.0, -0.5) != (
        calibration_context_fingerprint(first, matrix, -3.0, -0.5))


def test_cluster_median_predictor_and_actual_target_score_length():
    params = synthetic_evalue_params([
        {"id": "a", "tokens": [0] * 10, "cluster": "c0"},
        {"id": "b", "tokens": [0] * 10, "cluster": "c0"},
        {"id": "c", "tokens": [0] * 100, "cluster": "c1"},
    ])
    params["target_aware_gumbel"]["coefficients_mu_logbeta"][2][0] = 1.0
    query = np.array([0] * 20)
    score = 2.0
    expected = sum(-np.expm1(-np.exp(-(score - (0.5 + np.log(length)))))
                   for length in (10.0, 100.0))
    assert EV.target_aware_initial_evalue(params, query, score) == pytest.approx(expected)
    assert EV.target_aware_evalue(params, query, 40.0, 100) == pytest.approx(expected)
    assert EV.target_aware_evalue(params, query, 20.0, 10) == pytest.approx(expected)


def test_power_extrapolation_is_proportional_and_capped():
    calibration = {
        "a": 3.0068629485039857e-06, "b": 3.9894187160910453,
        "fit_raw_evalue_max": 100.0, "n_clusters": 15271,
        "tail_rule": EV.TAIL_RULE,
    }
    raw = np.array([0.0, 50.0, 100.0, 200.0, 15271.0])
    expected = calibration["a"] * np.minimum(raw, 100.0) ** calibration["b"]
    expected *= np.maximum(1.0, raw / 100.0)
    expected = np.minimum(expected, 15271)
    assert EV.apply_power_calibration(raw, calibration) == pytest.approx(expected)
    assert EV.apply_power_calibration(200.0, calibration) == pytest.approx(expected[3])
    with pytest.raises(ValueError, match="extrapolation"):
        EV.apply_power_calibration(1.0, {**calibration, "tail_rule": None})


def test_no_packaged_evalue_asset_or_legacy_calculators():
    assert not Path(asset("evalue_params.json")).exists()
    assert not hasattr(EV, "evalue_clustermax")
    assert not hasattr(EV, "clustermax_params_for_query")
    assert not hasattr(EV, "fit_power_calibration")


def test_search_without_parameters_does_not_report_evalues(tmp_path):
    entries = [{"id": "a", "tokens": [1, 2, 3, 4] * 10},
               {"id": "b", "tokens": [1, 2, 3, 4] * 10}]
    db = tmp_path / "plain.pt"
    torch.save({"n_codes": 80, "entries": entries}, db)
    hits = Searcher(str(db)).search_id("a", rerank="sw", top_k=1)
    assert hits and "evalue" not in hits[0] and "raw_evalue" not in hits[0]


@pytest.mark.parametrize("change,match", [
    (lambda p: p.update(query_parameter_model={}), "obsolete calibration"),
    (lambda p: p.pop("target_aware_gumbel"), "only target-aware"),
    (lambda p: p.update(validation_status="provisional"), "unvalidated"),
    (lambda p: p.update(target_database_fingerprint="wrong"), "different database"),
    (lambda p: p.update(calibration_context_fingerprint="wrong"), "scoring context"),
    (lambda p: p["target_aware_gumbel"].update(cluster_median_lengths=[99.0]),
     "target-cluster lengths"),
])
def test_searcher_rejects_invalid_or_legacy_assets(tmp_path, change, match):
    entries = [{"id": "a", "tokens": [1] * 20, "cluster": "c0"}]
    params = synthetic_evalue_params(entries)
    change(params)
    db = tmp_path / "db.pt"
    torch.save({"n_codes": 80, "entries": entries, "evalue_params": params}, db)
    with pytest.raises(ValueError, match=match):
        Searcher(str(db))


def test_attach_and_search_use_only_new_parameters(tmp_path):
    entries = add_coords([
        {"id": "chain0", "tokens": [1, 2, 3, 4] * 10},
        {"id": "chain1", "tokens": [1, 2, 3, 4] * 10},
        {"id": "chain2", "tokens": [4, 3, 2, 1] * 10},
    ])
    source = tmp_path / "source.pt"
    torch.save({"n_codes": 80, "entries": entries}, source)
    clusters = tmp_path / "nr80.clstr"
    clusters.write_text(">Cluster 0\n0\t40nt, >chain0... *\n"
                        "1\t40nt, >chain1... at +/95.00%\n"
                        ">Cluster 1\n0\t40nt, >chain2... *\n")
    params = synthetic_evalue_params(entries, [0, 0, 1])
    path = tmp_path / "params.json"
    path.write_text(json.dumps(params))
    target = tmp_path / "target.pt"
    assert attach_evalue_calibration(str(source), str(clusters), str(path),
                                     str(target))["n_clusters"] == 2
    searcher = Searcher(str(target))
    hits = searcher.search_id("chain0", rerank="sw", top_k=2)
    assert hits
    for hit in hits:
        assert "bits" not in hit
        assert hit["normalized_sw_score"] == pytest.approx(
            hit["sw_score"] / min(40, hit["length"]))
        assert hit["evalue"] == pytest.approx(EV.apply_power_calibration(
            hit["raw_evalue"], params["power_calibration"]))


def test_cli_header_shows_normalized_score_and_cluster_count(tmp_path, capsys):
    from rfoldsearch import cli

    entries = add_coords([
        {"id": "chain0", "tokens": [1, 2, 3, 4] * 10, "cluster": "c0"},
        {"id": "chain1", "tokens": [1, 2, 3, 4] * 10, "cluster": "c1"},
    ])
    db = tmp_path / "db.pt"
    torch.save({"n_codes": 80, "entries": entries,
                "evalue_params": synthetic_evalue_params(entries)}, db)
    cli.main(["search", "--query", "chain0", "--db", str(db),
              "--rerank", "sw", "--top_k", "1"])
    header = capsys.readouterr().out.splitlines()[0]
    assert "2 chains" in header and "2 clusters" in header
