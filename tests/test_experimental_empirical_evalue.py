"""Experimental raw-SW target-aware E-value deployment checks."""
import json
import numpy as np
import pytest
import torch

from rfoldsearch import experimental_evalue as EV
from rfoldsearch.cli import main
from rfoldsearch.database import (calibration_context_fingerprint,
                                  database_fingerprint)
from rfoldsearch.search import GAP_EXTEND, GAP_OPEN, Searcher, load_matrix
from rfoldsearch.experimental_calibrate import (_leader_tables, fit_from_matrices,
                                                sha256)
from rfoldsearch.smith_waterman import smith_waterman_numba


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


def test_multilabel_family_and_clan_overlaps_are_excluded():
    scores = np.array([[50., 40., 30., 20.]])
    clusters = np.array(["a", "b", "c", "q"])
    family = np.array(["RF1;RF2", "RF3", "RF4", "RF2"])
    clan = np.array(["", "CL2|CL3", "", "CL3"])
    leaders, annotated, before = _leader_tables(
        scores, np.array([3]), clusters, [[0], [1], [2], [3]],
        family, clan, np.full((1, 4), .1))
    assert leaders.tolist() == [[-1, -1, 2, -1]]
    assert annotated.tolist() == [[-1, -1, 2, -1]]
    assert before.tolist() == [1]


def test_empirical_map_is_monotone_and_uses_declared_tails():
    fitted = EV.fit_empirical([np.array([.1, .3, 1., 3., 10.]),
                               np.array([.2, .4, 2., 4., 20.])],
                              np.array(["RF1", "RF2"]), np.ones(2), 100)
    values = EV.apply_empirical(np.array([0., .01, .1, 1., 20., 21., 10000.]), fitted)
    assert np.all(np.diff(values) >= 0)
    assert values[0] == 0
    assert values[-1] == pytest.approx(min(100, fitted["observed_counts"][-1]
        * (10000 / fitted["raw_evalues"][-1]) ** fitted["low_tail_power_b"]))
    assert EV.apply_empirical(1e100, fitted) == 100
    assert fitted["upper_tail_rule"] == "anchored-power-cap"
    assert values[-2] == pytest.approx(min(100, fitted["observed_counts"][-1]
        * (21 / fitted["raw_evalues"][-1]) ** fitted["low_tail_power_b"]))
    assert EV.apply_empirical(np.nextafter(20., np.inf), fitted) == pytest.approx(
        fitted["observed_counts"][-1])
    assert values[1] == pytest.approx(
        fitted["observed_counts"][0] * (.01 / .1) ** fitted["low_tail_power_b"])
    assert EV.apply_empirical(1e-300, fitted) > 0
    legacy = dict(fitted)
    legacy.pop("upper_tail_rule")
    assert EV.apply_empirical(21., legacy) == 100


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
    uncensored, _, _ = fit_from_matrices(
        entries, tokens, ids, clusters, family, clan, np.arange(24),
        scores, tm, censored=False, log=lambda _: None)
    assert uncensored["censoring"] == "none"


def test_experimental_cli_fit_attach_and_search(tmp_path):
    rng = np.random.default_rng(41)
    entries = [{"id": f"t{i}", "tokens": rng.integers(0, 20, 28 + i % 7).tolist(),
                "cluster": f"c{i}"} for i in range(32)]
    db = tmp_path / "db.pt"
    torch.save({"n_codes": 80, "entries": entries, "n_clusters": 32}, db)
    labels = tmp_path / "labels.tsv"
    labels.write_text("chain_id\tfamily\tclan\n" + "".join(
        f"t{i}\tRF{i % 6}\t\n" for i in range(32)))
    matrix = load_matrix()
    scores = np.array([[smith_waterman_numba(
        np.asarray(q["tokens"]), np.asarray(t["tokens"]), matrix,
        GAP_OPEN, GAP_EXTEND) for t in entries] for q in entries], dtype=np.float32)
    score_path = tmp_path / "scores.npy"
    np.save(score_path, scores)
    from rfoldsearch.search import asset
    matrix_path = asset("sub_matrix.pt")
    score_manifest = tmp_path / "scores.json"
    score_manifest.write_text(json.dumps({
        "database_sha256": sha256(db), "matrix_sha256": sha256(matrix_path),
        "n_targets": 32, "query_ids": [f"t{i}" for i in range(32)],
        "score_definition": "raw_sw", "scores_sha256": sha256(score_path)}))
    tm_path = tmp_path / "tm.npy"
    np.save(tm_path, np.full((32, 32), .1))
    tm_manifest = tmp_path / "tm.json"
    tm_manifest.write_text(json.dumps({
        "database_sha256": sha256(db), "tm_sha256": sha256(tm_path),
        "n_targets": 32, "query_ids": [f"t{i}" for i in range(32)]}))
    params_path = tmp_path / "params.json"
    main(["fit-experimental-evalues", "--db", str(db), "--labels", str(labels),
          "--workdir", str(tmp_path / "fit"), "--output", str(params_path),
          "--scores", str(score_path), "--score-manifest", str(score_manifest),
          "--tm-matrix", str(tm_path), "--tm-manifest", str(tm_manifest)])
    params = json.loads(params_path.read_text())
    assert params["target_aware_gumbel"]["censoring"] == "left-censored-95th-percentile"
    assert params["empirical_calibration"]["upper_tail_rule"] == "anchored-power-cap"
    cluster_file = tmp_path / "clusters.tsv"
    cluster_file.write_text("".join(f"t{i}\tc{i}\n" for i in range(32)))
    attached = tmp_path / "attached.pt"
    main(["attach-calibration", "--db", str(db), "--clusters", str(cluster_file),
          "--evalue_params", str(params_path), "--output", str(attached)])
    searcher = Searcher(str(attached))
    assert searcher.experimental_ev is not None
    hit = searcher.search(np.asarray(entries[0]["tokens"]),
                          rerank="sw", top_k=1)[0]
    assert np.isfinite(hit["evalue"])
    assert hit["evalue"] >= 0
    assert hit["evalue"] == pytest.approx(EV.apply_empirical(
        hit["raw_evalue"], params["empirical_calibration"]))


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
