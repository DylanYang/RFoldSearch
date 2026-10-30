"""Predicted-database composition-adjusted E-value fitting and deployment."""
import json

import numpy as np
import pytest
import torch

from helpers import add_coords
from rfoldsearch import compbias, experimental_evalue as EV
from rfoldsearch.cli import main
from rfoldsearch.database import (calibration_context_fingerprint,
                                  database_fingerprint)
from rfoldsearch.predicted_calibrate import fit_from_matrices
from rfoldsearch.experimental_calibrate import sha256
from rfoldsearch.search import GAP_EXTEND, GAP_OPEN, Searcher, load_matrix
from rfoldsearch.smith_waterman import smith_waterman_numba


def test_composition_profile_matches_positionwise_formula():
    matrix = np.array([[2., -1.], [-1., 2.]], dtype=np.float32)
    query = np.array([0, 0, 1])
    frequencies = np.array([.6, .4])
    positions, profile = compbias.query_profile(query, matrix, frequencies)
    expected = frequencies @ matrix
    assert positions.tolist() == [0, 1, 2]
    assert profile[0] == pytest.approx(matrix[0] - (matrix[0] + matrix[1]) / 2 + expected)
    assert profile[2] == pytest.approx(matrix[1] - matrix[0] + expected)


def test_clean_predicted_fit_uses_full_background_without_expansion():
    rng = np.random.default_rng(12)
    tokens = [rng.integers(0, 10, 30 + i % 6).tolist() for i in range(40)]
    entries = [{"id": f"t{i}", "tokens": token} for i, token in enumerate(tokens)]
    ids = [entry["id"] for entry in entries]
    clusters = np.array([f"c{i}" for i in range(40)])
    family = np.array([f"RF{i % 5}" for i in range(40)])
    clan = np.array([""] * 40)
    scores = np.maximum(rng.gumbel(12, 2, (32, 40)), 0)
    tm = np.full(scores.shape, .1)
    model, empirical, audit = fit_from_matrices(
        entries, tokens, ids, clusters, family, clan,
        np.arange(24), np.arange(24, 32), scores, tm, log=lambda _: None)
    assert audit["n_core_training_queries"] == 24
    assert audit["n_auxiliary_training_queries"] == 0
    assert audit["n_calibration_queries"] == 8
    assert empirical["background_scaling"] == "complete-eligible-clusters"
    assert empirical["n_queries"] == 8
    assert model["censoring"] == "left-censored-95th-percentile"


def test_compressed_gumbel_likelihood_matches_uncompressed_fit():
    rng = np.random.default_rng(73)
    scores = rng.gumbel(10, 2, (24, 40))
    leaders = np.broadcast_to(np.arange(40), scores.shape).copy()
    lengths = np.linspace(25, 120, 24)
    cluster_lengths = np.repeat(np.linspace(30, 150, 8), 5)
    entropy = rng.uniform(1, 3, 24)
    families = np.array([f"RF{i % 6}" for i in range(24)])
    plain = EV.fit_gumbel(scores, leaders, lengths, entropy,
                          cluster_lengths, families, compress=False)
    packed = EV.fit_gumbel(scores, leaders, lengths, entropy,
                           cluster_lengths, families, compress=True)
    assert packed["n_compressed_observations"] <= packed["n_observations"]
    assert np.asarray(packed["coefficients_mu_logbeta"]) == pytest.approx(
        np.asarray(plain["coefficients_mu_logbeta"]), abs=1e-4)


def test_search_uses_compbias_only_for_evalue_and_rejects_wrong_context(tmp_path):
    entries = add_coords([{"id": f"t{i}", "tokens": [i % 5, (i + 1) % 5] * 15,
                           "cluster": f"c{i}"} for i in range(5)])
    tokens = [entry["tokens"] for entry in entries]
    matrix = load_matrix()
    background = compbias.background_frequencies(tokens, matrix.shape[0])
    target = database_fingerprint(entries, [e["cluster"] for e in entries], 80)
    context = calibration_context_fingerprint(target, matrix, GAP_OPEN, GAP_EXTEND)
    params = {
        "scheme": "target-aware-compbias20-raw-sw-empirical-v1",
        "score_definition": "compbias20-raw-sw", "composition_window": 20,
        "composition_background": background.tolist(), "n_clusters": 5,
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
            "background_scaling": "complete-eligible-clusters",
        },
    }
    db = tmp_path / "db.pt"
    torch.save({"n_codes": 80, "entries": entries, "n_clusters": 5,
                "evalue_params": params}, db)
    searcher = Searcher(str(db))
    query = np.asarray(tokens[0])
    hit = searcher.search(query, query_entry=entries[0], top_k=1)[0]
    positions, profile = compbias.query_profile(query, matrix, background)
    target_index = searcher.id2i[hit["target"]]
    adjusted = smith_waterman_numba(positions, np.asarray(tokens[target_index]),
                                    profile, GAP_OPEN, GAP_EXTEND)
    assert hit["evalue_score"] == pytest.approx(adjusted)
    assert hit["sw_score"] == pytest.approx(smith_waterman_numba(
        query, np.asarray(tokens[target_index]), matrix, GAP_OPEN, GAP_EXTEND))
    assert hit["raw_evalue"] == pytest.approx(EV.initial_evalue(
        adjusted, len(query), EV.token_entropy(query),
        params["target_aware_gumbel"]))
    assert hit["evalue"] == pytest.approx(EV.apply_empirical(
        hit["raw_evalue"], params["empirical_calibration"]))
    params["composition_background"][0] += .01
    bad = tmp_path / "wrong.json"
    bad.write_text(json.dumps(params))
    with pytest.raises(ValueError, match="composition background"):
        Searcher(str(db), evalue_params_path=str(bad))


def test_fitting_cli_writes_attachable_predicted_parameters(tmp_path):
    rng = np.random.default_rng(30)
    entries = add_coords([{"id": f"t{i}", "tokens": rng.integers(0, 15, 25 + i % 8).tolist(),
                           "cluster": f"c{i}"} for i in range(40)])
    tokens = [entry["tokens"] for entry in entries]
    db = tmp_path / "db.pt"
    torch.save({"n_codes": 80, "entries": entries, "n_clusters": 40}, db)
    labels = tmp_path / "labels.tsv"
    labels.write_text("chain_id\tfamily\tclan\n" + "".join(
        f"t{i}\tRF{i % 6}\t\n" for i in range(40)))
    training = tmp_path / "train.npy"
    auxiliary = tmp_path / "aux.npy"
    calibration = tmp_path / "cal.npy"
    np.save(training, np.arange(20))
    np.save(auxiliary, np.arange(20, 24))
    np.save(calibration, np.arange(24, 32))
    matrix_path = tmp_path / "matrix.pt"
    from rfoldsearch.search import asset
    matrix_path.write_bytes(open(asset("sub_matrix.pt"), "rb").read())
    matrix = load_matrix(str(matrix_path))
    frequencies = compbias.background_frequencies(tokens, matrix.shape[0])
    scores = np.empty((32, 40), dtype=np.float32)
    for row in range(32):
        positions, profile = compbias.query_profile(tokens[row], matrix, frequencies)
        for target in range(40):
            scores[row, target] = smith_waterman_numba(
                positions, np.asarray(tokens[target]), profile, GAP_OPEN, GAP_EXTEND)
    scores_path = tmp_path / "scores.npy"
    np.save(scores_path, scores)
    manifest = tmp_path / "scores.json"
    manifest.write_text(json.dumps({
        "database_sha256": sha256(db), "matrix_sha256": sha256(matrix_path),
        "n_targets": 40, "query_ids": [f"t{i}" for i in range(32)],
        "score_definition": "compbias20_sw", "scores_sha256": sha256(scores_path)}))
    tm_path = tmp_path / "tm.npy"
    np.save(tm_path, np.full(scores.shape, .1))
    tm_manifest = tmp_path / "tm.json"
    tm_manifest.write_text(json.dumps({
        "database_sha256": sha256(db), "tm_sha256": sha256(tm_path),
        "n_targets": 40, "query_ids": [f"t{i}" for i in range(32)]}))
    output = tmp_path / "predicted.json"
    command = ["fit-predicted-evalues", "--db", str(db), "--labels", str(labels),
          "--training-indices", str(training), "--calibration-indices", str(calibration),
          "--auxiliary-indices", str(auxiliary),
          "--workdir", str(tmp_path / "fit"), "--output", str(output),
          "--audit-ceiling", "120",
          "--matrix", str(matrix_path), "--scores", str(scores_path),
          "--score-manifest", str(manifest), "--tm-matrix", str(tm_path),
          "--tm-manifest", str(tm_manifest)]
    main(command)
    params = json.loads(output.read_text())
    assert params["scheme"] == "target-aware-compbias20-raw-sw-empirical-v1"
    assert params["audit_raw_evalue_ceiling"] == 120
    assert params["empirical_calibration"]["upper_tail_rule"] == "anchored-power-cap"
    assert params["empirical_calibration"]["background_scaling"] == "complete-eligible-clusters"
    searcher = Searcher(str(db), matrix_path=str(matrix_path),
                        evalue_params_path=str(output))
    assert searcher.search(np.asarray(tokens[0]), query_entry=entries[0],
                           top_k=1)[0]["evalue"] >= 0
    clusters = tmp_path / "clusters.tsv"
    clusters.write_text("".join(f"t{i}\tc{i}\n" for i in range(40)))
    attached = tmp_path / "attached.pt"
    main(["attach-calibration", "--db", str(db), "--clusters", str(clusters),
          "--evalue_params", str(output), "--output", str(attached)])
    assert Searcher(str(attached), matrix_path=str(matrix_path)).predicted_ev is not None
    audit_file = tmp_path / "fit" / "new_usalign.tsv"
    audit_file.write_text("query_id\ttarget_id\ttm_query\ttm_target\n")
    (tmp_path / "fit" / "new_usalign.manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="US-align cache does not match"):
        main(command)
