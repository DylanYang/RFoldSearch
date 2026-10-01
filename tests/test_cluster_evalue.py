"""The search path counts the database in clusters, not chains."""
import json
import numpy as np
import pytest
import torch

from rfoldseek import evalue as EV
from rfoldseek.database import (assign_clusters, attach_evalue_calibration,
                                read_clusters)
from helpers import add_coords


def test_read_clusters_parses_a_cdhit_clstr(tmp_path):
    p = tmp_path / "nr80.clstr"
    p.write_text(">Cluster 0\n"
                 "0\t120nt, >chainA... *\n"
                 "1\t118nt, >chainB... at +/98.00%\n"
                 ">Cluster 1\n"
                 "0\t90nt, >chainC... *\n")
    assert read_clusters(str(p)) == {"chainA": 0, "chainB": 0, "chainC": 1}


def test_read_clusters_parses_a_tsv(tmp_path):
    p = tmp_path / "map.tsv"
    p.write_text("chain_id\tcluster\nchainA\tc1\nchainB\tc1\nchainC\tc2\n")
    assert read_clusters(str(p)) == {"chainA": "c1", "chainB": "c1", "chainC": "c2"}


def test_assign_clusters_gives_unmapped_chains_their_own_cluster():
    ents = [{"id": "/data/chainA.cif"}, {"id": "/data/chainB.cif"},
            {"id": "/data/orphan1.cif"}, {"id": "/data/orphan2.cif"}]
    labels = assign_clusters(ents, {"chainA": 0, "chainB": 0})
    assert labels[0] == labels[1], "mapped chains must keep their cluster"
    # two unmapped chains are two clusters, not one pooled bucket: pooling
    # them would make an unclustered database look less redundant than it is
    assert labels[2] != labels[3]
    assert labels[2] not in labels[:2] and labels[3] not in labels[:2]
    assert len(set(labels)) == 3


def test_evalue_counts_clusters_so_redundancy_does_not_inflate_it():
    """Duplicating every chain must not change the E-value of a hit.

    This is the property the chain-level scheme lacked: there, the same
    database with each family stored twice reported E-values twice as large.
    """
    mu, beta = 8.0, 3.0
    e_small = EV.evalue_clustermax(30.0, mu, beta, 500)
    e_same_clusters = EV.evalue_clustermax(30.0, mu, beta, 500)
    e_more_clusters = EV.evalue_clustermax(30.0, mu, beta, 1000)
    assert e_small == e_same_clusters
    assert e_more_clusters == pytest.approx(2 * e_small, rel=1e-6)


def test_bit_score_and_evalue_agree_in_the_tail():
    """E = N_clusters * 2**(-S') wherever the survival function is exponential."""
    mu, beta, n = 8.0, 3.0, 400
    s = 60.0                                  # far into the tail
    bits = EV.bit_score_clustermax(s, mu, beta)
    assert EV.evalue_clustermax(s, mu, beta, n) == pytest.approx(n * 2.0 ** -bits,
                                                                 rel=1e-6)


def test_evalue_extreme_tail_does_not_collapse_to_zero():
    """The Gumbel survival function must retain precision for tiny E-values."""
    value = EV.evalue_clustermax(120.0, mu=8.0, beta=3.0, n_clusters=882)
    expected = 882.0 * np.exp(-(120.0 - 8.0) / 3.0)
    assert value > 0.0
    assert value == pytest.approx(expected, rel=1e-12)


def test_empirical_calibration_is_a_monotone_weighted_background_count():
    calibration = EV.fit_empirical_calibration(
        [0.1, 0.1, 1.0, 10.0], weights=[0.2, 0.3, 0.5, 1.0], n_clusters=12)

    assert calibration["raw_evalue_knots"] == [0.1, 1.0, 10.0]
    assert calibration["calibrated_evalues"] == pytest.approx([0.5, 1.0, 2.0])
    observed = EV.apply_empirical_calibration(
        np.array([0.01, 0.1, 0.5, 1.0, 100.0]), calibration)
    assert observed == pytest.approx([0.1, 0.5, 0.5, 1.0, 2.0])
    assert np.all(np.diff(observed) >= 0)


def test_invalid_empirical_calibration_is_rejected():
    with pytest.raises(ValueError, match="monotone"):
        EV.apply_empirical_calibration(
            1.0, {"raw_evalue_knots": [1.0, 0.1],
                  "calibrated_evalues": [0.1, 1.0]})


def test_bounded_empirical_calibration_does_not_saturate_weak_hits():
    calibration = {
        "raw_evalue_knots": [1.0, 10.0, 99.0],
        "calibrated_evalues": [0.1, 1.0, 18.0],
        "finite_sample_floor": 0.01,
        "candidate_raw_evalue_ceiling": 100.0,
    }
    assert EV.apply_empirical_calibration(
        [0.1, 99.5, 100.1, 1000.0], calibration) == pytest.approx(
            [0.01, 18.0, 100.1, 1000.0])


def test_shipped_params_are_the_cluster_max_scheme():
    from rfoldseek.search import asset
    par = json.load(open(asset("evalue_params.json")))
    assert par["scheme"] == "cluster-max"
    assert par["length_table"], "no length grid in the packaged parameters"
    for row in par["length_table"]:
        assert {"length", "mu", "beta"} <= set(row)
    mu, beta = EV.clustermax_params_for_length(par, 90)
    assert beta > 0


def test_query_conditional_gumbel_uses_token_entropy():
    params = {"query_parameter_model": {
        "distribution": "Gumbel",
        "training_domain": {"log_length_min": np.log(4),
                            "log_length_max": np.log(8),
                            "entropy_min": 0.0, "entropy_max": 1.0},
        "centre": {"log_length_mean": 0.0, "log_length_scale": 1.0,
                   "entropy_mean": 0.0, "entropy_scale": 1.0},
        "coefficients_mu_logbeta": [
            [10.0, 0.0], [1.0, 0.0], [0.0, 0.0],
            [2.0, 0.0], [0.0, 0.0], [0.0, 0.0],
        ],
    }}
    one_token = EV.clustermax_params_for_query(params, [0, 0, 0, 0])
    two_tokens = EV.clustermax_params_for_query(params, [0, 0, 1, 1])
    assert one_token == pytest.approx((10.0 + np.log(4), 1.0))
    assert two_tokens == pytest.approx((12.0 + np.log(4), 1.0))
    with pytest.warns(RuntimeWarning, match="outside"):
        long_query = EV.clustermax_params_for_query(params, [0, 1] * 8)
    with pytest.warns(RuntimeWarning, match="outside"):
        high_entropy = EV.clustermax_params_for_query(params, list(range(8)))
    assert long_query == pytest.approx((12.0 + np.log(8), 1.0))
    assert high_entropy == pytest.approx((12.0 + np.log(8), 1.0))
    # A polynomial log-beta term must also use the bounded predictors.
    params["query_parameter_model"]["coefficients_mu_logbeta"][2][1] = -1.0
    params["query_parameter_model"]["coefficients_mu_logbeta"][4][1] = -1.0
    at_boundary = EV.clustermax_params_for_query(params, [0, 1] * 4)
    with pytest.warns(RuntimeWarning, match="outside"):
        long_and_diverse = EV.clustermax_params_for_query(params, list(range(16)))
    assert long_and_diverse == pytest.approx(at_boundary)
    with pytest.warns(RuntimeWarning, match="outside"):
        short_query = EV.clustermax_params_for_query(params, [0, 0])
    assert short_query == pytest.approx(
        EV.clustermax_params_for_query(params, [0, 0, 0, 0]))
    del params["query_parameter_model"]["training_domain"]
    with pytest.raises(ValueError, match="training_domain"):
        EV.clustermax_params_for_query(params, [0, 1, 0, 1])


def test_predicted_calibration_requires_matching_clustered_database(tmp_path):
    from rfoldseek.search import Searcher

    rng = np.random.default_rng(91)
    entries = add_coords([{"id": f"chain{i}",
                           "tokens": rng.integers(0, 80, size=40).tolist()}
                          for i in range(12)])
    source = tmp_path / "unclustered.pt"
    torch.save({"n_codes": 80, "entries": entries}, source)
    clusters = tmp_path / "nr80.clstr"
    clusters.write_text("".join(
        f">Cluster {i // 2}\n0\t40nt, >t{i}... *\n"
        if i % 2 == 0 else f"1\t40nt, >t{i}... at +/90.00%\n"
        for i in range(12)))
    params = {
        "scheme": "predicted-query-conditional-clustermax-empirical",
        "n_clusters": 6,
        "query_parameter_model": {
            "distribution": "Gumbel",
            "training_domain": {"log_length_min": np.log(20),
                                "log_length_max": np.log(80),
                                "entropy_min": 0.0, "entropy_max": 6.0},
            "centre": {"log_length_mean": 0.0, "log_length_scale": 1.0,
                       "entropy_mean": 0.0, "entropy_scale": 1.0},
            "coefficients_mu_logbeta": [[8.0, np.log(3.0)]]
                                      + [[0.0, 0.0]] * 5,
        },
        "empirical_calibration": {
            "raw_evalue_knots": [0.1, 1.0, 10.0],
            "calibrated_evalues": [0.01, 0.1, 1.0],
            "finite_sample_floor": 0.005,
        },
    }
    path = tmp_path / "predicted.json"
    path.write_text(json.dumps(params))
    with pytest.raises(ValueError, match="recalibrate for this database"):
        Searcher(str(source), evalue_params_path=str(path))

    target = tmp_path / "calibrated.pt"
    attached = attach_evalue_calibration(
        str(source), str(clusters), str(path), str(target), index_prefix="t")
    assert attached["n_clusters"] == 6
    searcher = Searcher(str(target))
    hits = searcher.search(np.asarray(entries[0]["tokens"]), query_entry=entries[0],
                           top_k=2)
    assert hits
    for hit in hits:
        expected = EV.apply_empirical_calibration(hit["raw_evalue"],
                                                   params["empirical_calibration"])
        assert hit["evalue"] == expected
    assert [hit["structure_score"] for hit in hits] == sorted(
        [hit["structure_score"] for hit in hits], reverse=True)


@pytest.mark.filterwarnings(
    "ignore:E-value parameters were calibrated on .* clusters:UserWarning")
def test_searcher_uses_the_cluster_count_of_the_loaded_database(tmp_path):
    """Two databases with the same chains but different clustering.

    The one whose chains collapse into fewer clusters must report smaller
    E-values for the same alignment, because fewer independent questions were
    asked of it.
    """
    from rfoldseek.search import Searcher

    rng = np.random.default_rng(0)
    ents = []
    for i in range(12):
        t = [int(rng.integers(80)) for _ in range(40)]
        ents.append({"id": f"chain{i}", "tokens": t})
    add_coords(ents)

    def write(path, labels):
        for e, lab in zip(ents, labels):
            e = dict(e)
        db = {"n_codes": 80, "checkpoint": "none",
              "entries": [dict(e, cluster=lab) for e, lab in zip(ents, labels)]}
        torch.save(db, path)
        return path

    many = write(str(tmp_path / "many.pt"), [f"c{i}" for i in range(12)])
    few = write(str(tmp_path / "few.pt"), [f"c{i // 6}" for i in range(12)])

    s_many = Searcher(many)
    s_few = Searcher(few)
    assert s_many.n_clusters == 12
    assert s_few.n_clusters == 2

    q = np.asarray(ents[0]["tokens"], dtype=np.int64)
    h_many = s_many.search(q, query_entry=ents[0], top_k=1)
    h_few = s_few.search(q, query_entry=ents[0], top_k=1)
    assert h_many and h_few, "the synthetic query must retrieve something"
    assert "evalue" in h_many[0], "the packaged parameters were not applied"
    assert h_few[0]["evalue"] < h_many[0]["evalue"]


def test_searcher_applies_matching_embedded_empirical_calibration(tmp_path):
    from rfoldseek.search import Searcher

    rng = np.random.default_rng(3)
    entries = add_coords([{"id": f"chain{i}",
                           "tokens": rng.integers(0, 80, size=40).tolist(),
                           "cluster": f"c{i}"} for i in range(12)])
    from rfoldseek.search import asset
    params = json.load(open(asset("evalue_params.json")))
    params["empirical_calibration"] = EV.fit_empirical_calibration(
        [0.01, 0.1, 1.0, 10.0], [0.1, 0.2, 0.3, 0.4], n_clusters=12)
    db = tmp_path / "calibrated.pt"
    torch.save({"n_codes": 80, "entries": entries,
                "evalue_params": params}, db)

    searcher = Searcher(str(db))
    hit = searcher.search(np.asarray(entries[0]["tokens"]), query_entry=entries[0],
                          top_k=1)[0]
    assert hit["evalue"] == EV.apply_empirical_calibration(
        hit["raw_evalue"], params["empirical_calibration"])


def test_searcher_rejects_empirical_mapping_for_another_database(tmp_path):
    from rfoldseek.search import Searcher, asset

    rng = np.random.default_rng(4)
    entries = [{"id": f"chain{i}",
                "tokens": rng.integers(0, 80, size=40).tolist(),
                "cluster": f"c{i}"} for i in range(12)]
    params = json.load(open(asset("evalue_params.json")))
    params["empirical_calibration"] = EV.fit_empirical_calibration(
        [0.1, 1.0], n_clusters=99)
    db = tmp_path / "mismatch.pt"
    torch.save({"n_codes": 80, "entries": entries,
                "evalue_params": params}, db)

    with pytest.raises(ValueError, match="recalibrate for this database"):
        Searcher(str(db))


@pytest.mark.filterwarnings(
    "ignore:E-value parameters were calibrated on .* clusters:UserWarning")
def test_cli_header_reports_the_cluster_count_when_it_reports_an_evalue(tmp_path,
                                                                        capsys):
    """The E-value's search space is clusters, so the header must name them.

    A reader given only "885 chains" cannot tell what N the E-value used, which
    matters most on the redundant databases where the two numbers differ.
    """
    import numpy as np, torch
    from rfoldseek import cli

    rng = np.random.default_rng(0)
    ents = add_coords([{"id": f"chain{i}", "tokens": [int(rng.integers(80)) for _ in range(40)],
                        "cluster": f"c{i // 4}"} for i in range(12)])
    db = tmp_path / "db.pt"
    torch.save({"n_codes": 80, "checkpoint": "none", "entries": ents}, str(db))

    cli.main(["search", "--query", "chain0", "--db", str(db), "--top_k", "1"])
    head = capsys.readouterr().out.splitlines()[0]
    assert "12 chains" in head
    assert "3 clusters" in head
