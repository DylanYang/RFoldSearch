"""The search path counts the database in clusters, not chains."""
import json
import numpy as np
import pytest
import torch

from rfoldseek import evalue as EV
from rfoldseek.database import assign_clusters, read_clusters


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


def test_shipped_params_are_the_cluster_max_scheme():
    from rfoldseek.search import asset
    par = json.load(open(asset("evalue_params.json")))
    assert par["scheme"] == "cluster-max"
    assert par["length_table"], "no length grid in the packaged parameters"
    for row in par["length_table"]:
        assert {"length", "mu", "beta"} <= set(row)
    mu, beta = EV.clustermax_params_for_length(par, 90)
    assert beta > 0


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
    h_many = s_many.search(q, top_k=1)
    h_few = s_few.search(q, top_k=1)
    assert h_many and h_few, "the synthetic query must retrieve something"
    assert "evalue" in h_many[0], "the packaged parameters were not applied"
    assert h_few[0]["evalue"] < h_many[0]["evalue"]
