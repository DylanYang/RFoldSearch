"""RFoldseek-TM: the --rerank tm path through Searcher.search."""
import numpy as np
import pytest
import torch

from rfoldseek import usalign
from rfoldseek.search import Searcher


@pytest.fixture
def db_path(tmp_path):
    """A 40-chain synthetic database with real per-residue coordinates."""
    rng = np.random.default_rng(3)
    base = rng.integers(0, 80, size=120).astype(np.int16)
    entries = []
    for i in range(40):
        t = base.copy()
        flips = rng.integers(0, len(t), size=rng.integers(0, 40))
        t[flips] = rng.integers(0, 80, size=len(flips))
        entries.append({
            "id": f"chain_{i:03d}.pdb",
            "tokens": t,
            "coords": rng.normal(size=(len(t), 5, 3)).astype(np.float32) * 10.0,
            "mask": np.ones(len(t), dtype=bool),
            "n_valid": len(t),
        })
    p = tmp_path / "tiny_db.pt"
    torch.save({"n_codes": 80, "entries": entries, "checkpoint": "synthetic"}, p)
    return str(p)


def _fake_scores(monkeypatch, table):
    """Make score_pool return `table` ({id: (tm_query, tm_target)})."""
    monkeypatch.setattr(usalign, "find_usalign", lambda explicit=None: "/fake/USalign")

    def fake_pool(binary, qc, qm, candidates, **kw):
        return {cid: table[cid] for cid, _, _ in candidates if cid in table}, []

    monkeypatch.setattr(usalign, "score_pool", fake_pool)


def test_tm_rerank_ranks_by_the_mean_of_both_normalisations(db_path, monkeypatch):
    s = Searcher(db_path)
    ids = [i for i in s.ids if i != "chain_000"]
    # chain B wins on the mean (0.60) despite a lower tm_query than chain A.
    table = {cid: (0.10, 0.10) for cid in ids}
    table[ids[0]] = (0.90, 0.20)   # mean 0.55
    table[ids[1]] = (0.55, 0.65)   # mean 0.60
    _fake_scores(monkeypatch, table)

    hits = s.search_id("chain_000", top_k=3, rerank="tm")

    assert hits[0]["target"] == ids[1]
    assert hits[0]["tm_mean"] == pytest.approx(0.60)
    assert hits[1]["target"] == ids[0]


def test_tm_hits_carry_both_components_and_no_evalue(db_path, monkeypatch):
    s = Searcher(db_path)
    _fake_scores(monkeypatch, {cid: (0.4, 0.6) for cid in s.ids})

    hit = s.search_id("chain_000", top_k=1, rerank="tm")[0]

    assert hit["tm_query"] == pytest.approx(0.4)
    assert hit["tm_target"] == pytest.approx(0.6)
    assert "evalue" not in hit and "bits" not in hit and "sw_score" not in hit


def test_sw_rerank_is_still_the_default(db_path):
    s = Searcher(db_path)

    hit = s.search_id("chain_000", top_k=1)[0]

    assert "sw_score" in hit and "tm_mean" not in hit


def test_unknown_rerank_backend_is_rejected(db_path):
    s = Searcher(db_path)

    with pytest.raises(ValueError, match="rerank"):
        s.search_id("chain_000", rerank="tmalign")


def test_cli_prints_the_tm_table_and_names_the_backend(db_path, monkeypatch, capsys):
    from rfoldseek import cli

    _fake_scores(monkeypatch, {f"chain_{i:03d}": (0.4, 0.6) for i in range(40)})
    cli.main(["search", "--query", "chain_000", "--db", db_path,
              "--top_k", "2", "--rerank", "tm"])

    out = capsys.readouterr().out
    assert "rerank: US-align TM" in out
    assert "TM_mean" in out and "TM_query" in out and "TM_target" in out
    assert "SW_score" not in out and "E-value" not in out
    assert "0.5000" in out


def test_cli_sw_output_is_unchanged(db_path, capsys):
    from rfoldseek import cli

    cli.main(["search", "--query", "chain_000", "--db", db_path, "--top_k", "2"])

    out = capsys.readouterr().out
    assert "SW_score" in out and "rerank:" not in out
