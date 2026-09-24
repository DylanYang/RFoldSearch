"""End-to-end search over a synthetic token database."""
import numpy as np
import pytest
import torch

from rfoldseek.search import Searcher

pytestmark = pytest.mark.filterwarnings(
    "ignore:E-value parameters were calibrated on .* clusters:UserWarning")


def test_search_pipeline(tmp_path):
    rng = np.random.default_rng(3)
    entries = []
    base = rng.integers(0, 80, size=120).astype(np.int16)
    for i in range(60):
        t = base.copy()
        flips = rng.integers(0, len(t), size=rng.integers(0, 40))
        t[flips] = rng.integers(0, 80, size=len(flips))
        entries.append({"id": f"chain_{i:03d}.pdb", "tokens": t,
                        "coords": np.zeros((1, 3, 3)), "mask": np.ones(1),
                        "n_valid": len(t)})
    db_path = tmp_path / "tiny_db.pt"
    torch.save({"n_codes": 80, "entries": entries, "checkpoint": "synthetic"},
               db_path)

    s = Searcher(str(db_path))
    hits = s.search_id("chain_000", top_k=5)
    assert 1 <= len(hits) <= 5
    assert hits[0]["target"] != "chain_000"          # self-hit removed
    scores = [h["sw_score"] for h in hits]
    assert scores == sorted(scores, reverse=True)    # ranked
    assert all("evalue" in h and h["evalue"] >= 0 for h in hits)
