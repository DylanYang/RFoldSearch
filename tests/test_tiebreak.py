"""Deterministic, order-independent tie-breaking."""
import numpy as np
import pytest
import torch

from rfoldseek import tiebreak
from rfoldseek.search import Searcher

pytestmark = pytest.mark.filterwarnings(
    "ignore:E-value parameters were calibrated on .* clusters:UserWarning")


def test_keys_depend_on_id_not_position():
    a = tiebreak.tie_keys(["chainA", "chainB", "chainC"], seed=0)
    b = tiebreak.tie_keys(["chainC", "chainA", "chainB"], seed=0)

    assert a[0] == b[1] and a[1] == b[2] and a[2] == b[0]


def test_keys_change_with_seed():
    a = tiebreak.tie_keys(["chainA", "chainB"], seed=0)
    b = tiebreak.tie_keys(["chainA", "chainB"], seed=1)

    assert not np.array_equal(a, b)


def test_keys_are_reproducible_across_calls():
    assert np.array_equal(tiebreak.tie_keys(["x", "y"], 7),
                          tiebreak.tie_keys(["x", "y"], 7))


def test_keys_lie_in_the_unit_interval():
    k = tiebreak.tie_keys([f"c{i}" for i in range(500)], seed=0)

    assert k.min() >= 0.0 and k.max() < 1.0
    assert len(np.unique(k)) == 500


@pytest.fixture
def db_path(tmp_path):
    """Chains with identical token strings and coordinates, so every score
    (Smith-Waterman and alignment TM-score alike) ties exactly.

    With nothing but ties, which targets survive the top_n cut is decided
    entirely by the tie-break, which is what these tests are about.
    """
    rng = np.random.default_rng(0)
    tok = rng.integers(0, 80, size=60).astype(np.int16)
    xyz = rng.normal(size=(len(tok), 5, 3)).astype(np.float32)
    entries = [{"id": f"chain_{i:03d}.pdb", "tokens": tok.copy(),
                "coords": xyz.copy(),
                "mask": np.ones(len(tok), bool), "n_valid": len(tok)}
               for i in range(40)]
    p = tmp_path / "db.pt"
    torch.save({"n_codes": 80, "entries": entries, "checkpoint": "synthetic"}, p)
    return str(p)


def test_tie_seed_changes_which_candidates_survive_top_n(db_path):
    s = Searcher(db_path)

    pools = {tuple(sorted(h["target"] for h in
                          s.search_id("chain_000", top_k=3, top_n=3, tie_seed=sd)))
             for sd in range(5)}

    assert len(pools) > 1, "tie_seed did not change the admitted pool"


def test_index_order_admits_the_lowest_indices_when_everything_ties(db_path):
    s = Searcher(db_path)

    hits = s.search_id("chain_000", top_k=3, top_n=3)

    assert [h["target"] for h in hits] == ["chain_001", "chain_002", "chain_003"]


def test_same_tie_seed_gives_the_same_ranking(db_path):
    s = Searcher(db_path)

    a = s.search_id("chain_000", top_k=5, top_n=5, tie_seed=3)
    b = s.search_id("chain_000", top_k=5, top_n=5, tie_seed=3)

    assert [h["target"] for h in a] == [h["target"] for h in b]


def test_default_keeps_the_existing_index_order_behaviour(db_path):
    s = Searcher(db_path)

    assert s.search_id("chain_000", top_k=5) == s.search_id("chain_000", top_k=5,
                                                            tie_seed=None)


def test_tie_seed_orders_equal_tm_scores(db_path, monkeypatch):
    """Equal TM-scores are ordered by the hash key, not by candidate order.

    The reranker is driven directly with one fixed candidate array, so the
    prefilter's own tie-break cannot leak into the comparison: every seed
    reranks the identical pool in the identical order.
    """
    import numpy as np
    from rfoldseek import usalign

    s = Searcher(db_path)
    monkeypatch.setattr(usalign, "find_usalign", lambda explicit=None: "/fake/USalign")
    monkeypatch.setattr(usalign, "score_pool",
                        lambda binary, qc, qm, cands, **kw:
                        ({cid: (0.5, 0.5) for cid, _, _ in cands}, []))
    cand = np.arange(1, 12, dtype=np.int64)
    entry = s.ents[0]

    runs = [[h["target"] for h in s._rerank_tm(entry, cand, 11, None, sd)]
            for sd in range(5)]
    assert len({frozenset(r) for r in runs}) == 1, "pool differed; test is confounded"

    assert len({tuple(r) for r in runs}) > 1, "tie_seed did not reorder equal TM-scores"
