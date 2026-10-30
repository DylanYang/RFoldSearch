"""Integration against the paper's bundled 885-chain database, if available.

Set RFOLDSEARCH_PAPER_DB to the analysis repo's files/models/db.pt to enable;
the expected scores are the README example of RFoldSearch-paper-code.
"""
import os

import pytest

from rfoldsearch.search import Searcher

DB = os.environ.get("RFOLDSEARCH_PAPER_DB")


@pytest.mark.skipif(not DB, reason="RFOLDSEARCH_PAPER_DB not set")
def test_readme_example():
    s = Searcher(DB)
    hits = s.search_id("PDB_00001FFZ_1_A", top_k=200)
    sw = {h["target"]: round(h["sw_score"], 2) for h in hits}
    assert sw["PDB_00001FFK_1_0"] == 203.54
    assert sw["PDB_00005AN9_1_N"] == 180.02
    assert sw["PDB_00004V9F_1_0"] == 168.13
