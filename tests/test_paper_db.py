"""Integration against the paper's bundled 885-chain database, if available.

Set RFOLDSEEK_PAPER_DB to the analysis repo's files/models/db.pt to enable;
the expected scores are the README example of RFoldseek-paper-code.
"""
import os

import pytest

from rfoldseek.search import Searcher

DB = os.environ.get("RFOLDSEEK_PAPER_DB")


@pytest.mark.skipif(not DB, reason="RFOLDSEEK_PAPER_DB not set")
def test_readme_example():
    s = Searcher(DB)
    hits = s.search_id("PDB_00001FFZ_1_A", top_k=3, rerank="sw")
    got = [(h["target"], round(h["sw_score"], 2)) for h in hits]
    assert got == [("PDB_00001FFK_1_0", 203.54),
                   ("PDB_00005AN9_1_N", 180.02),
                   ("PDB_00004V9F_1_0", 168.13)]
