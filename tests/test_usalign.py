"""Unit tests for the US-align reranking backend.

None of these need a USalign binary on PATH; the one test that does is
skipped when it is absent.
"""
import os
import shutil

import numpy as np
import pytest

from rfoldsearch import usalign


def _coords(rng, n):
    """(n, 5, 3) backbone coords; channel 0 is C3'."""
    return rng.normal(size=(n, 5, 3)).astype(np.float32) * 10.0


def test_write_c3_pdb_writes_the_c3_channel_of_valid_residues(tmp_path):
    from biotite.structure.io.pdb import PDBFile

    rng = np.random.default_rng(0)
    coords = _coords(rng, 6)
    mask = np.array([1, 1, 0, 1, 0, 1], dtype=bool)
    path = tmp_path / "q.pdb"

    n = usalign.write_c3_pdb(coords, mask, str(path))

    assert n == 4
    atoms = PDBFile.read(str(path)).get_structure(model=1)
    assert list(atoms.atom_name) == ["C3'"] * 4
    np.testing.assert_allclose(atoms.coord, coords[mask, 0, :], atol=1e-3)


def test_write_c3_pdb_numbers_residues_consecutively_over_valid_ones(tmp_path):
    from biotite.structure.io.pdb import PDBFile

    coords = _coords(np.random.default_rng(1), 5)
    mask = np.array([0, 1, 1, 0, 1], dtype=bool)
    path = tmp_path / "q.pdb"

    usalign.write_c3_pdb(coords, mask, str(path))

    atoms = PDBFile.read(str(path)).get_structure(model=1)
    assert list(atoms.res_id) == [1, 2, 3]


OUTFMT2 = """\
#PDBchain1\tPDBchain2\tTM1\tTM2\tRMSD\tID1\tID2\tIDali\tL1\tL2\tLali
query.pdb:A\t/tmp/t/targets/0.pdb:A\t0.7550\t0.7292\t2.31\t0.21\t0.20\t0.55\t128\t129\t120
query.pdb:A\t/tmp/t/targets/2.pdb:A\t0.5044\t0.5332\t3.90\t0.18\t0.17\t0.41\t128\t121\t95
"""


def test_parse_outfmt2_keeps_both_normalisations_per_candidate():
    parsed = usalign.parse_outfmt2(OUTFMT2)

    assert parsed == {0: (0.7550, 0.7292), 2: (0.5044, 0.5332)}


def test_parse_outfmt2_ignores_the_header_line():
    assert usalign.parse_outfmt2("#PDBchain1\tPDBchain2\tTM1\tTM2\n") == {}


def test_candidates_absent_from_the_output_score_zero():
    scores = usalign.assemble_scores(
        {0: "chainA", 1: "chainB", 2: "chainC"},
        {0: (0.75, 0.72), 2: (0.50, 0.53)},
    )

    assert scores == {"chainA": (0.75, 0.72), "chainB": (0.0, 0.0),
                      "chainC": (0.50, 0.53)}


def test_find_usalign_error_names_every_resolution_mechanism(monkeypatch):
    monkeypatch.delenv("USALIGN", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: None)

    with pytest.raises(FileNotFoundError) as e:
        usalign.find_usalign()

    msg = str(e.value)
    assert "--usalign" in msg and "USALIGN" in msg and "PATH" in msg


def test_find_usalign_prefers_the_explicit_path(tmp_path, monkeypatch):
    binary = tmp_path / "USalign"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    monkeypatch.setenv("USALIGN", "/nonexistent/USalign")

    assert usalign.find_usalign(str(binary)) == str(binary)


@pytest.mark.skipif(shutil.which("USalign") is None and not os.environ.get("USALIGN"),
                    reason="USalign binary not installed")
def test_score_pool_against_the_real_binary():
    """A chain against itself scores ~1; an unrelated chain scores lower."""
    rng = np.random.default_rng(7)
    a = _coords(rng, 40)
    b = _coords(rng, 40)

    scores, skipped = usalign.score_pool(
        usalign.find_usalign(), a, np.ones(40, bool),
        [("self", a, np.ones(40, bool)), ("other", b, np.ones(40, bool))])

    assert skipped == []
    assert scores["self"][0] == pytest.approx(1.0, abs=1e-3)
    assert scores["self"][1] == pytest.approx(1.0, abs=1e-3)
    assert scores["other"][0] < scores["self"][0]
    assert all(0.0 <= v <= 1.0 for pair in scores.values() for v in pair)
