"""One-step E-value calibration: rfoldsearch calibrate."""
import json
import stat
import sys

import numpy as np
import pytest
import torch

from helpers import add_coords
from rfoldsearch.cli import main

N_QUERIES = 10

FAKE_USALIGN = """#!{python}
# Stand-in for USalign -outfmt 2: every target scores TM 0.2 both ways.
import os, sys
args = sys.argv[1:]
query, tdir, listfile = args[0], args[2], args[3]
print("#PDBchain1\\tPDBchain2\\tTM1\\tTM2")
for name in open(listfile).read().split():
    print(f"{{query}}:A\\t{{os.path.join(tdir, name)}}:A\\t0.2\\t0.2")
"""


def write_inputs(tmp_path):
    rng = np.random.default_rng(5)
    targets = add_coords([{"id": f"t{i}",
                           "tokens": rng.integers(0, 4 + i % 12,
                                                  size=24 + i).tolist()}
                          for i in range(26)], seed=1)
    queries = add_coords([{"id": f"q{i}",
                           "tokens": rng.integers(0, 4 + i % 12,
                                                  size=26 + 2 * i).tolist()}
                          for i in range(N_QUERIES)], seed=2)
    db, qdb = tmp_path / "db.pt", tmp_path / "queries.pt"
    torch.save({"n_codes": 80, "entries": targets}, db)
    torch.save({"n_codes": 80, "entries": queries}, qdb)
    clusters = tmp_path / "nr80.tsv"
    clusters.write_text("".join(f"t{i}\tc{i // 2 if i < 4 else i}\n"
                                for i in range(26)))
    labels = tmp_path / "labels.tsv"
    labels.write_text("chain_id\tfamily\tclan\n"
                      + "".join(f"t{i}\tRF{i:05d}\t\n" for i in range(26))
                      + "".join(f"q{i}\tRF9{i:04d}\t\n" for i in range(N_QUERIES)))
    return db, qdb, clusters, labels


def fake_usalign(tmp_path):
    path = tmp_path / "bin" / "USalign"
    path.parent.mkdir()
    path.write_text(FAKE_USALIGN.format(python=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def test_calibrate_runs_every_stage_and_attaches_the_result(tmp_path, monkeypatch):
    from rfoldsearch.search import Searcher

    monkeypatch.delenv("USALIGN", raising=False)
    db, qdb, clusters, labels = write_inputs(tmp_path)
    output = tmp_path / "calibrated.pt"
    main(["calibrate", "--db", str(db), "--clusters", str(clusters),
          "--labels", str(labels), "--queries", str(qdb),
          "--db_type", "predicted", "--usalign", str(fake_usalign(tmp_path)),
          "--output", str(output)])

    work = tmp_path / "calibrated.calibration"
    for name in ("clustermax_params.json", "query_fits.tsv", "background.tsv",
                 "background.queries.tsv", "evalue_params.json",
                 "usalign_audit.tsv"):
        assert (work / name).exists(), name
    assert len((work / "usalign_audit.tsv").read_text().splitlines()) > 1
    params = json.loads((work / "evalue_params.json").read_text())
    assert params["scheme"] == "cluster-max-query-conditional-power"
    power = params["power_calibration"]
    assert power["a"] > 0 and power["b"] > 0
    assert power["validation_status"] == "background-audit-complete"
    assert power["background_scaling"] == "none"

    searcher = Searcher(str(output))
    assert searcher.power_ev is not None
    assert searcher.n_clusters == 24

    # The US-align results are cached in the work directory, so a rerun needs
    # no binary at all.
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    rerun = tmp_path / "rerun.pt"
    main(["calibrate", "--db", str(db), "--clusters", str(clusters),
          "--labels", str(labels), "--queries", str(qdb),
          "--db_type", "predicted", "--workdir", str(work),
          "--output", str(rerun)])
    assert rerun.exists()


def test_calibrate_without_usalign_lists_the_unreviewed_leaders(tmp_path,
                                                                monkeypatch):
    monkeypatch.delenv("USALIGN", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    db, qdb, clusters, labels = write_inputs(tmp_path)
    output = tmp_path / "calibrated.pt"
    with pytest.raises(SystemExit, match="lack US-align adjudication"):
        main(["calibrate", "--db", str(db), "--clusters", str(clusters),
              "--labels", str(labels), "--queries", str(qdb),
              "--db_type", "predicted", "--output", str(output)])
    assert not output.exists()
    missing = tmp_path / "calibrated.calibration" / "gumbel.missing-audit.tsv"
    assert missing.read_text().startswith("query_id\ttarget_id")


def test_calibrate_refuses_to_overwrite_the_output(tmp_path):
    db, qdb, clusters, labels = write_inputs(tmp_path)
    output = tmp_path / "calibrated.pt"
    output.write_bytes(b"")
    with pytest.raises(SystemExit, match="already exists"):
        main(["calibrate", "--db", str(db), "--clusters", str(clusters),
              "--labels", str(labels), "--queries", str(qdb),
              "--db_type", "predicted", "--output", str(output)])
