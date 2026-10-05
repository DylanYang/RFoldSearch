import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def test_rebuild_command_writes_query_conditional_model(tmp_path):
    rng = np.random.default_rng(17)
    entries = []
    for index in range(25):
        length = 24 + index
        alphabet = 3 + index % 12
        entries.append({
            "id": f"chain{index}",
            "tokens": rng.integers(0, alphabet, size=length).tolist(),
            "cluster": f"cluster{index}",
        })
    database = tmp_path / "db.pt"
    torch.save({"n_codes": 80, "entries": entries}, database)
    labels = tmp_path / "labels.tsv"
    labels.write_text(
        "chain_id\tfamily\tclan\n" +
        "".join(f"chain{index}\tRF{index:05d}\t\n" for index in range(25))
    )
    panel = tmp_path / "panel.tsv"
    panel.write_text("query_id\nchain0\nchain7\nchain20\n")
    output = tmp_path / "params.json"
    fits = tmp_path / "query_fits.tsv"
    adjudication = tmp_path / "adjudication.tsv"
    adjudication.write_text(
        "query_id\ttarget_id\tstructurally_related\n" +
        "".join(f"chain{query}\tchain{target}\tfalse\n"
                for query in (0, 7, 20) for target in range(25)
                if query != target)
    )
    script = Path(__file__).resolve().parents[1] / "scripts/rebuild_evalue_params.py"

    unaudited = subprocess.run([
        sys.executable, str(script), "--db", str(database),
        "--labels", str(labels), "--query-panel", str(panel),
        "--query-fits-output", str(fits), "--output", str(output),
    ], capture_output=True, text=True)
    assert unaudited.returncode != 0
    assert "lack US-align adjudication" in unaudited.stderr
    assert not output.exists()
    assert output.with_name("params.missing-audit.tsv").exists()

    subprocess.run([
        sys.executable, str(script), "--db", str(database),
        "--labels", str(labels), "--query-panel", str(panel),
        "--related-pairs", str(adjudication),
        "--query-fits-output", str(fits), "--output", str(output),
    ], check=True)

    params = json.loads(output.read_text())
    assert params["scheme"] == "cluster-max-query-conditional"
    assert params["search_scope"] == "full_database"
    assert params["n_calibration_queries"] == 3
    assert params["validation_status"] == "background-audit-complete"
    assert params["structural_audit_ceiling"] == 100.0
    assert len(params["target_database_fingerprint"]) == 64
    assert len(params["calibration_context_fingerprint"]) == 64
    assert np.asarray(params["query_parameter_model"][
        "coefficients_mu_logbeta"]).shape == (3, 2)
    assert "length_table" not in params
    assert len(fits.read_text().splitlines()) == 4
    assert not output.with_name("params.missing-audit.tsv").exists()
