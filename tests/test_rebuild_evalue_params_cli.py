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
    script = Path(__file__).resolve().parents[1] / "scripts/rebuild_evalue_params.py"

    subprocess.run([
        sys.executable, str(script), "--db", str(database),
        "--labels", str(labels), "--query-panel", str(panel),
        "--query-fits-output", str(fits), "--output", str(output),
    ], check=True)

    params = json.loads(output.read_text())
    assert params["scheme"] == "cluster-max-query-conditional"
    assert params["search_scope"] == "full_database"
    assert params["n_calibration_queries"] == 3
    assert np.asarray(params["query_parameter_model"][
        "coefficients_mu_logbeta"]).shape == (3, 2)
    assert "length_table" not in params
    assert len(fits.read_text().splitlines()) == 4
