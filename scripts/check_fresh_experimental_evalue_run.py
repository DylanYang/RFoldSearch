"""Fail closed on internal consistency of the new experimental comparison."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--require-clean-audit", action="store_true")
    ap.add_argument("--reference-folds", type=Path)
    args = ap.parse_args()
    result = json.loads((args.run / "comparison.json").read_text())
    manifest = json.loads((args.run / "score_manifest.json").read_text())
    if args.require_clean_audit:
        from scripts.fresh_experimental_evalue_compare import file_sha256
        audit = json.loads((args.run / "structural_members.json").read_text())
        assert audit["complete"] and audit["remaining_pairs"] == 0
        assert result["protocol"]["structural_audit_sha256"] == file_sha256(args.run / "structural_members.tsv")
    if args.reference_folds:
        assert (args.run / "cv_folds.tsv").read_text() == args.reference_folds.read_text()
    assert result["protocol"]["n_structures"] == manifest["n_targets"] == 885
    assert result["protocol"]["n_annotated_queries"] == manifest["n_annotated_queries"] == 516
    for filename in ("raw_sw.npy", "compbias20_sw.npy"):
        scores = np.load(args.run / filename, mmap_mode="r")
        assert scores.shape == (manifest["n_queries"], manifest["n_targets"])
        assert np.isfinite(scores).all() and np.all(scores >= 0)
    with (args.run / "cv_folds.tsv").open(encoding="utf-8", newline="") as handle:
        folds = list(csv.DictReader(handle, delimiter="\t"))
    by_family, by_clan = {}, {}
    for row in folds:
        by_family.setdefault(row["family"], set()).add(row["fold"])
        if row["clan"]:
            by_clan.setdefault(row["clan"], set()).add(row["fold"])
    assert all(len(indices) == 1 for indices in by_family.values())
    assert all(len(indices) == 1 for indices in by_clan.values())
    levels = np.asarray(result["levels"])
    assert np.all(np.diff(levels) > 0)
    for mode, record in result["methods"].items():
        models = [record["full_gumbel"]] + [fold["gumbel"]
                                                 for fold in record["fold_models"]]
        models += [inner["gumbel"] for fold in record["fold_models"]
                   for inner in fold["inner_gumbel"]]
        assert len(record["fold_models"]) == 4 and len(models) == 17
        for model in models:
            assert model["max_abs_gradient"] < 2e-3
            assert np.isfinite(model["coefficients"]).all()
            assert model["n_tail_observations"] > 0
        powers = [record["full_power"]] + [fold["power"]
                                               for fold in record["fold_models"]]
        for pair in powers:
            for name in ("global", "four_bin"):
                model = pair[name]
                assert model["a"] > 0 and model["b"] > 0
                assert model["fit_min"] <= model["fit_max"] == 100
            bins = pair["four_bin"]["bins"]
            assert 1 <= len(bins) <= 4, (mode, len(bins))
            assert np.all(np.diff([b["log_length_centre"] for b in bins]) > 0)
        for calibration, curve in record["curves"].items():
            values = np.asarray(curve["observed"])
            assert len(values) == len(levels)
            assert np.all(np.diff(values) >= -1e-9), (mode, calibration)
            assert np.all(values >= 0)
        for calibration, values in record["self_search"].items():
            assert 0 <= values["min"] <= values["median"] <= values["p95"] <= values["max"]
            assert values["max"] <= result["protocol"]["n_clusters"]
        assert 0 <= record["auc"]["value"] <= 1
    output = {"status": "passed", "n_score_modes": len(result["methods"]),
              "n_cv_models_per_mode": 17, "n_query_folds": 4,
              "supported_bin_counts": {mode: [len(f["power"]["four_bin"]["bins"])
                                               for f in record["fold_models"]]
                                       for mode, record in result["methods"].items()},
              "structural_audit_required": args.require_clean_audit,
              "family_and_clan_blocking": True,
              "score_arrays_finite_and_nonnegative": True,
              "calibration_curves_monotone": True}
    (args.run / "qa_checks.json").write_text(json.dumps(output, indent=2) + "\n",
                                               encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
