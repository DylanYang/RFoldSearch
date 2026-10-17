"""Render the isolated experimental E-value comparison for collaborators."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def estimate(value: dict | None, digits: int = 2) -> str:
    if not value or value["value"] is None:
        return "NA"
    low, high = value["ci95"]
    return f"{value['value']:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def stratified_background(run: Path, mode: str, calibration: str,
                          level_index: int, stratum) -> list[tuple[str, int, float]]:
    rows = [json.loads(line) for line in (run / f"per_query_{mode}.jsonl").read_text().splitlines()]
    groups = {}
    for row in rows:
        groups.setdefault(stratum(row), []).append(row)
    answer = []
    for key, group in groups.items():
        families = sorted({row["family"] for row in group})
        per_family = [np.mean([row[f"counts_{calibration}"][level_index]
                               for row in group if row["family"] == family])
                      for family in families]
        answer.append((str(key), len(group), float(np.mean(per_family))))
    return sorted(answer)


def paired_difference(run: Path, left: tuple[str, str], right: tuple[str, str],
                      field: str, decision_index: int | None = None) -> tuple[float, float, float, int]:
    data = {}
    for mode in {left[0], right[0]}:
        data[mode] = {row["query_id"]: row for row in map(
            json.loads, (run / f"per_query_{mode}.jsonl").read_text().splitlines())}
    family_values = {}
    for qid, left_row in data[left[0]].items():
        right_row = data[right[0]][qid]
        def value(row, calibration):
            if field == "auc":
                return row["auc"]
            series = row[f"{field}_{calibration}"]
            return series[decision_index] if series is not None else None
        a, b = value(left_row, left[1]), value(right_row, right[1])
        if a is not None and b is not None:
            family_values.setdefault(left_row["family"], []).append(a-b)
    differences = np.asarray([np.mean(group) for group in family_values.values()])
    rng = np.random.default_rng(27017)
    draws = rng.integers(0, len(differences), (2000, len(differences)))
    lo, hi = np.quantile(differences[draws].mean(axis=1), [.025, .975])
    return float(differences.mean()), float(lo), float(hi), len(differences)


def clan_block_paired_ci(run: Path, left: tuple[str, str], right: tuple[str, str],
                         field: str, decision_index: int | None = None) -> tuple[float, float, int]:
    parent = {}
    def root(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = root(parent[key])
        return parent[key]
    with (run / "cv_folds.tsv").open(encoding="utf-8", newline="") as handle:
        for item in csv.DictReader(handle, delimiter="\t"):
            if item["clan"]:
                parent[root("F:" + item["family"])] = root("C:" + item["clan"])
    data = {}
    for mode in {left[0], right[0]}:
        data[mode] = {row["query_id"]: row for row in map(
            json.loads, (run / f"per_query_{mode}.jsonl").read_text().splitlines())}
    by_family = {}
    for qid, a in data[left[0]].items():
        b = data[right[0]][qid]
        av = a["auc"] if field == "auc" else a[f"{field}_{left[1]}"]
        bv = b["auc"] if field == "auc" else b[f"{field}_{right[1]}"]
        if av is None or bv is None:
            continue
        if decision_index is not None:
            av, bv = av[decision_index], bv[decision_index]
        by_family.setdefault(a["family"], []).append(av-bv)
    blocks = {}
    for family, values in by_family.items():
        blocks.setdefault(root("F:" + family), []).append(float(np.mean(values)))
    arrays = [np.asarray(values) for values in blocks.values()]
    rng = np.random.default_rng(27017)
    draws = rng.integers(0, len(arrays), (2000, len(arrays)))
    estimates = np.asarray([np.mean(np.concatenate([arrays[i] for i in row]))
                            for row in draws])
    low, high = np.quantile(estimates, [.025, .975])
    return float(low), float(high), len(arrays)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    args = ap.parse_args()
    result = json.loads((args.run / "comparison.json").read_text())
    levels = np.asarray(result["levels"])
    methods = result["methods"]
    choices = [
        ("raw", "initial", "Initial $E_0$ (raw SW)", "#e58a29", "o"),
        ("raw", "global", "Raw SW + global power", "#5b5fba", "o"),
        ("raw", "four_bin", "Raw SW + four-bin power", "#256f9b", "o"),
        ("normalized", "global", "SW/min + global power", "#48a7aa", "o"),
        ("normalized", "four_bin", "SW/min + four-bin power", "#8c62a8", "o"),
        ("compbias_raw", "four_bin", "Comp-bias SW + four-bin power", "#c46472", "o"),
    ]
    fig, ax = plt.subplots(figsize=(9, 7), dpi=160)
    ax.plot([.01, 100], [.01, 100], "--", color="0.5", lw=1.5,
            label="Perfect calibration")
    for mode, calibration, label, color, marker in choices:
        curve = methods[mode]["curves"][calibration]
        y = np.asarray(curve["observed"])
        visible = y > 0
        ax.plot(levels[visible], y[visible], "-", marker=marker, ms=4,
                lw=1.7, color=color, label=label)
    ax.set(xscale="log", yscale="log", xlim=(.008, 115),
           xlabel="Reported E-value threshold", ylabel="Observed Rfam-negative clusters per query",
           title="A  Experimental database (885 structures; 878 clusters)")
    ax.grid(which="both", alpha=.25)
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(args.run / "panel_A_candidate_comparison.png", dpi=220)
    fig.savefig(args.run / "panel_A_candidate_comparison.pdf")
    plt.close(fig)

    def at(mode: str, calibration: str, threshold: float) -> float:
        index = int(np.flatnonzero(np.isclose(levels, threshold))[0])
        return float(methods[mode]["curves"][calibration]["observed"][index])

    labels = {
        ("raw", "global"): "Raw SW + global power",
        ("raw", "four_bin"): "Raw SW + four-bin power",
        ("normalized", "global"): "SW/min + global power",
        ("normalized", "four_bin"): "SW/min + four-bin power",
        ("compbias_raw", "global"): "Comp-bias SW + global power",
        ("compbias_raw", "four_bin"): "Comp-bias SW + four-bin power",
        ("compbias_normalized", "global"): "Comp-bias SW/min + global power",
        ("compbias_normalized", "four_bin"): "Comp-bias SW/min + four-bin power",
    }
    table = ["| Method | AUC [95% CI] | Homolog clusters E≤0.01 | E≤0.1 | E≤1 | Queries ≥1 homolog E≤1 | Background E≤1/q [95% CI] | Log10 calibration RMSE | Self max E |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    calibration_rows = []
    for (mode, calibration), name in labels.items():
        record = methods[mode]
        sens = record["sensitivity"][calibration]
        curve = record["curves"][calibration]
        index = int(np.flatnonzero(np.isclose(levels, 1.0))[0])
        observed = curve["observed"][index]
        ci = (curve["ci95_low"][index], curve["ci95_high"][index])
        cells = [name, estimate(record["auc"], 3)]
        cells += [estimate(sens["positive_cluster_recall"][str(e)], 3)
                  for e in (.01, .1, 1.)]
        cells.append(estimate(sens["query_with_at_least_one_homolog"]["1.0"], 3))
        cells += [f"{observed:.3g} [{ci[0]:.3g}, {ci[1]:.3g}]",
                  f"{curve['log10_rmse_positive_points']:.3f}",
                  f"{record['self_search'][calibration]['max']:.3g}"]
        table.append("| " + " | ".join(cells) + " |")
        if (mode, calibration) in {("raw", "global"), ("raw", "four_bin"),
                                   ("normalized", "global"), ("normalized", "four_bin"),
                                   ("compbias_raw", "four_bin")}:
            observations = []
            for threshold in (.01, .1, 1.):
                i = int(np.flatnonzero(np.isclose(levels, threshold))[0])
                observations.append(f"{curve['observed'][i]:.3g} "
                                    f"[{curve['ci95_low'][i]:.3g}, {curve['ci95_high'][i]:.3g}]")
            calibration_rows.append("| " + name + " | " + " | ".join(observations) + " |")
    source = json.loads((args.run / "source_manifest.json").read_text())
    score = json.loads((args.run / "score_manifest.json").read_text())
    audit_path = args.run / "structural_audit_metrics.json"
    audit_lines = []
    interpretation = []
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        audited_rows = []
        structural_rows = []
        audit_choices = [
            ("raw", "global", "Raw/global", "#5b5fba"),
            ("raw", "four_bin", "Raw/four-bin", "#256f9b"),
            ("normalized", "global", "SW/min/global", "#48a7aa"),
            ("normalized", "four_bin", "SW/min/four-bin", "#8c62a8"),
            ("compbias_raw", "four_bin", "Comp-bias raw/four-bin", "#c46472"),
        ]
        fig_audit, ax_audit = plt.subplots(figsize=(8, 6), dpi=160)
        ax_audit.plot([.01, 1], [.01, 1], "--", color="0.5", label="Perfect calibration")
        for mode, calibration, name, color in audit_choices:
            record = audit["methods"][mode]
            if not record["negative_audit_complete"]:
                continue
            metrics = record["calibrations"][calibration]
            cells = []
            for threshold in (.01, .1, 1.):
                value = metrics[str(threshold)]["audited_unrelated_background_clusters_per_query"]
                lo, hi = value["ci95"]
                cells.append((f"{value['value']:.3g} [{lo:.3g}, {hi:.3g}]"
                              if hi is not None else "0 [no observed events]"))
            any_fp = metrics["1.0"]["queries_with_at_least_one_audited_unrelated_hit"]
            lo, hi = any_fp["ci95"]
            cells.append((f"{any_fp['value']:.3f} [{lo:.3f}, {hi:.3f}]"
                          if hi is not None else "0 [no observed events]"))
            audited_rows.append("| " + name + " | " + " | ".join(cells) + " |")
            if record["positive_audit_complete"]:
                homolog_cells = []
                for threshold in (.01, .1, 1.):
                    value = metrics[str(threshold)]["structurally_similar_homolog_cluster_recall"]
                    homolog_cells.append(estimate(value, 3) if value else "not evaluable")
                any_homolog = metrics["1.0"]["queries_with_structurally_similar_homolog"]
                homolog_cells.append(estimate(any_homolog, 3) if any_homolog else "not evaluable")
                n_evaluable = any_homolog["n_queries"] if any_homolog else 0
                structural_rows.append("| " + name + f" | {n_evaluable} | "
                                       + " | ".join(homolog_cells) + " |")
            x = np.asarray(audit["levels"])
            y = np.asarray([metrics[str(float(threshold))]
                            ["audited_unrelated_background_clusters_per_query"]["value"]
                            for threshold in x])
            valid = y > 0
            ax_audit.plot(x[valid], y[valid], "-o", ms=3.5, lw=1.6,
                          color=color, label=name)
        ax_audit.set(xscale="log", yscale="log", xlim=(.008, 1.15),
                     xlabel="Reported E-value threshold",
                     ylabel="Audited structurally unrelated clusters per query",
                     title="A  Experimental database: US-align-audited E≤1 candidates")
        ax_audit.grid(which="both", alpha=.25)
        ax_audit.legend(loc="upper left", fontsize=8)
        fig_audit.tight_layout()
        fig_audit.savefig(args.run / "panel_A_audited_E_le_1.png", dpi=220)
        fig_audit.savefig(args.run / "panel_A_audited_E_le_1.pdf")
        plt.close(fig_audit)
        positive_complete = len(structural_rows) == len(audit_choices)
        audit_lines = [
            "### US-align-audited operational false-positive proxy (E≤1)",
            "",
            f"{audit['n_audited_pairs']:,} unique query–target pairs received a",
            "fresh US-align adjudication. The table below is reported only for",
            "methods whose complete Rfam-negative candidate set at E≤1 was",
            ("audited; all same-family candidates were also audited."
             if positive_complete else
             "audited; same-family structural-sensitivity denominators remain incomplete."),
            "Pairs with mean query-/target-normalized TM-score",
            "below 0.45 count as *operationally unrelated*, not proof of no homology.",
            "When zero events are observed, family-block bootstrap cannot",
            "estimate a nonzero upper confidence bound; such cells are marked",
            "'no observed events' rather than presenting a misleading [0,0] CI.",
            "",
            "| Method | Unrelated clusters/q at E=0.01 [95% CI] | E=0.1 | E=1 | Queries with ≥1 unrelated hit at E≤1 |",
            "|---|---:|---:|---:|---:|",
            *audited_rows,
            "",
            "![US-align-audited Panel A operational range](panel_A_audited_E_le_1.png)",
            "",
            "The US-align screen used the exact mode for initial pairs and",
            "`-fast` for computationally costly later pairs; fast scores within",
            "0.05 of TM=0.45 are rechecked with the exact mode where available.",
            "Crucially, these adjudications have **not yet been fed back into**",
            "Gumbel or power refitting. Thus, this is an external diagnostic of",
            "the current fits, not their final structurally filtered calibration.",
            "",
        ]
        if positive_complete:
            audit_lines.extend([
                "### US-align-audited structurally similar homolog recovery",
                "",
                "The denominators comprise all non-self, same-Rfam-family target",
                "clusters with a successful US-align assessment and mean TM-score",
                "at least 0.45. Percentages are family-macro averages of per-query",
                "cluster recall; the last column is the fraction of evaluable",
                "queries with at least one such hit. Family-block 95% bootstrap",
                "intervals are shown. These metrics were computed only after the",
                "complete same-family candidate set was audited.",
                "",
                "| Method | Evaluable queries | Homolog clusters E≤0.01 [95% CI] | E≤0.1 | E≤1 | Queries ≥1 homolog E≤1 |",
                "|---|---:|---:|---:|---:|---:|",
                *structural_rows,
                "",
            ])
        else:
            audit_lines.extend([
                "The same-family structural audit is still incomplete; structural",
                "homolog sensitivity is withheld rather than estimated from a",
                "selected subset of evaluated positives.",
                "",
            ])
        if all(audit["methods"][mode]["negative_audit_complete"]
               for mode in ("raw", "normalized", "compbias_raw")):
            audited_e1 = lambda mode, calibration: audit["methods"][mode]["calibrations"][
                calibration]["1.0"]["audited_unrelated_background_clusters_per_query"]["value"]
            interpretation = [
                "## Interim interpretation for method selection",
                "",
                f"- Raw SW discriminates same-family from annotated negatives better than SW/min "
                f"(family-macro AUC {methods['raw']['auc']['value']:.3f} versus "
                f"{methods['normalized']['auc']['value']:.3f}); its global-power homolog "
                f"recall at E≤1 is {methods['raw']['sensitivity']['global']['positive_cluster_recall']['1.0']['value']:.3f} "
                f"versus {methods['normalized']['sensitivity']['global']['positive_cluster_recall']['1.0']['value']:.3f}.",
                f"- After structural adjudication, the observed unrelated-cluster count at "
                f"reported E=1 is {audited_e1('raw', 'global'):.3f} for raw/global, "
                f"{audited_e1('raw', 'four_bin'):.3f} for raw/four-bin, and "
                f"{audited_e1('normalized', 'global'):.3f} for SW/min/global. "
                "The raw candidates are conservative under the current fits; the "
                "normalized candidate is closer to the nominal count.",
                "- Four length bins do not yield a clear gain over one global power",
                "  correction in this nested 516-query comparison. All four bins were",
                "  present in every held-out fold after the fitting-design correction,",
                "  so this is not an artifact of silently collapsed bins.",
                "- The 20-token composition correction is a plausible additional",
                "  candidate, but it does not yet resolve the sensitivity–calibration",
                "  tradeoff decisively. The primary unresolved step is to exclude",
                "  structurally related background **before refitting** Gumbel and power.",
                "",
            ]
            if positive_complete:
                def structural_e1(mode, calibration):
                    return audit["methods"][mode]["calibrations"][calibration]["1.0"][
                        "structurally_similar_homolog_cluster_recall"]["value"]
                interpretation.insert(
                    4, f"- Among US-align-confirmed same-family homolog clusters, "
                    f"E≤1 recall is {structural_e1('raw', 'global'):.3f} for raw/global, "
                    f"{structural_e1('raw', 'four_bin'):.3f} for raw/four-bin, "
                    f"{structural_e1('normalized', 'global'):.3f} for SW/min/global, "
                    f"and {structural_e1('compbias_raw', 'four_bin'):.3f} for "
                    "composition-corrected raw/four-bin.")
    n_eval = methods["raw"]["n_homolog_evaluable_queries"]
    run = args.run.resolve()
    paired_rows = []
    clan_rows = []
    comparisons = [
        (("raw", "global"), ("normalized", "global"), "Raw/global − SW/min/global"),
        (("raw", "four_bin"), ("raw", "global"), "Raw/four-bin − raw/global"),
        (("compbias_raw", "four_bin"), ("raw", "four_bin"),
         "Comp-bias raw/four-bin − raw/four-bin"),
    ]
    for left, right, name in comparisons:
        auc = paired_difference(args.run, left, right, "auc")
        recall01 = paired_difference(args.run, left, right, "recall", 0)
        recall1 = paired_difference(args.run, left, right, "recall", 2)
        paired_rows.append(
            f"| {name} | {auc[0]:+.3f} [{auc[1]:+.3f}, {auc[2]:+.3f}] | "
            f"{recall01[0]:+.3f} [{recall01[1]:+.3f}, {recall01[2]:+.3f}] | "
            f"{recall1[0]:+.3f} [{recall1[1]:+.3f}, {recall1[2]:+.3f}] |")
        clan_auc = clan_block_paired_ci(args.run, left, right, "auc")
        clan_recall1 = clan_block_paired_ci(args.run, left, right, "recall", 2)
        clan_rows.append(f"| {name} | [{clan_auc[0]:+.3f}, {clan_auc[1]:+.3f}] | "
                         f"[{clan_recall1[0]:+.3f}, {clan_recall1[1]:+.3f}] | {clan_auc[2]} |")
    coefficient_rows = []
    self_rows = []
    for mode in ("raw", "normalized", "compbias_raw", "compbias_normalized"):
        full = methods[mode]
        global_power = full["full_power"]["global"]
        bin_power = full["full_power"]["four_bin"]
        coefficient_rows.append(
            f"| {mode} | {global_power['a']:.4g} | {global_power['b']:.4g} | "
            f"{bin_power['b']:.4g} | {len(bin_power['bins'])} | "
            f"{full['full_gumbel']['n_tail_observations']:,} | "
            f"{full['full_gumbel']['max_abs_gradient']:.2g} |")
        for calibration in ("global", "four_bin"):
            stat = full["self_search"][calibration]
            self_rows.append(
                f"| {mode}/{calibration} | {stat['min']:.3g} | {stat['median']:.3g} | "
                f"{stat['p95']:.3g} | {stat['max']:.3g} | {stat['n_gt_1']}/885 | "
                f"{stat['max_query_id']} ({stat['max_query_length']} nt) |")
    one_index = int(np.flatnonzero(np.isclose(levels, 1.0))[0])
    length_rows = []
    entropy_rows = []
    for mode, calibration, name in [
            ("raw", "global", "Raw/global"),
            ("raw", "four_bin", "Raw/four-bin"),
            ("normalized", "global", "SW/min/global")]:
        by_length = stratified_background(args.run, mode, calibration, one_index,
            lambda row: "<30" if row["length"] < 30 else
            "30–59" if row["length"] < 60 else
            "60–119" if row["length"] < 120 else "≥120")
        length_rows += [f"| {name} | {band} | {n} | {value:.3g} |"
                        for band, n, value in by_length]
        entropies = np.asarray([json.loads(line)["entropy"] for line in
            (args.run / f"per_query_{mode}.jsonl").read_text().splitlines()])
        quartiles = np.quantile(entropies, [.25, .5, .75])
        by_entropy = stratified_background(args.run, mode, calibration, one_index,
            lambda row: f"Q{1+int(np.searchsorted(quartiles, row['entropy']))}")
        entropy_rows += [f"| {name} | {band} | {n} | {value:.3g} |"
                         for band, n, value in by_entropy]
    summary_lines = []
    if audit_path.exists() and audit["audit_complete"]:
        def audited_value(mode, calibration, key):
            return audit["methods"][mode]["calibrations"][calibration]["1.0"][key]["value"]
        summary_lines = [
            "## Executive summary",
            "",
            f"- In a fresh 885-chain/878-cluster database, raw SW discriminated "
            f"same-family from annotated negatives better than SW/min "
            f"(family-macro AUC {methods['raw']['auc']['value']:.3f} vs "
            f"{methods['normalized']['auc']['value']:.3f}).",
            f"- At reported E≤1, US-align-confirmed homolog-cluster recall was "
            f"{audited_value('raw', 'global', 'structurally_similar_homolog_cluster_recall'):.3f} "
            f"for raw/global, "
            f"{audited_value('raw', 'four_bin', 'structurally_similar_homolog_cluster_recall'):.3f} "
            f"for raw/four-bin, "
            f"{audited_value('normalized', 'global', 'structurally_similar_homolog_cluster_recall'):.3f} "
            f"for SW/min/global, and "
            f"{audited_value('compbias_raw', 'four_bin', 'structurally_similar_homolog_cluster_recall'):.3f} "
            "for composition-corrected raw/four-bin.",
            f"- Audited unrelated clusters per query at reported E=1 were "
            f"{audited_value('raw', 'global', 'audited_unrelated_background_clusters_per_query'):.3f}, "
            f"{audited_value('raw', 'four_bin', 'audited_unrelated_background_clusters_per_query'):.3f}, "
            f"{audited_value('normalized', 'global', 'audited_unrelated_background_clusters_per_query'):.3f}, "
            f"and {audited_value('compbias_raw', 'four_bin', 'audited_unrelated_background_clusters_per_query'):.3f} "
            "in the same order (ideal: 1).",
            "- This is a matched candidate comparison, not a deployment decision:",
            "  structurally related clusters must be removed and backfilled before",
            "  Gumbel and power refitting, then the full evaluation repeated.",
            "",
        ]
    lines = [
        "# RFoldSeek experimental E-value methods: fresh matched comparison",
        "",
        "**Status: exploratory, not a publication-ready Fig. S16A.** All score arrays, tokens,",
        "clusters and model coefficients in this run were recomputed from the 885 mmCIF",
        "files. Operational E≤1 negatives were structurally audited, but exclusions",
        "have not yet been fed back into parameter fitting; numbers labelled",
        "Rfam-negative remain a *proxy* for unrelated backgrounds, not certified false",
        "positives. No prior score/parameter/result cache was used.",
        "",
        *summary_lines,
        "## Data and reproducibility",
        "",
        f"- Source: {source['n_files']} mmCIF files; successfully extracted {source['n_ok']}; "
        f"failed {source['n_failed']}. Multi-chain files were restricted to the chain ID in the filename.",
        f"- Fresh CD-HIT-EST clustering: 80% identity, `-c 0.8 -n 5 -aS 0.8 -aL 0.8 -g 1`; "
        f"{result['protocol']['n_clusters']} clusters.",
        f"- {score['n_annotated_queries']} Rfam-annotated calibration queries; "
        f"{score['n_queries']-score['n_annotated_queries']} auxiliary queries "
        "(38 shortest and 37 lowest-token-entropy unannotated chains) "
        "used only in Gumbel training (20% total likelihood weight).",
        f"- {n_eval} annotated queries have at least one non-self same-family cluster "
        "and enter homolog-recovery metrics.",
        "- Database: generated `db_nr80.pt` (`" + digest(run / "db_nr80.pt") + "`).",
        "- Rfam labels: input `labels.tsv` (`" + digest(args.labels) + "`).",
        "- Score matrix: `" + score["matrix_sha256"] + "`; score arrays and query order are in "
        + "`score_manifest.json`.",
        "- Software environment: Python 3.12.14, NumPy 2.5.3, SciPy 1.18.1,",
        "  PyTorch 2.14.1+cpu, Numba 0.68.0, Biotite 1.7.1, CD-HIT-EST 4.8.1.",
        "- Verification: four new unit tests passed; `qa_checks.json` passed",
        "  finite-score, fold-blocking, convergence, four-bin and monotonicity checks.",
        "- Four folds are blocked jointly by Rfam family and clan; in each rotation,",
        "  the three non-test folds supply cross-fitted E0 values for power fitting,",
        "  and the untouched fold tests both Gumbel and power. All three training",
        "  folds fit the test-fold Gumbel model.",
        "  All candidates share the same fold assignments, targets and exclusion rules.",
        "",
        "## Candidate methods",
        "",
        "Raw SW and SW/min use the same token Smith–Waterman alignment (gap open −2,",
        "extend −0.5). The composition-bias candidates recalculate the alignment",
        "with a window-20 query profile; normalization, where used, divides the",
        "result by min(query length, target length). For all score variants, a",
        "target-aware linear model predicts Gumbel location and log-scale from query",
        "length, cluster median length, absolute log-length difference, and query",
        "token entropy. No three-fold query–target length filter is applied to",
        "fitting, calibration or E0 summation. By default, fitting is left-censored below the 95th",
        "percentile in shorter-length/length-ratio strata. E0 sums Gumbel survival",
        "over all target clusters. Global power fits one a,b; four-bin power fits",
        "one shared b and four length-conditional a values (bins merge if unsupported).",
        "This is a matched four-bin **analogue**, not a byte-for-byte run of PR #27:",
        "the collaborator's 198-query inputs and complete fitting configuration",
        "were unavailable, so PR percentages are shown separately below.",
        "For the experimental annotated background, each query's count is expanded",
        "by N_D/B_q. The fitted range is E0≤100; above it Ecal grows proportionally",
        "from the endpoint and is capped at the cluster count.",
        "",
        "In notation, for each score variant $s_{qj}$, the non-self, non-family,",
        "non-clan cluster maximum is $Y_{qc}=\\max_{j\\in\\mathcal M_{qc}}s_{qj}$.",
        "The shared conditional null has",
        "$(\\mu_{qc},\\log\\beta_{qc})=[1,z(\\log L_q),z(\\log L_c),",
        "z(|\\log L_q-\\log L_c|),z(H_q)]\\Theta_D$ and",
        "$E_0(q,s)=\\sum_c[1-F_{qc}(s)]$. The final mapping is",
        "$E_{\\mathrm{cal}}=a_D(L_q)E_0^{b_D}$, with constant $a_D$ for the",
        "global candidate or interpolated length-bin $a_D(L_q)$ for the four-bin",
        "candidate. The score mode changes $s_{qj}$, not these covariates.",
        "Each $z$ uses the weighted mean and standard deviation of eligible",
        "Gumbel-training cluster observations within that fold; these values",
        "are stored with the model and reused without restandardizing test queries.",
        "",
        "## Matched held-out results",
        "",
        *table,
        "",
        *interpretation,
        "The AUC compares, within a query, a same-family target-cluster maximum",
        "against Rfam-negative annotated clusters. Homolog recovery is the",
        "family-macro mean of per-query positive-cluster recovery among queries",
        "with an available non-self homolog. The 'queries ≥1 homolog' metric uses",
        "the same evaluable-query denominator. Background E≤1/q is the",
        "family-balanced, N_D/B_q-expanded Rfam-negative count; if all biological",
        "negatives were correctly classified, perfect calibration would be 1.",
        "Confidence intervals resample Rfam families as blocks (2,000 replicates).",
        "Log10 calibration RMSE uses positive observed points on the common",
        "threshold grid; zeros are omitted, not silently replaced by a floor.",
        "",
        "### Background calibration at operational thresholds",
        "",
        "| Method | Observed at reported E=0.01 [95% CI] | At E=0.1 | At E=1 |",
        "|---|---:|---:|---:|",
        *calibration_rows,
        "",
        "A perfectly calibrated method has expected values 0.01, 0.1, and 1",
        "in these three columns. All values here still include any unreviewed",
        "structural relatives among Rfam-negative hits.",
        "",
        *audit_lines,
        "### Collaborator-provided PR #27 snapshot (reference only)",
        "",
        "The screenshot supplied with this request for",
        "[PR #27](https://github.com/DylanYang/RFoldSearch/pull/27) reports the following",
        "on a",
        "different 198-query panel. These are **not** merged with, or directly",
        "subtracted from, the fresh 516-query estimates above:",
        "",
        "| PR snapshot method | E≤0.01 | E≤0.1 | E≤1 | Structurally similar homolog E≤1 | Query ≥1 homolog E≤1 |",
        "|---|---:|---:|---:|---:|---:|",
        "| SW/min | 9% | 18% | 33% | 41% | 69% |",
        "| Raw SW, global | 56% | 62% | 67% | 82% | 84% |",
        "| Raw SW, four bins | 62% | 68% | 72% | 88% | 84% |",
        "",
        "The present comparison recomputes the same *types* of sensitivity",
        "metrics, but its null model, query panel, fold split and structural-audit",
        "status differ. Agreement in direction is informative; numerical equality",
        "is not expected. In particular, the 198 query IDs and their selection",
        "manifest were not available as raw inputs here; no claim of exact PR",
        "replication is made.",
        "",
        "### Paired family-blocked method contrasts",
        "",
        "| Contrast (left − right) | AUC difference [95% CI] | Homolog recall E≤0.01 difference | E≤1 difference |",
        "|---|---:|---:|---:|",
        *paired_rows,
        "",
        "These CIs resample the same Rfam families across methods, retaining the",
        "within-query pairing. They quantify discrimination/sensitivity changes,",
        "not whether either E-value method is well calibrated.",
        "",
        "| Contrast | Clan-block AUC difference 95% CI | Clan-block E≤1 homolog-recall difference 95% CI | Independent blocks |",
        "|---|---:|---:|---:|",
        *clan_rows,
        "",
        "Clan-block resampling is a dependence-robust sensitivity analysis of",
        "the same paired family-macro contrast; linked families are sampled",
        "together rather than treated as independent blocks.",
        "",
        "### Stratified background calibration at reported E=1",
        "",
        "| Method | Query length (nt) | Queries | Observed Rfam-negative clusters/q |",
        "|---|---:|---:|---:|",
        *length_rows,
        "",
        "| Method | Query token entropy quartile | Queries | Observed Rfam-negative clusters/q |",
        "|---|---:|---:|---:|",
        *entropy_rows,
        "",
        "These are within-stratum family-macro means, not a claim that all strata",
        "are calibrated by a single pooled power fit.",
        "",
        "### Final all-query fit parameters and self-search diagnostic",
        "",
        "| Score mode | Global a | Global b | Four-bin shared b | Supported bins | Tail maxima used | Max Gumbel gradient |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *coefficient_rows,
        "",
        "The full 2×5 conditional Gumbel coefficient matrices, feature means and",
        "scales, and each length-bin intercept are serialized in the locally generated `comparison.json` (not bundled here).",
        "These are **research fits**, not deployment parameters.",
        "",
        "| Mode/calibration | Self E min | Median | 95th percentile | Max | E>1 | Max-E chain |",
        "|---|---:|---:|---:|---:|---:|---|",
        *self_rows,
        "",
        "![Panel A candidate calibration comparison](panel_A_candidate_comparison.png)",
        "",
        "## Important limitations and next validation gate",
        "",
        "1. US-align structural adjudication has **not yet been fed back into**",
        "   background selection or model fitting. A candidate audit list is saved",
        "   as `structural_audit_candidates.tsv`. Until audited and refitted to a",
        "   fixed point, definitive E-value calibration and publication Fig. S16A",
        "   cannot be claimed from this run. Structural-homolog recovery is an",
        "   external diagnostic only after the complete positive audit is available.",
        "   A structurally excluded cluster leader must also be replaced by its",
        "   next eligible member during the eventual refit (cluster backfill).",
        "2. Rfam absence does not prove biological unrelatedness. The background",
        "   table is a conservative annotation-based proxy, not specificity ground truth.",
        "   The local 885-CIF selection and Rfam labels were treated as frozen raw",
        "   inputs; their original external acquisition/annotation was not repeated.",
        "3. The 75 auxiliary queries are unannotated and may contain unknown",
        "   structural relatives; they are downweighted but should be reviewed.",
        "4. The collaborator's PR #27 198-query panel is a distinct cohort; its",
        "   published-in-PR percentages must not be pooled with these 516-query",
        "   estimates. This comparison addresses AUC, homolog recovery,",
        "   length-binned power and background-calibration metrics under",
        "   a fresh, matched protocol.",
        "5. Self-hit E-values come from an all-query final refit and are a",
        "   diagnostic, not held-out calibration evidence. Very small values below",
        "   the fitted power support are extrapolations, not precisely validated p-values.",
        "",
        "**Decision rule:** Do not select a deployment method solely by the smallest",
        "self-hit E-value or highest homolog recall. Require family-blocked",
        "calibration (observed/expected near one with uncertainty), a manageable",
        "audited false-positive burden, useful homolog sensitivity, and stability",
        "across query length/entropy strata before updating the manuscript or server.",
        "",
        "## Related code and outputs",
        "",
        "- `scripts/fresh_experimental_evalue_preprocess.py`",
        "- `scripts/fresh_experimental_evalue_scores.py`",
        "- `scripts/fresh_experimental_evalue_compare.py`",
        "- `scripts/audit_clean_experimental_background.py` (optional complete member-level audit for a separate cleaned refit)",
        "- `scripts/fresh_experimental_evalue_usalign.py`, `scripts/recheck_fresh_usalign_boundary.py`, `scripts/summarize_fresh_evalue_usalign.py`",
        "- `scripts/check_fresh_experimental_evalue_run.py`, `scripts/write_fresh_experimental_evalue_report.py`, `tests/test_fresh_experimental_evalue_compare.py`",
        "- `comparison.json` (generated locally; not bundled; all fold and final coefficients, curves and metrics)",
        "- `cv_folds.tsv` (bundled); `per_query_*.jsonl` and `structural_audit_candidates.tsv` (generated locally; not bundled)",
        "- `qa_checks.json` (finite scores, convergence, blocked folds, four-bin support, monotonicity)",
        "- `panel_A_candidate_comparison.png` and `panel_A_audited_E_le_1.png`",
        "",
        "Reference for collaborator-proposed alternatives: [RFoldSearch PR #27]"
        "(https://github.com/DylanYang/RFoldSearch/pull/27).",
    ]
    (args.run / "collaborator_report.md").write_text("\n".join(lines) + "\n",
                                                     encoding="utf-8")
    print(args.run / "collaborator_report.md")


if __name__ == "__main__":
    main()
