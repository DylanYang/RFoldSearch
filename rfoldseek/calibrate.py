"""Database-specific E-value calibration, from one command.

Calibration has three stages, each of which used to be a separate script:

1. fit the query-conditional cluster-max Gumbel model (``E0``);
2. score independent calibration queries against every target cluster to
   collect one background observation per query and cluster;
3. fit the power law ``E_cal = a_D * E0**b_D`` to that background.
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import evalue as EV
from .database import (attach_evalue_calibration, calibration_context_fingerprint,
                       database_fingerprint, load_database, read_clusters,
                       short_id, strict_cluster_labels)
from .search import GAP_EXTEND, GAP_OPEN, load_matrix
from .compbias import background_frequencies, query_profile
from .smith_waterman import smith_waterman_numba
from .usalign import USalignNotFound, find_usalign, score_pool

UNKNOWN = {"", "N/A", "NA", "None", "none", ".", "-"}
TM_THRESHOLD = 0.45
AUDIT_CEILING = 100.0
# Background scope and weight scaling for each kind of target database.
DATABASE_TYPES = {
    "predicted": ("all", "none"),
    "experimental": ("annotated", "annotated-to-full"),
}
DEFAULT_LENGTHS = (20, 30, 40, 60, 80, 120, 160, 220, 300, 500, 1000, 2000)
MIN_QUERIES_PER_LENGTH_BIN = 20


class CalibrationInputError(ValueError):
    """Inputs that cannot be combined; the scripts report these as usage errors."""


class AuditIncomplete(RuntimeError):
    """Cluster leaders still lack a US-align result."""


# ---------------------------------------------------------------- input files

def first_present(row, names):
    for name in names:
        value = (row.get(name) or "").strip()
        if value:
            return value
    return ""


def truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def known_equal(first, second):
    return first not in UNKNOWN and second not in UNKNOWN and first == second


def read_labels(path):
    """Chain id -> {"family", "clan"}, keyed by both raw and short ids."""
    labels = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            chain_id = first_present(row, ("chain_id", "entry_id", "id",
                                           "query_id", "target_id"))
            if not chain_id:
                raise ValueError("labels TSV needs an id column")
            annotation = {
                "family": first_present(row, ("family", "rfam_acc", "rfam_family")),
                "clan": first_present(row, ("clan", "clan_acc", "rfam_clan")),
            }
            labels[chain_id] = annotation
            labels[short_id(chain_id, chain_id)] = annotation
    return labels


def read_query_panel(path):
    """Ordered query ids from a TSV, or None for no panel."""
    if path is None:
        return None
    query_ids = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            query_id = first_present(row, ("query_id", "chain_id", "entry_id", "id"))
            if not query_id:
                raise ValueError("query panel needs a query_id column")
            query_ids.append(query_id)
    if not query_ids:
        raise ValueError("query panel is empty")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("query panel contains duplicate query ids")
    return query_ids


def _aliases(query, target):
    return {(query, target), (short_id(query, query), short_id(target, target))}


def add_adjudication(audited, related, query, target, is_related):
    aliases = _aliases(query, target)
    audited.update(aliases)
    audited.update((second, first) for first, second in aliases)
    if is_related:
        related.update(aliases)
        related.update((second, first) for first, second in aliases)


def read_adjudication(path, tm_threshold=TM_THRESHOLD, audited=None, related=None):
    """US-align adjudication TSV -> (audited pairs, related pairs).

    A pair is related when marked so, or when the mean of its two
    length-normalised TM-scores reaches ``tm_threshold``.
    """
    audited = set() if audited is None else audited
    related = set() if related is None else related
    if path is None:
        return audited, related
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            query = first_present(row, ("query_id", "query"))
            target = first_present(row, ("target_id", "target"))
            marked = first_present(row, ("structurally_related",)).lower()
            tm_query = first_present(row, ("tm_query_normalized", "tm_query"))
            tm_target = first_present(row, ("tm_target_normalized", "tm_target"))
            tm_mean = first_present(row, ("tm_mean", "tm_symmetric", "tm_sym"))
            is_related = marked in {"1", "true", "yes", "y"}
            has_result = marked in {"0", "1", "false", "true", "no", "yes", "n", "y"}
            if tm_query and tm_target:
                has_result = True
                is_related = is_related or 0.5 * (
                    float(tm_query) + float(tm_target)) >= tm_threshold
            elif tm_mean:
                has_result = True
                is_related = is_related or float(tm_mean) >= tm_threshold
            elif first_present(row, ("tm_symmetric_min",)):
                raise ValueError(
                    "tm_symmetric_min is not the mean TM-score; provide both "
                    "normalized scores or a tm_mean column")
            if query and target and has_result:
                add_adjudication(audited, related, query, target, is_related)
    return audited, related


def _write_tsv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------- stage 1: query Gumbel model

def fit_clustermax_params(db_path, entries, tokens, chain_ids, n_codes, clusters,
                          labels, matrix, audited, related, *, query_panel=None,
                          model="query-conditional", audit_ceiling=AUDIT_CEILING,
                          allow_unreviewed=False, ridge=1e-3,
                          lengths=DEFAULT_LENGTHS, n_queries=24, seed=0,
                          len_ratio=None, comp_bias_window=0):
    """Fit the uncorrected cluster-max model.

    ``comp_bias_window`` is the composition-bias correction half-width the
    scores use (0 = none); search must use the same, so it is recorded in
    the parameters and in their scoring-context fingerprint.

    Returns ``(params, query_fits, missing)``. When ``missing`` (leaders at
    ``E0 <= audit_ceiling`` without an adjudication) is non-empty, ``params``
    is None. ``related`` is read at call time, so adding to it and calling
    again refits with the new exclusions.
    """
    target_fingerprint = database_fingerprint(entries, clusters, n_codes)
    family = np.asarray([labels.get(chain_id, {}).get("family", "")
                         for chain_id in chain_ids], dtype=object)
    clan = np.asarray([labels.get(chain_id, {}).get("clan", "")
                       for chain_id in chain_ids], dtype=object)

    def exclude(first, second):
        pair = (chain_ids[first], chain_ids[second])
        return (known_equal(family[first], family[second])
                or known_equal(clan[first], clan[second])
                or any(alias in related for alias in _aliases(*pair)))

    def is_audited(first, second):
        pair = (chain_ids[first], chain_ids[second])
        return any(alias in audited for alias in _aliases(*pair))

    if model != "query-conditional" and int(comp_bias_window):
        raise CalibrationInputError(
            "--comp-bias-window is only available with the query-conditional model")
    query_fits = []
    if model == "query-conditional":
        if len_ratio is not None:
            raise CalibrationInputError(
                "--len-ratio is only available with --model length-grid; "
                "the deployed query-conditional model uses the full database")
        index_of = {}
        for index, chain_id in enumerate(chain_ids):
            for alias in (chain_id, short_id(chain_id, chain_id)):
                if alias in index_of and index_of[alias] != index:
                    raise ValueError(f"ambiguous database chain id: {alias}")
                index_of[alias] = index
        if query_panel is None:
            query_indices = list(range(len(chain_ids)))
        else:
            absent = [query_id for query_id in query_panel if query_id not in index_of]
            if absent:
                raise ValueError(f"query-panel ids absent from database: {absent[:3]}")
            query_indices = [index_of[query_id] for query_id in query_panel]
        fitted = EV.calibrate_clustermax_query_conditional(
            matrix, tokens, clusters, query_indices,
            gap_open=GAP_OPEN, gap_extend=GAP_EXTEND, exclude=exclude, ridge=ridge,
            is_audited=None if allow_unreviewed else is_audited,
            audit_evalue_ceiling=None if allow_unreviewed else audit_ceiling,
            comp_bias_window=comp_bias_window,
        )
        missing = [{
            "query_id": chain_ids[row["query_index"]],
            "target_id": chain_ids[row["target_index"]],
            "cluster_id": row["cluster"],
            "raw_score": row["raw_score"],
            "raw_evalue": row["raw_evalue"],
        } for row in fitted.pop("missing_audit", [])]
        if missing:
            return None, [], missing
        query_fits = fitted.pop("query_fits")
        for row in query_fits:
            row["query_id"] = chain_ids[row.pop("query_index")]
    else:
        fitted = EV.calibrate_clustermax_length_dependent(
            matrix, tokens, clusters, gap_open=GAP_OPEN, gap_extend=GAP_EXTEND,
            lengths=tuple(lengths), n_queries=n_queries, seed=seed,
            exclude=exclude, len_ratio=len_ratio,
        )
    params = {
        "scheme": ("cluster-max-query-conditional"
                   if model == "query-conditional" else "cluster-max"),
        "gap_open": GAP_OPEN,
        "gap_extend": GAP_EXTEND,
        "seed": seed,
        "len_ratio": len_ratio,
        "comp_bias_window": int(comp_bias_window),
        "calibration_database": str(Path(db_path).resolve()),
        "n_clusters": fitted["n_clusters"],
        "n_chains": fitted["n_chains"],
        "target_database_fingerprint": target_fingerprint,
        "calibration_context_fingerprint": calibration_context_fingerprint(
            target_fingerprint, matrix, GAP_OPEN, GAP_EXTEND, comp_bias_window),
        "search_scope": ("full_database" if len_ratio is None
                         else f"length_ratio_{len_ratio:g}"),
        "clusters": "stored cluster labels in the RFoldseek database",
        "null": "cluster maxima over real database chains; same annotated "
                "family or clan and adjudicated structural relatives excluded; "
                "cluster leaders are selected after exclusions",
    }
    if model == "query-conditional":
        params["query_parameter_model"] = fitted["query_parameter_model"]
        params["n_calibration_queries"] = len(query_fits)
        params["validation_status"] = (
            "provisional-incomplete-structural-audit" if allow_unreviewed
            else "background-audit-complete")
        if not allow_unreviewed:
            params["structural_audit_ceiling"] = audit_ceiling
    else:
        params["skipped_lengths"] = fitted["skipped_lengths"]
        params["length_table"] = fitted["length_table"]
    return params, query_fits, []


# ------------------------------------------- stage 2: calibration background

BACKGROUND_FIELDS = ("query_id", "family", "clan", "cluster_id", "target_id",
                     "raw_score", "raw_evalue", "structurally_related",
                     "n_database_clusters",
                     "background_clusters_before_structural_exclusion")
QUERY_METADATA_FIELDS = ("query_id", "family", "clan", "background_scope",
                         "structural_audit_ceiling", "structural_audit_complete",
                         "n_database_clusters",
                         "background_clusters_before_structural_exclusion",
                         "n_retained_background_clusters")
MISSING_FIELDS = ("query_id", "target_id", "cluster_id", "raw_score", "raw_evalue")


def build_background(params, target_entries, targets, target_ids, n_codes,
                     clusters, queries, query_ids, labels, matrix, audited,
                     related, *, audit_ceiling=AUDIT_CEILING,
                     background_scope="all", log=print):
    """One cluster-max observation per calibration query and target cluster.

    Returns ``(rows, query_metadata, missing)``; ``rows`` and
    ``query_metadata`` are empty when ``missing`` is not.
    """
    n_clusters = len(set(clusters))
    target_fingerprint = database_fingerprint(target_entries, clusters, n_codes)
    fitted_target = params.get("target_database_fingerprint")
    if fitted_target is not None and fitted_target != target_fingerprint:
        raise ValueError("parameter file belongs to a different target database")
    fitted_context = params.get("calibration_context_fingerprint")
    window = int(params.get("comp_bias_window", 0))
    actual_context = calibration_context_fingerprint(
        target_fingerprint, matrix, GAP_OPEN, GAP_EXTEND, window)
    if fitted_context is not None and fitted_context != actual_context:
        raise ValueError("parameter file belongs to a different scoring context")
    background = background_frequencies(targets, n_codes) if window else None
    target_index = {chain_id: index for index, chain_id in enumerate(target_ids)}
    members = {}
    for index, cluster in enumerate(clusters):
        members.setdefault(str(cluster), []).append(index)
    blank = {"family": "", "clan": ""}

    rows, metadata, missing = [], [], []
    for query_number, (query_id, query) in enumerate(zip(query_ids, queries), 1):
        query_label = labels.get(query_id, blank)
        self_cluster = None
        if query_id in target_index:
            self_cluster = str(clusters[target_index[query_id]])
        mu, beta = EV.clustermax_params_for_query(params, query)
        qpos, qprof = query_profile(query, matrix, background, window)
        query_rows = []
        before_exclusion = 0
        for cluster, indices in members.items():
            if cluster == self_cluster:
                continue
            scored = []
            for index in indices:
                target_id, target = target_ids[index], targets[index]
                target_label = labels.get(target_id, blank)
                if (background_scope == "annotated"
                        and target_label["family"] in UNKNOWN):
                    continue
                if known_equal(query_label["family"], target_label["family"]):
                    continue
                if known_equal(query_label["clan"], target_label["clan"]):
                    continue
                score = float(smith_waterman_numba(
                    qpos, target, qprof, GAP_OPEN, GAP_EXTEND))
                scored.append((score, target_id))
            if not scored:
                continue
            before_exclusion += 1
            best = None
            for score, target_id in sorted(scored, reverse=True):
                aliases = _aliases(query_id, target_id)
                if any(pair in related for pair in aliases):
                    continue
                raw_evalue = float(EV.evalue_clustermax(score, mu, beta, n_clusters))
                if (raw_evalue <= audit_ceiling
                        and not any(pair in audited for pair in aliases)):
                    missing.append({"query_id": query_id, "target_id": target_id,
                                    "cluster_id": cluster, "raw_score": score,
                                    "raw_evalue": raw_evalue})
                    break
                best = (score, target_id, raw_evalue)
                break
            if best is None:
                continue
            score, target_id, raw_evalue = best
            query_rows.append({
                "query_id": query_id,
                "family": query_label["family"],
                "clan": query_label["clan"],
                "cluster_id": cluster,
                "target_id": target_id,
                "raw_score": score,
                "raw_evalue": raw_evalue,
                "structurally_related": "false",
                "n_database_clusters": n_clusters,
                "background_clusters_before_structural_exclusion": None,
            })
        for row in query_rows:
            row["background_clusters_before_structural_exclusion"] = before_exclusion
        rows.extend(query_rows)
        metadata.append({
            "query_id": query_id,
            "family": query_label["family"],
            "clan": query_label["clan"],
            "background_scope": background_scope,
            "structural_audit_ceiling": audit_ceiling,
            "structural_audit_complete": "true",
            "n_database_clusters": n_clusters,
            "background_clusters_before_structural_exclusion": before_exclusion,
            "n_retained_background_clusters": len(query_rows),
        })
        if log and query_number % 10 == 0:
            log(f"processed {query_number}/{len(queries)} queries")
    if missing:
        return [], [], missing
    return rows, metadata, []


# ------------------------------------------------------ stage 3: power law

def read_power_panel(path):
    """Query-panel TSV (optionally with stage-2 metadata) -> {query_id: metadata}."""
    panel = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            query_id, family = row["query_id"].strip(), row["family"].strip()
            if not query_id or not family:
                raise ValueError("query panel needs nonempty query_id and family")
            if query_id in panel:
                raise ValueError(f"duplicate calibration query: {query_id}")
            bq = (row.get("background_clusters_before_structural_exclusion") or "").strip()
            panel_n = (row.get("n_database_clusters") or "").strip()
            scope = (row.get("background_scope") or "").strip()
            audit_complete = (row.get("structural_audit_complete") or "").strip()
            audit_ceiling = (row.get("structural_audit_ceiling") or "").strip()
            panel[query_id] = {
                "family": family,
                "bq": int(bq) if bq else None,
                "n_clusters": int(panel_n) if panel_n else None,
                "background_scope": scope or None,
                "audit_complete": truthy(audit_complete) if audit_complete else None,
                "audit_ceiling": float(audit_ceiling) if audit_ceiling else None,
            }
    if not panel:
        raise ValueError("query panel is empty")
    return panel


def read_background(path, raw_evalue_ceiling=None):
    """Background TSV rows, minus related rows and rows above the ceiling."""
    rows = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if truthy(row.get("structurally_related", "")):
                continue
            if (raw_evalue_ceiling is not None
                    and float(row["raw_evalue"]) > raw_evalue_ceiling):
                continue
            rows.append(row)
    return rows


def fit_power_params(params, rows, n_clusters, *, panel=None,
                     raw_evalue_ceiling=None, background_scaling="none",
                     length_bins=1, query_lengths=None):
    """Return a copy of ``params`` carrying the fitted power calibration.

    With ``length_bins`` > 1 the calibration queries are split into that many
    equal-count bins by length and a power law is fitted in each; search
    interpolates ``log a`` and ``b`` between bin centres in log query length.
    One law for all lengths can leave short queries far too conservative and
    long ones optimistic. ``query_lengths`` maps query id -> token count.
    """
    fitted_n = params.get("n_clusters")
    if fitted_n is not None and int(fitted_n) != n_clusters:
        raise ValueError(
            f"parametric E-value model has {fitted_n} clusters, "
            f"but the target database has {n_clusters}")
    if background_scaling == "annotated-to-full" and panel is None:
        raise CalibrationInputError(
            "annotated-to-full scaling requires --query-panel with B_q metadata")
    if (background_scaling == "annotated-to-full"
            and any(metadata["background_scope"] != "annotated"
                    for metadata in panel.values())):
        raise CalibrationInputError(
            "annotated-to-full scaling requires query metadata produced "
            "with --background-scope annotated")
    calibration = _fit_power_once(rows, n_clusters, panel, raw_evalue_ceiling,
                                  background_scaling)
    if int(length_bins) > 1:
        if panel is None or query_lengths is None:
            raise CalibrationInputError(
                "length-binned calibration needs the query panel and query lengths")
        ordered = sorted(panel, key=lambda query_id: query_lengths[query_id])
        k = int(length_bins)
        groups = [ordered[i * len(ordered) // k:(i + 1) * len(ordered) // k]
                  for i in range(k)]
        bins = []
        while True:
            # A bin with too few low-E0 observations cannot be fitted; merge it
            # into a neighbour rather than abort the whole calibration.
            try:
                fits = [_fit_power_once(
                    [row for row in rows if row["query_id"] in set(members)],
                    n_clusters, {q: panel[q] for q in members},
                    raw_evalue_ceiling, background_scaling) for members in groups]
                break
            except ValueError:
                if len(groups) == 1:
                    raise
                sizes = [sum(1 for row in rows if row["query_id"] in set(g)) for g in groups]
                i = int(np.argmin(sizes))
                j = i + 1 if i + 1 < len(groups) else i - 1
                lo, hi = sorted((i, j))
                groups[lo:hi + 1] = [groups[lo] + groups[hi]]
        for members, sub in zip(groups, fits):
            lengths = [query_lengths[q] for q in members]
            bins.append({
                "log_length_centre": float(np.mean(np.log(lengths))),
                "length_min": int(min(lengths)), "length_max": int(max(lengths)),
                "a": sub["a"], "b": sub["b"],
                "n_calibration_queries": len(members),
                "n_observations": sub["n_observations"],
                "fit_raw_evalue_min": sub["fit_raw_evalue_min"],
            })
        calibration["scheme"] = "power-law-by-length"
        calibration["length_bins"] = bins
        calibration["note"] = ("a and b at the top level are the pooled fit; search "
                               "uses length_bins")
    params = dict(params)
    params["scheme"] = "cluster-max-query-conditional-power"
    params["power_calibration"] = calibration
    return params


def _fit_power_once(rows, n_clusters, panel, raw_evalue_ceiling, background_scaling):
    if not rows:
        raise ValueError("no background rows remain after filtering")

    values = [float(row["raw_evalue"]) for row in rows]
    explicit = [(row.get("weight") or "").strip() for row in rows]
    if all(explicit):
        if panel is not None or background_scaling != "none":
            raise CalibrationInputError(
                "explicit weights cannot be combined with --query-panel "
                "or --background-scaling")
        weights = [float(value) for value in explicit]
        weighting = "explicit"
    elif any(explicit):
        raise ValueError("weight must be present on every row or omitted from every row")
    else:
        if any(not (row.get("query_id") or "").strip() for row in rows):
            raise ValueError("background TSV needs weight or query_id on every row")
        queries = sorted(panel) if panel is not None else sorted(
            {row["query_id"] for row in rows})
        if panel is not None:
            unknown = {row["query_id"] for row in rows} - set(panel)
            if unknown:
                raise ValueError(
                    f"background queries missing from panel: {sorted(unknown)[:3]}")
        have_family = panel is not None or all(
            (row.get("family") or "").strip() for row in rows)
        if have_family:
            family_queries = defaultdict(set)
            if panel is not None:
                for query_id, metadata in panel.items():
                    family_queries[metadata["family"]].add(query_id)
                for row in rows:
                    expected_family = panel[row["query_id"]]["family"]
                    if (row.get("family") or "").strip() not in ("", expected_family):
                        raise ValueError(f"family mismatch for {row['query_id']}")
            else:
                for row in rows:
                    family_queries[row["family"]].add(row["query_id"])
            weights = []
            for row in rows:
                family = (panel[row["query_id"]]["family"]
                          if panel is not None else row["family"])
                weight = 1.0 / (len(family_queries) * len(family_queries[family]))
                if background_scaling == "annotated-to-full":
                    metadata = panel[row["query_id"]]
                    if metadata["bq"] is None or metadata["bq"] <= 0:
                        raise ValueError(
                            f"missing or invalid B_q for {row['query_id']}")
                    if (metadata["n_clusters"] is not None
                            and metadata["n_clusters"] != n_clusters):
                        raise ValueError(
                            f"database cluster count mismatch for {row['query_id']}")
                    weight *= n_clusters / metadata["bq"]
                weights.append(weight)
            weighting = "equal-family, equal-query-within-family"
            if background_scaling == "annotated-to-full":
                weighting += ", multiplied by N_D/B_q"
        else:
            weights = [1.0 / len(queries)] * len(rows)
            weighting = "equal-query"
    fit_maximum = raw_evalue_ceiling if raw_evalue_ceiling is not None else max(values)
    calibration = EV.fit_power_calibration(values, weights, n_clusters,
                                           maximum=fit_maximum)
    calibration["weighting"] = weighting
    if panel is not None:
        calibration["n_calibration_queries"] = len(panel)
        calibration["n_families"] = len({
            metadata["family"] for metadata in panel.values()})
    calibration["background_scaling"] = background_scaling
    audit_metadata = ([metadata["audit_complete"] for metadata in panel.values()]
                      if panel is not None else [])
    if audit_metadata and all(value is True for value in audit_metadata):
        calibration["validation_status"] = "background-audit-complete"
        ceilings = {metadata["audit_ceiling"] for metadata in panel.values()}
        if len(ceilings) != 1 or None in ceilings:
            raise ValueError("query metadata has inconsistent structural audit ceilings")
        audit_ceiling = ceilings.pop()
        if raw_evalue_ceiling is None or raw_evalue_ceiling > audit_ceiling:
            raise ValueError(
                "power fit ceiling must not exceed the structural audit ceiling")
        calibration["structural_audit_ceiling"] = audit_ceiling
    else:
        calibration["validation_status"] = "structural-audit-not-recorded"
    if raw_evalue_ceiling is not None:
        calibration["fit_raw_evalue_ceiling"] = raw_evalue_ceiling
    return calibration


# --------------------------------------------------------- US-align audit

AUDIT_FIELDS = ("query_id", "target_id", "tm_query", "tm_target")


def _pool_for(query_id, target_list, query_entries, target_entries):
    query = query_entries[query_id]
    if query.get("coords") is None or query.get("mask") is None:
        raise ValueError(
            f"{query_id} has no coordinates to align; rebuild the "
            "database from structures or pass --related_pairs")
    pool = []
    for target_id in dict.fromkeys(target_list):
        target = target_entries[target_id]
        if target.get("coords") is None or target.get("mask") is None:
            raise ValueError(
                f"{target_id} has no coordinates to align; rebuild the "
                "database from structures or pass --related_pairs")
        pool.append((target_id, target["coords"], target["mask"]))
    return query, pool


def audit_pairs(binary, pairs, query_entries, target_entries, path, audited,
                related, tm_threshold=TM_THRESHOLD, workers=None):
    """Align each (query_id, target_id) with US-align and record the result.

    Results are appended to ``path`` and added to ``audited``/``related``. A
    chain with too few resolved residues to align scores zero, i.e. it is
    counted as unrelated. Queries are aligned in ``workers`` concurrent
    US-align processes (default: one per CPU).
    """
    by_query = defaultdict(list)
    for query_id, target_id in pairs:
        by_query[query_id].append(target_id)
    jobs = [(query_id,) + _pool_for(query_id, target_list, query_entries,
                                    target_entries)
            for query_id, target_list in by_query.items()]

    def align(job):
        query_id, query, pool = job
        try:
            scores, _ = score_pool(binary, query["coords"], query["mask"], pool)
        except ValueError:
            scores = {}              # query too short to superpose
        return query_id, pool, scores

    path = Path(path)
    new_file = not path.exists()
    with path.open("a", newline="") as handle, ThreadPoolExecutor(
            workers or os.cpu_count() or 1) as executor:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS, delimiter="\t")
        if new_file:
            writer.writeheader()
        for query_id, pool, scores in executor.map(align, jobs):
            for target_id, _, _ in pool:
                tm_query, tm_target = scores.get(target_id, (0.0, 0.0))
                writer.writerow({"query_id": query_id, "target_id": target_id,
                                 "tm_query": tm_query, "tm_target": tm_target})
                add_adjudication(audited, related, query_id, target_id,
                                 0.5 * (tm_query + tm_target) >= tm_threshold)
            handle.flush()


# -------------------------------------------------------------- pipeline

def calibrate(db, clusters, labels, queries, output, *, db_type="predicted",
              related_pairs=None, query_panel=None, usalign=None, workdir=None,
              index_prefix=None, matrix=None, audit_ceiling=AUDIT_CEILING,
              tm_threshold=TM_THRESHOLD, max_rounds=20, comp_bias_window=0,
              length_bins=1, log=print):
    """Calibrate E-values for ``db`` and write the calibrated database to ``output``.

    ``db`` is an encoded database, ``clusters`` its CD-HIT-EST (80%) cluster
    file, ``labels`` a TSV of Rfam family/clan per chain, and ``queries`` an
    independent database of calibration queries. ``db_type`` selects the
    background: every eligible cluster for predicted structures, annotated
    targets reweighted to the full database for experimental ones.
    ``comp_bias_window`` > 0 calibrates composition-bias-corrected scores;
    the attached parameters then switch search to the same correction.
    ``length_bins`` > 1 fits the power law separately per query-length bin.
    """
    if db_type not in DATABASE_TYPES:
        raise ValueError(f"db_type must be one of {sorted(DATABASE_TYPES)}")
    if int(length_bins) < 1:
        raise ValueError("length_bins must be at least 1")
    background_scope, background_scaling = DATABASE_TYPES[db_type]
    if Path(db).resolve() == Path(output).resolve():
        raise ValueError("output must differ from the input database")
    if Path(output).exists():
        raise FileExistsError(f"output database already exists: {output}")
    work = Path(workdir) if workdir else Path(output).with_name(
        Path(output).stem + ".calibration")
    work.mkdir(parents=True, exist_ok=True)
    log(f"calibrating {db} ({db_type}); intermediate files in {work}")

    _, entries, tokens, chain_ids, n_codes, _ = load_database(db)
    cluster_labels = strict_cluster_labels(entries, read_clusters(clusters),
                                           index_prefix)
    _, query_entries, query_tokens, query_ids, _, _ = load_database(queries)
    if (int(length_bins) > 1
            and len(query_ids) < MIN_QUERIES_PER_LENGTH_BIN * int(length_bins)):
        raise CalibrationInputError(
            f"{len(query_ids)} calibration queries are too few for {length_bins} "
            f"length bins; use at least {MIN_QUERIES_PER_LENGTH_BIN} per bin")
    sub = load_matrix(matrix)
    chain_labels = read_labels(labels)
    panel = read_query_panel(query_panel)

    audit_file = work / "usalign_audit.tsv"
    audited, related = read_adjudication(related_pairs, tm_threshold)
    if audit_file.exists():
        read_adjudication(audit_file, tm_threshold, audited, related)
    target_by_id = dict(zip(chain_ids, entries))
    cluster_of = dict(zip(chain_ids, cluster_labels))
    cluster_members = defaultdict(list)
    for chain_id, label in cluster_of.items():
        cluster_members[label].append(chain_id)
    query_by_id = dict(zip(query_ids, query_entries))
    binary = []

    def resolve_missing(stage, missing, query_source):
        """Audit ``missing`` with US-align, or stop with the list written out."""
        if not binary:
            try:
                binary.append(find_usalign(usalign))
            except USalignNotFound as error:
                path = work / f"{stage}.missing-audit.tsv"
                _write_tsv(path, MISSING_FIELDS, missing)
                raise AuditIncomplete(
                    f"{len(missing)} cluster leaders at E0<={audit_ceiling:g} lack "
                    f"US-align adjudication (listed in {path}). {error} "
                    "Alternatively pass --related_pairs with their TM-scores."
                ) from None
        # Excluding a related leader promotes the next member of its cluster,
        # which would need its own round; align the whole cluster at once.
        pairs = []
        for row in missing:
            query_id = row["query_id"]
            query_label = chain_labels.get(query_id, {})
            for target_id in cluster_members[cluster_of[row["target_id"]]]:
                target_label = chain_labels.get(target_id, {})
                if (target_id == query_id
                        or known_equal(query_label.get("family", ""),
                                       target_label.get("family", ""))
                        or known_equal(query_label.get("clan", ""),
                                       target_label.get("clan", ""))
                        or any(pair in audited
                               for pair in _aliases(query_id, target_id))):
                    continue
                pairs.append((query_id, target_id))
        pairs = list(dict.fromkeys(pairs))
        log(f"  {stage}: aligning {len(missing)} unreviewed cluster leaders "
            f"and their cluster members ({len(pairs)} pairs) with US-align")
        audit_pairs(binary[0], pairs, query_source, target_by_id, audit_file,
                    audited, related, tm_threshold)

    log("[1/4] fitting the query-conditional Gumbel model")
    for _ in range(max_rounds):
        params, query_fits, missing = fit_clustermax_params(
            db, entries, tokens, chain_ids, n_codes, cluster_labels,
            chain_labels, sub, audited, related, query_panel=panel,
            audit_ceiling=audit_ceiling, comp_bias_window=comp_bias_window)
        if not missing:
            break
        resolve_missing("gumbel", missing, target_by_id)
    else:
        raise AuditIncomplete(f"structural audit did not settle in {max_rounds} rounds")
    params_path = work / "clustermax_params.json"
    params_path.write_text(json.dumps(params, indent=2) + "\n")
    _write_tsv(work / "query_fits.tsv", list(query_fits[0]), query_fits)

    log("[2/4] scoring calibration queries against every target cluster")
    for _ in range(max_rounds):
        rows, metadata, missing = build_background(
            params, entries, tokens, chain_ids, n_codes, cluster_labels,
            query_tokens, query_ids, chain_labels, sub, audited, related,
            audit_ceiling=audit_ceiling, background_scope=background_scope,
            log=log)
        if not missing:
            break
        resolve_missing("background", missing, query_by_id)
    else:
        raise AuditIncomplete(f"structural audit did not settle in {max_rounds} rounds")
    background_path = work / "background.tsv"
    metadata_path = work / "background.queries.tsv"
    _write_tsv(background_path, BACKGROUND_FIELDS, rows)
    _write_tsv(metadata_path, QUERY_METADATA_FIELDS, metadata)

    log("[3/4] fitting the power-law calibration")
    fitted = fit_power_params(
        params, read_background(background_path, audit_ceiling),
        len(set(cluster_labels)), panel=read_power_panel(metadata_path),
        raw_evalue_ceiling=audit_ceiling, background_scaling=background_scaling,
        length_bins=length_bins,
        query_lengths={q: len(t) for q, t in zip(query_ids, query_tokens)})
    evalue_path = work / "evalue_params.json"
    evalue_path.write_text(json.dumps(fitted, indent=2) + "\n")

    log("[4/4] attaching the parameters to the database")
    result = attach_evalue_calibration(db, clusters, str(evalue_path), output,
                                       index_prefix=index_prefix)
    power = fitted["power_calibration"]
    log(f"wrote {output} ({result['n_chains']} chains, {result['n_clusters']} "
        f"clusters; a_D={power['a']:.3g}, b_D={power['b']:.3g})")
    return result
