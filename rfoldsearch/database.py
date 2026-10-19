"""Build and load searchable token databases (.pt)."""
from __future__ import annotations
import hashlib
import json
import math
import os

import numpy as np
import torch

from . import experimental_evalue as _experimental_evalue
from . import predicted_evalue as _predicted_evalue

from .backbone import extract_backbone_coords
from .edges import find_base_pair_and_stack_idx
from .prefilter import KMER_K
from .tokenizer import encode_entry, load_model


def _validate_power_calibration(calibration):
    if calibration is None:
        return
    a, b = float(calibration["a"]), float(calibration["b"])
    if not math.isfinite(a) or not math.isfinite(b) or a <= 0 or b <= 0:
        raise ValueError("power E-value calibration requires positive finite a and b")


def database_fingerprint(entries, clusters, n_codes: int = 80) -> str:
    """Stable identity for the logical token database and its clustering.

    Local paths and arbitrary cluster labels are excluded. Cluster membership
    is represented by the sorted identities of its members, so rebuilding the
    same content in another directory produces the same fingerprint.
    """
    if len(entries) != len(clusters):
        raise ValueError("clusters must give one label per database entry")
    identities = []
    for index, entry in enumerate(entries):
        tokens = np.asarray(entry["tokens"], dtype="<i4").reshape(-1)
        token_hash = hashlib.sha256(tokens.tobytes()).hexdigest()
        identities.append((short_id(entry.get("id"), index), token_hash, len(tokens)))
    members = {}
    for identity, cluster in zip(identities, clusters):
        members.setdefault(str(cluster), []).append(identity)
    cluster_signatures = {}
    for cluster, group in members.items():
        digest = hashlib.sha256()
        for identity in sorted(group):
            digest.update(json.dumps(identity, separators=(",", ":")).encode())
            digest.update(b"\0")
        cluster_signatures[cluster] = digest.hexdigest()
    rows = [identity + (cluster_signatures[str(cluster)],)
            for identity, cluster in zip(identities, clusters)]
    # The salt keeps the tool's former name so that existing calibrations still match.
    digest = hashlib.sha256(b"rfoldseek-logical-token-db-v1\0")
    digest.update(str(int(n_codes)).encode())
    digest.update(b"\0")
    for row in sorted(rows):
        digest.update(json.dumps(row, separators=(",", ":")).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def calibration_context_fingerprint(database_digest: str, matrix,
                                    gap_open: float, gap_extend: float) -> str:
    """Bind E-value parameters to the target, score matrix and gap costs."""
    substitution = np.asarray(matrix, dtype="<f4")
    digest = hashlib.sha256(b"rfoldseek-evalue-context-v1\0")
    digest.update(str(database_digest).encode())
    digest.update(b"\0")
    digest.update(substitution.tobytes())
    digest.update(np.asarray([gap_open, gap_extend], dtype="<f8").tobytes())
    return digest.hexdigest()


def structure_to_entry(path: str, max_len: int | None = None) -> dict:
    """One PDB/mmCIF file -> the JSON entry the tokenizer consumes."""
    edge_idx = find_base_pair_and_stack_idx(path)
    coords, mask, seq = extract_backbone_coords(path, chain_id=None, max_len=max_len)
    return {
        "id": path,
        "coords": coords.tolist(),
        "mask": mask.tolist(),
        "seq": seq,
        "pair_idx": edge_idx["pair_idx"].tolist(),
        "pair_orient": edge_idx["pair_orient"].tolist(),
        "stack_idx": edge_idx["stack_idx"].tolist(),
    }


def read_clusters(path: str) -> dict:
    """Read a chain -> cluster mapping from a CD-HIT .clstr file or a TSV.

    A .clstr file is what ``cd-hit-est -c 0.8 -n 5 -aS 0.8 -aL 0.8 -g 1``
    writes; the TSV form is two columns, chain id and cluster label. Keys are
    matched against both the raw entry id and its ``short_id``, so a mapping
    built from a FASTA of basenames still lines up with a database whose ids
    are full paths.
    """
    out: dict = {}
    with open(path) as f:
        head = f.readline()
        f.seek(0)
        if head.startswith(">Cluster"):
            cid = -1
            for line in f:
                if line.startswith(">Cluster"):
                    cid = int(line.split()[1])
                elif ">" in line:
                    out[line.split(">")[1].split("...")[0]] = cid
        else:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2 and parts[0] not in ("id", "chain_id", "target_id"):
                    out[parts[0]] = parts[1]
    if not out:
        raise ValueError(f"no clusters read from {path}")
    return out


def assign_clusters(ents, mapping) -> list:
    """One cluster label per entry; unmapped chains become singleton clusters.

    An unmapped chain is its own cluster rather than being dropped or pooled:
    that is the conservative reading, since counting it as its own question of
    the database can only make its E-value larger.
    """
    labels, spare = [], 0
    for i, e in enumerate(ents):
        raw = str(e.get("id", ""))
        lab = mapping.get(raw, mapping.get(short_id(raw, i)))
        if lab is None:
            lab = f"_singleton_{spare}"
            spare += 1
        labels.append(lab)
    return labels


def build_database(checkpoint: str, data_paths: list[str], output: str,
                   min_len: int | None = None, device: str = "cpu",
                   progress: bool = True, clusters: str | None = None,
                   evalue_params: str | None = None) -> dict:
    """Encode every entry in the given JSONL files into one .pt database.

    ``min_len`` drops chains with fewer than that many valid residues.
    The default is ``None``, which keeps every chain the tokeniser can
    encode. The paper's 885-chain corpus used ``min_len=20``; pass it
    explicitly to reproduce that set.

    Chains shorter than the prefilter's k-mer length contribute no k-mers
    to the inverted index, so they can never be returned as a hit and
    cannot be used as a query. They are still encoded and stored when no
    ``min_len`` is given — the count is reported so the choice is visible
    — but note that each one adds a cluster to the E-value search space
    while being unreachable, which makes every E-value slightly larger.

    ``clusters`` is the CD-HIT .clstr (or TSV) that groups near-duplicate
    chains. E-values are counted in clusters, so the mapping is stored with
    the database; without it every chain is its own cluster and the E-value
    over-counts a redundant database.
    """
    dev = torch.device(device)
    model, ckpt, n_codes, _ = load_model(checkpoint, dev)
    norm_mean = ckpt.get("norm_mean", None)
    norm_std = ckpt.get("norm_std", None)
    active_blocks = ckpt.get("active_blocks", None)

    entries, seen = [], set()
    for data_path in data_paths:
        with open(data_path) as f:
            for line in f:
                e = json.loads(line)
                eid = str(e.get("id", ""))
                if eid in seen:
                    continue
                seen.add(eid)
                if min_len is not None and sum(e["mask"]) < min_len:
                    continue
                enc = encode_entry(model, e, dev, norm_mean, norm_std,
                                   active_blocks=active_blocks)
                if enc is not None:
                    entries.append(enc)
                if progress and len(entries) % 100 == 0 and entries:
                    print(f"  encoded {len(entries)}", flush=True)

    labels = (assign_clusters(entries, read_clusters(clusters)) if clusters
              else [f"_singleton_{i}" for i in range(len(entries))])
    for e, lab in zip(entries, labels):
        e["cluster"] = lab
    n_clusters = len(set(labels))
    target_fingerprint = database_fingerprint(entries, labels, int(n_codes))
    embedded_evalue_params = None
    if evalue_params is not None:
        with open(evalue_params) as handle:
            embedded_evalue_params = json.load(handle)
        query_model = embedded_evalue_params.get("query_parameter_model")
        if query_model is not None and "training_domain" not in query_model:
            raise ValueError("query-conditional E-value model needs a training_domain")
        fitted_clusters = embedded_evalue_params.get("n_clusters")
        if fitted_clusters is not None and int(fitted_clusters) != n_clusters:
            raise ValueError(
                f"E-value parameters have {int(fitted_clusters)} clusters, "
                f"but the database has {n_clusters}"
            )
        fitted_target = embedded_evalue_params.get("target_database_fingerprint")
        if fitted_target is not None and fitted_target != target_fingerprint:
            raise ValueError("E-value parameters were fitted for a different target database")
        if embedded_evalue_params.get("scheme") == _experimental_evalue.SCHEME:
            _experimental_evalue.validate_parameters(
                embedded_evalue_params, n_clusters, target_fingerprint,
                embedded_evalue_params.get("calibration_context_fingerprint"))
        if embedded_evalue_params.get("scheme") == _predicted_evalue.SCHEME:
            _predicted_evalue.validate_parameters(
                embedded_evalue_params, n_clusters, target_fingerprint,
                embedded_evalue_params.get("calibration_context_fingerprint"),
                [entry["tokens"] for entry in entries])
        power = embedded_evalue_params.get("power_calibration")
        if power is not None:
            _validate_power_calibration(power)
            fitted_n = power.get("n_clusters", embedded_evalue_params.get("n_clusters"))
            if fitted_n is None:
                raise ValueError("power E-value calibration needs a cluster count")
            fitted_n = int(fitted_n)
            if fitted_n != n_clusters:
                raise ValueError(
                    f"power E-value calibration has {fitted_n} clusters, "
                    f"but the database has {n_clusters}"
                )
    db = {"n_codes": int(n_codes), "entries": entries, "checkpoint": checkpoint,
          "n_clusters": n_clusters, "clustered": bool(clusters),
          "database_fingerprint": target_fingerprint}
    if embedded_evalue_params is not None:
        db["evalue_params"] = embedded_evalue_params
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    torch.save(db, output)
    lens = np.array([len(e["tokens"]) for e in entries]) if entries else np.zeros(1)
    print(f"Saved: {output}\n  Entries: {len(entries)}  "
          f"(length: mean={lens.mean():.0f}, median={np.median(lens):.0f}, "
          f"min={lens.min():.0f}, max={lens.max():.0f})"
          f"{'' if min_len is None else f'  [min_len={min_len}]'}"
          f"\n  Clusters: {len(set(labels))}"
          f"{'' if clusters else ' (unclustered: every chain its own cluster)'}")
    n_unindexable = int((lens < KMER_K).sum())
    if n_unindexable:
        print(f"  Note: {n_unindexable} chain(s) shorter than the {KMER_K}-mer "
              f"prefilter length. They contribute no k-mers, so they can never "
              f"be returned as a hit or used as a query, yet each still counts "
              f"as a cluster in the E-value search space. Pass "
              f"--min_len {KMER_K} to drop them.")
    return db


def strict_cluster_labels(entries, mapping, index_prefix: str | None = None) -> list:
    """One cluster label per entry; an unmapped entry is an error.

    Unlike ``assign_clusters``, calibration must cover the whole target
    database, so nothing becomes a singleton silently. ``index_prefix``
    selects positional CD-HIT names such as ``t0``, ``t1``, ...
    """
    labels = []
    for index, entry in enumerate(entries):
        raw = str(entry.get("id", ""))
        key = f"{index_prefix}{index}" if index_prefix is not None else raw
        if key not in mapping and index_prefix is None:
            key = short_id(raw, index)
        if key not in mapping:
            raise ValueError(f"no cluster assignment for entry {index}: {raw}")
        labels.append(mapping[key])
    return labels


def attach_evalue_calibration(database: str, clusters: str,
                              evalue_params: str, output: str,
                              index_prefix: str | None = None) -> dict:
    """Add cluster labels and matching E-value parameters to an encoded DB.

    ``index_prefix`` explicitly selects CD-HIT names such as ``t0``, ``t1``,
    ... when the FASTA used positional IDs instead of structure IDs. Missing
    assignments are errors: calibration must cover the whole target database.
    The input database is never overwritten.
    """
    if os.path.abspath(database) == os.path.abspath(output):
        raise ValueError("output must differ from the input database")
    if os.path.exists(output):
        raise FileExistsError(f"output database already exists: {output}")
    db = torch.load(database, map_location="cpu", weights_only=False)
    entries = db["entries"]
    labels = strict_cluster_labels(entries, read_clusters(clusters), index_prefix)
    n_clusters = len(set(labels))
    with open(evalue_params) as handle:
        params = json.load(handle)
    query_model = params.get("query_parameter_model")
    if query_model is not None and "training_domain" not in query_model:
        raise ValueError("query-conditional E-value model needs a training_domain")
    fitted_n = params.get("n_clusters")
    power = params.get("power_calibration")
    if power is not None:
        _validate_power_calibration(power)
        fitted_n = power.get("n_clusters", fitted_n)
    if fitted_n is None or int(fitted_n) != n_clusters:
        raise ValueError(
            f"E-value parameters have {fitted_n} clusters, "
            f"but the cluster file assigns {n_clusters}"
        )
    target_fingerprint = database_fingerprint(
        entries, labels, int(db.get("n_codes", 80)))
    fitted_target = params.get("target_database_fingerprint")
    if fitted_target is not None and fitted_target != target_fingerprint:
        raise ValueError("E-value parameters were fitted for a different target database")
    if params.get("scheme") == _experimental_evalue.SCHEME:
        _experimental_evalue.validate_parameters(
            params, n_clusters, target_fingerprint,
            params.get("calibration_context_fingerprint"))
    if params.get("scheme") == _predicted_evalue.SCHEME:
        _predicted_evalue.validate_parameters(
            params, n_clusters, target_fingerprint,
            params.get("calibration_context_fingerprint"),
            [entry["tokens"] for entry in entries])
    for entry, label in zip(entries, labels):
        entry["cluster"] = label
    db["n_clusters"] = n_clusters
    db["clustered"] = True
    db["database_fingerprint"] = target_fingerprint
    db["evalue_params"] = params
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    torch.save(db, output)
    return {"n_chains": len(entries), "n_clusters": n_clusters,
            "output": output}


def short_id(raw, fallback):
    """Basename without extension, matching the analysis repo's convention."""
    if not raw:
        return str(fallback)
    stem = os.path.basename(str(raw))
    for ext in (".pdb", ".cif", ".json"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
    return stem


def load_database(db_path: str):
    """-> (raw db dict, entries, token arrays, short ids, n_codes, clusters)

    Databases built before clustering was stored carry no labels; each of
    their chains is then treated as its own cluster.
    """
    db = torch.load(db_path, map_location="cpu", weights_only=False)
    ents = db["entries"]
    toks = [np.asarray(e["tokens"], dtype=np.int64) for e in ents]
    ids = [short_id(e.get("id"), i) for i, e in enumerate(ents)]
    clusters = [e.get("cluster", f"_singleton_{i}") for i, e in enumerate(ents)]
    return db, ents, toks, ids, int(db.get("n_codes", 80)), clusters
