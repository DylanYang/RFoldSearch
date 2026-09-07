"""Build and load searchable token databases (.pt)."""
from __future__ import annotations
import json
import os

import numpy as np
import torch

from .backbone import extract_backbone_coords
from .edges import find_base_pair_and_stack_idx
from .tokenizer import encode_entry, load_model


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
                   min_len: int = 20, device: str = "cpu",
                   progress: bool = True, clusters: str | None = None) -> dict:
    """Encode every entry in the given JSONL files into one .pt database.

    ``clusters`` is the CD-HIT .clstr (or TSV) that groups near-duplicate
    chains. E-values are counted in clusters, so the mapping is stored with
    the database; without it every chain is its own cluster and the E-value
    over-counts a redundant database.
    """
    dev = torch.device(device)
    model, ckpt, n_codes, _ = load_model(checkpoint, dev)
    norm_mean = ckpt.get("norm_mean", None)
    norm_std = ckpt.get("norm_std", None)

    entries, seen = [], set()
    for data_path in data_paths:
        with open(data_path) as f:
            for line in f:
                e = json.loads(line)
                eid = str(e.get("id", ""))
                if eid in seen:
                    continue
                seen.add(eid)
                if sum(e["mask"]) < min_len:
                    continue
                enc = encode_entry(model, e, dev, norm_mean, norm_std)
                if enc is not None:
                    entries.append(enc)
                if progress and len(entries) % 100 == 0 and entries:
                    print(f"  encoded {len(entries)}", flush=True)

    labels = (assign_clusters(entries, read_clusters(clusters)) if clusters
              else [f"_singleton_{i}" for i in range(len(entries))])
    for e, lab in zip(entries, labels):
        e["cluster"] = lab
    db = {"n_codes": int(n_codes), "entries": entries, "checkpoint": checkpoint,
          "n_clusters": len(set(labels)), "clustered": bool(clusters)}
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    torch.save(db, output)
    lens = np.array([len(e["tokens"]) for e in entries]) if entries else np.zeros(1)
    print(f"Saved: {output}\n  Entries: {len(entries)}  "
          f"(length: mean={lens.mean():.0f}, median={np.median(lens):.0f}, "
          f"max={lens.max():.0f})\n  Clusters: {len(set(labels))}"
          f"{'' if clusters else ' (unclustered: every chain its own cluster)'}")
    return db


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
