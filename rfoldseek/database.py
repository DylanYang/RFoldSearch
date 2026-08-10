"""Build and load searchable token databases (.pt)."""
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


def build_database(checkpoint: str, data_paths: list[str], output: str,
                   min_len: int = 20, device: str = "cpu",
                   progress: bool = True) -> dict:
    """Encode every entry in the given JSONL files into one .pt database."""
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

    db = {"n_codes": int(n_codes), "entries": entries, "checkpoint": checkpoint}
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    torch.save(db, output)
    lens = np.array([len(e["tokens"]) for e in entries]) if entries else np.zeros(1)
    print(f"Saved: {output}\n  Entries: {len(entries)}  "
          f"(length: mean={lens.mean():.0f}, median={np.median(lens):.0f}, "
          f"max={lens.max():.0f})")
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
    """-> (raw db dict, entries, token arrays, short ids, n_codes)"""
    db = torch.load(db_path, map_location="cpu", weights_only=False)
    ents = db["entries"]
    toks = [np.asarray(e["tokens"], dtype=np.int64) for e in ents]
    ids = [short_id(e.get("id"), i) for i, e in enumerate(ents)]
    return db, ents, toks, ids, int(db.get("n_codes", 80))
