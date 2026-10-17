"""Re-extract the experimental RNA chains for an isolated E-value comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import biotite.structure.io as strucio

from rfoldsearch.database import structure_to_entry


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def extract(path: Path) -> tuple[str, dict | None, str | None]:
    try:
        try:
            entry = structure_to_entry(str(path))
        except ValueError as exc:
            if "Expected a single chain" not in str(exc):
                raise
            chain_id = path.stem.rsplit("_", 1)[-1]
            atoms = strucio.load_structure(str(path), model=1)
            selected = atoms[atoms.chain_id == chain_id]
            if selected.array_length() == 0:
                raise ValueError(f"chain {chain_id!r} not present") from exc
            with tempfile.TemporaryDirectory(dir=path.parent) as temp:
                single_chain = Path(temp) / path.name
                strucio.save_structure(str(single_chain), selected)
                entry = structure_to_entry(str(single_chain))
        entry["id"] = path.stem
        if len(entry["seq"]) != len(entry["mask"]):
            raise ValueError("sequence and mask lengths differ")
        return path.name, entry, None
    except Exception as exc:
        return path.name, None, f"{type(exc).__name__}: {exc}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    paths = sorted(args.source.glob("*.cif"))
    if not paths:
        raise ValueError(f"no mmCIF files in {args.source}")
    args.out.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    with (args.out / "structures.jsonl").open("w", encoding="utf-8") as structures, \
            (args.out / "sequences.fasta").open("w", encoding="utf-8") as fasta, \
            ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (name, entry, error) in enumerate(pool.map(extract, paths), 1):
            if entry is not None:
                structures.write(json.dumps(entry, separators=(",", ":")) + "\n")
                fasta.write(f">{entry['id']}\n{entry['seq']}\n")
            records.append({"file": name, "sha256": digest(paths[i - 1]),
                            "length": len(entry["seq"]) if entry else None,
                            "error": error})
            if i % 25 == 0 or i == len(paths):
                print(f"extracted {i}/{len(paths)}", flush=True)
    summary = {"source": str(args.source.resolve()), "n_files": len(paths),
               "n_ok": sum(r["error"] is None for r in records),
               "n_failed": sum(r["error"] is not None for r in records),
               "records": records}
    (args.out / "source_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{summary['n_ok']} succeeded; {summary['n_failed']} failed", flush=True)


if __name__ == "__main__":
    main()
