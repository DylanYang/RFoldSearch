"""US-align TM-score reranking: the only module that knows US-align exists.

RFoldseek-TM keeps the token prefilter and replaces the Smith-Waterman
reranking stage with a real structural superposition.

The binary is invoked once for the whole candidate pool via ``-dir2``, so
per-candidate process overhead does not scale with ``--top_n``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import numpy as np

USALIGN_URL = "https://github.com/pylelab/USalign"
MIN_RESIDUES = 3


class USalignNotFound(FileNotFoundError):
    """The USalign binary could not be resolved."""


class USalignFailed(RuntimeError):
    """The USalign binary ran but exited non-zero."""


def find_usalign(explicit: str = None) -> str:
    """Resolve the USalign binary: explicit path, then $USALIGN, then PATH."""
    for cand in (explicit, os.environ.get("USALIGN"), shutil.which("USalign")):
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    raise USalignNotFound(
        "USalign binary not found. Pass --usalign PATH, set the USALIGN "
        f"environment variable, or put USalign on your PATH. Build it from "
        f"{USALIGN_URL}."
    )


def write_c3_pdb(coords, mask, path: str, chain: str = "A") -> int:
    """Write the C3' atom of every valid residue as a minimal PDB.

    The database stores no sequence, so every residue is written as the
    placeholder base G; callers pass ``-mol RNA`` so US-align does not have to
    infer the molecule type. TM-score is computed from coordinates alone, so
    the placeholder does not affect the result. Returns the residue count.
    """
    coords = np.asarray(coords, dtype=np.float64)
    keep = coords[np.asarray(mask, dtype=bool), 0, :]
    with open(path, "w") as fh:
        for i, (x, y, z) in enumerate(keep, 1):
            fh.write(
                f"ATOM  {i:5d}  C3'   G {chain}{i:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00" + " " * 10 + " C\n"
            )
        fh.write(f"TER   {len(keep) + 1:5d}        G {chain}{len(keep):4d}\n")
        fh.write("END\n")
    return int(len(keep))


def parse_outfmt2(text: str) -> dict:
    """Parse ``USalign -outfmt 2`` output into {candidate index: (TM1, TM2)}.

    Columns are PDBchain1 PDBchain2 TM1 TM2 RMSD ID1 ID2 IDali L1 L2 Lali;
    TM1 is normalised by chain 1 (the query), TM2 by chain 2 (the target).
    Candidate files are named ``<index>.pdb``, so the index is recovered from
    the chain-2 field.
    """
    out = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) < 4:
            continue
        stem = os.path.basename(f[1].split(":")[0])
        if stem.endswith(".pdb"):
            stem = stem[:-4]
        try:
            idx, tm1, tm2 = int(stem), float(f[2]), float(f[3])
        except ValueError:
            continue
        out[idx] = (tm1, tm2)
    return out


def assemble_scores(index_to_id: dict, parsed: dict) -> dict:
    """{index: id} + {index: (TM1, TM2)} -> {id: (TM1, TM2)}, missing -> zero."""
    return {cid: parsed.get(idx, (0.0, 0.0)) for idx, cid in index_to_id.items()}


def score_pool(binary: str, query_coords, query_mask, candidates,
               min_residues: int = MIN_RESIDUES, fast: bool = False):
    """Score a whole candidate pool in one US-align call.

    ``candidates`` is a sequence of ``(id, coords, mask)``. Returns
    ``({id: (tm_query, tm_target)}, [skipped ids])``; a candidate is skipped
    when it has fewer than ``min_residues`` valid residues.

    ``fast`` passes US-align's ``-fast`` flag, which shortens its heuristic
    search for the superposition. Measured at roughly 2x on paired probes
    over 10,000 targets; TM-scores are slightly lower and a few near-ties
    can swap, so it is off by default and worth enabling for long queries,
    whose cost grows with the product of the two chain lengths.
    """
    with tempfile.TemporaryDirectory(prefix="rfoldseek_tm_") as tmp:
        qpath = os.path.join(tmp, "query.pdb")
        if write_c3_pdb(query_coords, query_mask, qpath) < min_residues:
            raise ValueError(
                f"query has fewer than {min_residues} valid residues; "
                "cannot compute a TM-score"
            )
        tdir = os.path.join(tmp, "targets")
        os.makedirs(tdir)
        index_to_id, names, skipped = {}, [], []
        for idx, (cid, coords, mask) in enumerate(candidates):
            name = f"{idx}.pdb"
            if write_c3_pdb(coords, mask, os.path.join(tdir, name)) < min_residues:
                skipped.append(cid)
                continue
            index_to_id[idx] = cid
            names.append(name)
        if not names:
            return {}, skipped
        listfile = os.path.join(tmp, "list.txt")
        with open(listfile, "w") as fh:
            fh.write("\n".join(names) + "\n")
        cmd = [binary, qpath, "-dir2", tdir + os.sep, listfile,
               "-mol", "RNA", "-outfmt", "2"]
        if fast:
            cmd.append("-fast")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise USalignFailed(
                f"US-align failed (exit {proc.returncode}): {proc.stderr.strip()}"
            )
        return assemble_scores(index_to_id, parse_outfmt2(proc.stdout)), skipped
