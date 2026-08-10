# Ported from RFoldseek-paper-code preprocessing/extract_backbone.py; only imports adjusted.
import json
import os
import warnings
import glob
import numpy as np

import biotite.structure as struc
import biotite.structure.io as strucio

warnings.filterwarnings("ignore", category=UserWarning, module="biotite")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="biotite")
warnings.filterwarnings("ignore", message=".*IncompleteStructureWarning.*")
try:
    from biotite.structure import IncompleteStructureWarning
    warnings.filterwarnings("ignore", category=IncompleteStructureWarning)
except ImportError:
    pass
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from .edges import find_base_pair_and_stack_idx

BACKBONE_ATOMS = ["C3'", "C4'", "O4'"]
BASE_ANCHOR_ATOMS = {
    "A": "N9",
    "G": "N9",
    "C": "N1",
    "U": "N1",
}
# Sugar + phosphate atoms excluded when computing a base-centroid anchor
# (used only when RFS_BASE_ANCHOR=centroid).
_SUGAR_PHOSPHATE_ATOMS = {
    "P", "OP1", "OP2", "OP3", "O5'", "C5'", "C4'", "O4'",
    "C3'", "O3'", "C2'", "O2'", "C1'",
}
COORD_ATOMS = BACKBONE_ATOMS + ["C1'", "BASE"]

# RNA residue identification — must match cdhit_dedup.py exactly so the
# sequence used for clustering and the coordinates used for training
# refer to the same set of residues. Do NOT use `struc.filter_nucleotides`:
# some biotite versions exclude modified ribonucleosides, which would
# silently truncate tRNA-like structures and create a length mismatch
# with the CD-HIT clustering output.
RNA_BASES = {"A", "C", "G", "U"}
MODIFIED_TO_CANONICAL = {
    "PSU": "U", "5MU": "U", "4SU": "U", "H2U": "U", "UR3": "U",
    "5MC": "C", "OMC": "C", "5BU": "U", "OMG": "G", "1MG": "G",
    "2MG": "G", "M2G": "G", "7MG": "G", "YG": "G", "YYG": "G",
    "1MA": "A", "6MZ": "A", "MA6": "A", "I": "A",
}
RNA_RESIDUE_NAMES = RNA_BASES | set(MODIFIED_TO_CANONICAL.keys())


def _chain_id_from_filename(path):
    """Extract chain ID from filename PDB_0000XXXX_M_C.cif"""
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        return parts[1]
    return None


def _is_rna_residue(arr, start_idx):
    """Return True if the residue starting at `start_idx` is an RNA
    residue (canonical A/C/G/U or a known modified ribonucleoside)."""
    name = str(arr.res_name[int(start_idx)]).strip().upper()
    return name in RNA_RESIDUE_NAMES


def _canonical_base_name(res_name):
    name = str(res_name).strip().upper()
    return MODIFIED_TO_CANONICAL.get(name, name)


def extract_backbone_coords(path, chain_id=None, max_len=None):
    """
    Extract backbone coordinates and mask from an RNA structure.

    Args:
        path (str): Path to PDB or CIF file
        chain_id (str or None): Chain ID. If None, inferred from filename
            or the alphabetically first chain for deterministic behaviour.
        max_len (int or None): If set, truncate to this length

    Returns:
        coords (np.ndarray): (L, 5, 3) — C3', C4', O4', C1',
            and base anchor coordinates. The base anchor is N9 for purines
            and N1 for pyrimidines. C3'/C4'/O4' define the validity mask;
            C1'/base-anchor are optional and filled with zeros if absent.
        mask (np.ndarray): (L,) — 1 = valid backbone, 0 = missing C3'/C4'/O4'
    """
    # Load without `extra_fields` / `altloc` — those options can fail on
    # older or non-standard CIFs and silently drop entries. Keep the call
    # minimal and defer field selection to our own logic below.
    arr = strucio.load_structure(path, model=1)

    if arr.array_length() == 0:
        raise ValueError("No atoms found in the structure.")

    # Determine which chain to use, then restrict to RNA residues only.
    all_starts = struc.get_residue_starts(arr)
    chain_ids = arr.chain_id[all_starts]
    unique_chains = np.unique(chain_ids)

    if unique_chains.size == 1:
        pass  # single chain – nothing to do
    elif chain_id is not None and chain_id in unique_chains:
        arr = arr[arr.chain_id == chain_id]
    else:
        # Infer from filename, otherwise pick alphabetically first for
        # deterministic behaviour.
        inferred = _chain_id_from_filename(path)
        if inferred is not None and inferred in unique_chains:
            arr = arr[arr.chain_id == inferred]
        else:
            first = sorted(unique_chains.tolist())[0]
            arr = arr[arr.chain_id == first]

    if arr.array_length() == 0:
        raise ValueError("No atoms in selected chain.")

    # Filter to RNA residues (canonical + modified). We build a
    # per-atom mask by checking the residue name at each residue start
    # and propagating the decision to every atom of that residue.
    starts = struc.get_residue_starts(arr)
    keep_atom = np.zeros(arr.array_length(), dtype=bool)
    for i, start in enumerate(starts):
        end = int(starts[i + 1]) if i + 1 < len(starts) else arr.array_length()
        if _is_rna_residue(arr, start):
            keep_atom[int(start):end] = True
    arr = arr[keep_atom]
    if arr.array_length() == 0:
        raise ValueError("No RNA residues in selected chain.")
    nuc_starts = struc.get_residue_starts(arr)

    # Extract per-residue backbone atoms + canonical nucleotide identity
    coords = []
    mask = []
    seq = []
    L = len(nuc_starts)

    for i in range(L):
        start = int(nuc_starts[i])
        end = int(nuc_starts[i + 1]) if i + 1 < L else arr.array_length()
        res_atoms = arr[start:end]

        res_coords = []
        missing = False
        for atom_name in BACKBONE_ATOMS:
            atom_mask = res_atoms.atom_name == atom_name
            if np.any(atom_mask):
                idx = np.where(atom_mask)[0][0]
                res_coords.append(res_atoms.coord[idx].tolist())
            else:
                res_coords.append([0.0, 0.0, 0.0])
                missing = True

        c1_mask = res_atoms.atom_name == "C1'"
        if np.any(c1_mask):
            idx = np.where(c1_mask)[0][0]
            res_coords.append(res_atoms.coord[idx].tolist())
        else:
            res_coords.append([0.0, 0.0, 0.0])

        base = _canonical_base_name(res_atoms.res_name[0])
        if os.environ.get("RFS_BASE_ANCHOR", "glycosidic") == "centroid":
            # Base reference = centroid of base heavy atoms (exclude
            # sugar/phosphate and hydrogens). Ablation alternative to N9/N1.
            base_atom_mask = np.array([
                (nm not in _SUGAR_PHOSPHATE_ATOMS) and (not str(nm).startswith("H"))
                for nm in res_atoms.atom_name
            ], dtype=bool)
            if np.any(base_atom_mask):
                res_coords.append(res_atoms.coord[base_atom_mask].mean(axis=0).tolist())
            else:
                res_coords.append([0.0, 0.0, 0.0])
        else:
            anchor_atom = BASE_ANCHOR_ATOMS.get(base)
            anchor_mask = res_atoms.atom_name == anchor_atom if anchor_atom else np.zeros(
                res_atoms.array_length(), dtype=bool
            )
            if np.any(anchor_mask):
                idx = np.where(anchor_mask)[0][0]
                res_coords.append(res_atoms.coord[idx].tolist())
            else:
                res_coords.append([0.0, 0.0, 0.0])

        coords.append(res_coords)
        mask.append(0 if missing else 1)
        # Single-letter A/U/C/G (modified residues already mapped via
        # _canonical_base_name); unknown / non-canonical → 'N'.
        seq.append(base if base in ("A", "U", "C", "G") else "N")

    # Truncation only (no padding)
    if max_len is not None and len(coords) > max_len:
        coords, mask, seq = coords[:max_len], mask[:max_len], seq[:max_len]

    coords = np.array(coords, dtype=np.float32)
    mask = np.array(mask, dtype=np.int64)
    seq = "".join(seq)
    return coords, mask, seq


