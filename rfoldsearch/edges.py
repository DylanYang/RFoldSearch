import inspect
import threading
from contextlib import contextmanager

import numpy as np
import biotite.structure as struc
import biotite.structure.io as strucio

try:
    import biotite.structure.basepairs as _bp
except ImportError:                                    # pragma: no cover
    _bp = None

# Must stay in sync with extract_backbone.py and cdhit_dedup.py so that
# every step of the pipeline operates on the same set of residues.
# Otherwise pair_idx / stack_idx returned here would index into a
# residue ordering that differs from the coords array produced by
# extract_backbone, causing silent off-by-one corruption.
RNA_BASES = {"A", "C", "G", "U"}
MODIFIED_TO_CANONICAL = {
    "PSU": "U", "5MU": "U", "4SU": "U", "H2U": "U", "UR3": "U",
    "5MC": "C", "OMC": "C", "5BU": "U", "OMG": "G", "1MG": "G",
    "2MG": "G", "M2G": "G", "7MG": "G", "YG": "G", "YYG": "G",
    "1MA": "A", "6MZ": "A", "MA6": "A", "I": "A",
}
RNA_RESIDUE_NAMES = RNA_BASES | set(MODIFIED_TO_CANONICAL.keys())


# ---- per-call cache of biotite's nucleotide base-frame fit ---------------
# biotite's base_pairs and base_stacking fit every nucleotide of every
# candidate pair to a standard base frame (basepairs._match_base), so one
# nucleotide is refitted once per candidate pair, about 18 times per residue.
# The fit depends only on the nucleotide's atoms, so while
# find_base_pair_and_stack_idx runs the result is cached by atom content and a
# copy is handed out on every repeat: biotite computes exactly what it would
# without the cache. The cache is thread-local and dropped when the call ends.
# If biotite's private function is missing or its signature changes, nothing
# is patched and the uncached path runs (same output, slower).
MEMOISE_BASE_FRAMES = True
_local = threading.local()


def _signature_ok(fn):
    try:
        return list(inspect.signature(fn).parameters) == [
            "nucleotide", "min_atoms_per_base"]
    except (TypeError, ValueError):
        return False


_ORIG_MATCH_BASE = getattr(_bp, "_match_base", None)
if _ORIG_MATCH_BASE is not None and not _signature_ok(_ORIG_MATCH_BASE):
    _ORIG_MATCH_BASE = None


def _frame_key(nucleotide, min_atoms_per_base):
    """Bytes identifying everything the base-frame fit can read, or None if
    some annotation cannot be keyed by value (object arrays)."""
    parts = [str(min_atoms_per_base).encode(),
             np.ascontiguousarray(nucleotide.coord).tobytes()]
    for name in sorted(nucleotide.get_annotation_categories()):
        a = np.asarray(nucleotide.get_annotation(name))
        if a.dtype.hasobject:
            return None
        parts.append(name.encode() + b"|" + a.dtype.str.encode() + b"|"
                     + np.ascontiguousarray(a).tobytes())
    return b"\x00".join(parts)


def _match_base_cached(nucleotide, min_atoms_per_base):
    cache = getattr(_local, "cache", None)
    key = None if cache is None else _frame_key(nucleotide, min_atoms_per_base)
    if key is None:
        return _ORIG_MATCH_BASE(nucleotide, min_atoms_per_base)
    if key in cache:
        hit = cache[key]
        return None if hit is None else hit.copy()
    out = _ORIG_MATCH_BASE(nucleotide, min_atoms_per_base)
    cache[key] = None if out is None else out.copy()
    return out


@contextmanager
def _memoised_base_frames():
    if not MEMOISE_BASE_FRAMES or _ORIG_MATCH_BASE is None:
        yield
        return
    # Installed once; outside a cached call it forwards to biotite unchanged.
    if _bp._match_base is not _match_base_cached:
        _bp._match_base = _match_base_cached
    previous = getattr(_local, "cache", None)
    _local.cache = {}
    try:
        yield
    finally:
        _local.cache = previous


def find_base_pair_and_stack_idx(
        path: str,
        model: int = 1,
        unique_bp: bool = True,  # each nucleotide pairs with at most one partner
):
    """
    Extract the three edge fields consumed by the 49-D descriptor:
        pair_idx     — sequence-nearest base-pair partner per residue
        pair_orient  — Leontis--Westhof cis/trans label (0 none, 1 cis, 2 trans)
        stack_idx    — sequence-nearest stacking partner per residue
    """
    with _memoised_base_frames():
        return _find_edges(path, model, unique_bp)


def _find_edges(path, model, unique_bp):
    # Match backbone.py's loader (label_asym_id, not author) so residue
    # counts stay in sync — an auth-only mismatch would leave edges
    # indexed 0..17 while coords run 0..19, and the encoder would drop
    # the entry as a shape mismatch.
    if path.lower().endswith((".cif", ".mmcif", ".cif.gz")):
        import biotite.structure.io.pdbx as pdbx
        cif_file = pdbx.CIFFile.read(path)
        arr = pdbx.get_structure(cif_file, model=model, use_author_fields=False)
    else:
        arr = strucio.load_structure(path, model=model)
    if arr.array_length() == 0:
        raise ValueError("No atoms found in the structure.")

    # Restrict to RNA residues (canonical + known modifications).
    starts = struc.get_residue_starts(arr)
    keep_atom = np.zeros(arr.array_length(), dtype=bool)
    for i, start in enumerate(starts):
        end = int(starts[i + 1]) if i + 1 < len(starts) else arr.array_length()
        name = str(arr.res_name[int(start)]).strip().upper()
        if name in RNA_RESIDUE_NAMES:
            keep_atom[int(start):end] = True
    arr = arr[keep_atom]
    if arr.array_length() == 0:
        raise ValueError("No RNA residues found in the structure.")

    # Rename modified residues to canonical bases in place so biotite's
    # `base_pairs` / `base_stacking` (which only recognise A/C/G/U
    # geometry) can detect pairings involving PSU, 5MU, 1MA etc.
    res_names = np.array(arr.res_name, copy=True)
    for mod, canon in MODIFIED_TO_CANONICAL.items():
        res_names[res_names == mod] = canon
    arr.res_name = res_names

    nuc_starts = struc.get_residue_starts(arr)
    L = int(nuc_starts.shape[0])
    start_to_nuc = {int(s): i for i, s in enumerate(nuc_starts)}

    chain_per_nuc = arr.chain_id[nuc_starts]
    unique_chains = np.unique(chain_per_nuc)
    if unique_chains.size != 1:
        raise ValueError(
            f"Expected a single chain, but found {unique_chains.size} chains: "
            f"{', '.join(map(str, unique_chains))}"
        )

    # base-pairing — keep the sequence-nearest partner per residue so
    # base triples reduce deterministically. Ties broken by Biotite's
    # internal pair order.
    pair_idx = np.full((L, 1), -1, dtype=np.int32)
    pair_orient = np.full((L, 1), -1, dtype=np.int32)
    try:
        bp_pairs = struc.base_pairs(arr, unique=unique_bp)
    except (TypeError, ValueError, IndexError):
        bp_pairs = np.empty((0, 2), dtype=int)
    try:
        bp_orient = struc.base_pairs_glycosidic_bond(arr, bp_pairs)
    except (TypeError, ValueError, IndexError):
        bp_orient = np.zeros((len(bp_pairs),), dtype=np.int32)

    bp_neighbors = [[] for _ in range(L)]
    for k, (s_i, s_j) in enumerate(bp_pairs):
        i = start_to_nuc.get(int(s_i))
        j = start_to_nuc.get(int(s_j))
        if i is None or j is None:
            continue
        bp_neighbors[i].append((j, k))
        bp_neighbors[j].append((i, k))

    for i, candidates in enumerate(bp_neighbors):
        if not candidates:
            continue
        j, k = min(candidates, key=lambda t: (abs(t[0] - i), t[1]))
        pair_idx[i, 0] = j
        pair_orient[i, 0] = int(bp_orient[k]) if k < len(bp_orient) else 0

    # base-stacking (sequence-nearest partner only)
    stack_idx = np.full((L, 1), -1, dtype=np.int32)
    try:
        st_pairs = struc.base_stacking(arr)
    except (TypeError, ValueError, IndexError):
        st_pairs = np.empty((0, 2), dtype=int)

    neighbors = [[] for _ in range(L)]
    for s_i, s_j in st_pairs:
        i = start_to_nuc.get(int(s_i))
        j = start_to_nuc.get(int(s_j))
        if i is None or j is None:
            continue
        neighbors[i].append(j)
        neighbors[j].append(i)

    for i, nb in enumerate(neighbors):
        if nb:
            stack_idx[i, 0] = min(set(nb), key=lambda n: abs(n - i))

    return {
        "pair_idx":    pair_idx,     # (L, 1)
        "pair_orient": pair_orient,  # -1 none, 0 invalid, 1 cis, 2 trans
        "stack_idx":   stack_idx,    # (L, 1)
    }


def find_base_pair_and_stack_edges(*args, **kwargs):
    """Backward-compatible alias for older callers."""
    return find_base_pair_and_stack_idx(*args, **kwargs)


if __name__ == '__main__':
    import sys
    out = find_base_pair_and_stack_idx(sys.argv[1])
    print("pair_idx:\n",    out["pair_idx"])
    print("pair_orient:\n", out["pair_orient"])
    print("stack_idx:\n",   out["stack_idx"])
