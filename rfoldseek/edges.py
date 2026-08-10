import numpy as np
import biotite.structure as struc
import biotite.structure.io as strucio

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
    path = "/Users/cassie/Desktop/train/PDB_00001DDY_1_A.cif"
    out = find_base_pair_and_stack_idx(path)
    print("pair_idx:\n",    out["pair_idx"])
    print("pair_orient:\n", out["pair_orient"])
    print("stack_idx:\n",   out["stack_idx"])
