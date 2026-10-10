"""Full-atom chains for the 3D view.

The databases keep five atoms per nucleotide: enough to search, too few to
draw a cartoon. The query is the file the user uploaded; a hit that comes
from the PDB is fetched from the RCSB ModelServer, one chain at a time, and
cached on disk. Residues are matched to tokens by their C4' position, so the
written PDB numbers residues by token, as the coarse model does, and the
superposition computed on the stored coordinates applies unchanged.
Residues that carry no token (modified nucleotides the tokenizer skipped,
residues with missing backbone atoms) are kept so the chain stays
continuous; they take the number of the preceding token and an insertion
code.
"""

from __future__ import annotations

import gzip
import io
import os
import re
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

MODELSERVER = "https://models.rcsb.org/v1/{pdb}/atoms?label_asym_id={asym}&encoding=cif"
CACHE = Path(os.environ.get("RFOLDSEARCH_CIF_CACHE",
                            Path(__file__).resolve().parent.parent / "cif_cache"))
FETCH_TIMEOUT = float(os.environ.get("RFOLDSEARCH_CIF_TIMEOUT", "20"))
MATCH_TOL = 0.05           # Å; stored coordinates are the deposited ones
MIN_MATCHED = 0.9          # fraction of tokens that must find their residue
_ICODES = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_SAFE = re.compile(r"^[A-Za-z0-9]{1,12}$")


def rcsb_chain(pdb: str, asym: str) -> bytes:
    """mmCIF text of one label_asym chain, from the cache or the ModelServer."""
    if not (_SAFE.match(pdb or "") and _SAFE.match(asym or "")):
        raise ValueError(f"not a PDB chain: {pdb!r} {asym!r}")
    path = CACHE / f"{pdb.upper()}_{asym}.cif.gz"
    if path.exists():
        return gzip.decompress(path.read_bytes())
    url = MODELSERVER.format(pdb=pdb.lower(), asym=asym)
    with urllib.request.urlopen(url, timeout=FETCH_TIMEOUT) as r:
        data = r.read()
    if b"_atom_site." not in data:
        raise ValueError(f"no atoms for {pdb} chain {asym}")
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(data))
    tmp.replace(path)
    return data


def load_atoms(data: bytes, filename: str):
    """First model of a PDB or mmCIF file, without hydrogens or water."""
    import biotite.structure as struc
    import biotite.structure.io.pdb as pdbio
    import biotite.structure.io.pdbx as pdbx

    if filename.lower().endswith((".cif", ".mmcif")) or b"_atom_site." in data[:200000]:
        f = pdbx.CIFFile.read(io.StringIO(data.decode(errors="replace")))
        arr = pdbx.get_structure(f, model=1, use_author_fields=False)
    else:
        with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as fh:
            fh.write(data.decode(errors="replace"))
            tmp = fh.name
        try:
            arr = pdbio.PDBFile.read(tmp).get_structure(model=1)
        finally:
            os.unlink(tmp)
    arr = arr[(arr.element != "H") & (arr.element != "D")]
    return arr[~struc.filter_solvent(arr)]


def chain_pdb(arr, token_c4, chain: str, rot=None, trans=None, highlight=None,
              first_serial: int = 1):
    """PDB records of the chain whose residues carry the given tokens.

    ``token_c4`` holds each token's stored C4' (NaN where the token is
    masked), in the same frame as ``arr``. Returns (text, next serial);
    raises ValueError when fewer than MIN_MATCHED of the tokens are found.
    """
    import biotite.structure as struc

    starts = struc.get_residue_starts(arr, add_exclusive_stop=True)
    c4 = np.full((len(starts) - 1, 3), np.nan)
    for i in range(len(starts) - 1):
        sl = slice(starts[i], starts[i + 1])
        hit = np.nonzero(arr.atom_name[sl] == "C4'")[0]
        if hit.size:
            c4[i] = arr.coord[sl][hit[0]]
    have = np.nonzero(np.isfinite(c4[:, 0]))[0]
    toks = np.nonzero(np.isfinite(token_c4[:, 0]))[0]
    if not have.size or not toks.size:
        raise ValueError("no nucleotides to match")
    d = np.linalg.norm(c4[have][:, None, :] - token_c4[toks][None, :, :], axis=2)
    best = d.argmin(axis=1)
    res_tok = {}
    for k, (ri, j) in enumerate(zip(have, best)):
        if d[k, j] < MATCH_TOL:
            res_tok[int(ri)] = int(toks[j])
    if len(set(res_tok.values())) < MIN_MATCHED * toks.size:
        raise ValueError(f"only {len(set(res_tok.values()))} of {toks.size} "
                         f"nucleotides found in the full-atom chain")
    # the chain that holds the matched residues, nucleotides only
    chains = arr.chain_id[starts[:-1]]
    keep_chain = max(set(chains[list(res_tok)]),
                     key=lambda c: sum(chains[r] == c for r in res_tok))
    marked = set(int(x) for x in highlight) if highlight is not None else set()
    xyz = arr.coord if rot is None else arr.coord @ rot.T + trans
    lines, serial, prev, icode = [], first_serial, 0, 0
    for i in range(len(starts) - 1):
        if chains[i] != keep_chain or not np.isfinite(c4[i, 0]):
            continue
        if i in res_tok:
            num, ins, prev, icode = res_tok[i] + 1, " ", res_tok[i] + 1, 0
            b = 1.0 if res_tok[i] in marked else 0.0
        else:
            if icode >= len(_ICODES):
                continue
            num, ins, b = prev, _ICODES[icode], 0.0
            icode += 1
        for k in range(starts[i], starts[i + 1]):
            name, el = arr.atom_name[k], arr.element[k]
            field = name if len(name) >= 4 or len(el) == 2 else f" {name}"
            x, y, z = xyz[k]
            lines.append(
                f"ATOM  {serial % 100000:5d} {field:<4s} {arr.res_name[k][:3]:>3s} "
                f"{chain}{num:4d}{ins}   {x:8.3f}{y:8.3f}{z:8.3f}  1.00{b:6.2f}"
                f"          {el:>2s}")
            serial += 1
    lines.append("TER")
    return "\n".join(lines) + "\n", serial
