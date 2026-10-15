"""The search engine, loaded once and held resident.

RFoldSearch's index is not memory-mapped: a database is a torch archive
that must be deserialised, and the k-mer inverted index is then built in
process (about 20 s for half a million chains). Spawning the CLI per
request, as the Foldseek web server does with its C++ binary, would pay
that cost on every query. So the databases are loaded at startup and
every request is served from the resident objects.

Searches run in a worker thread, one at a time per database: the
prefilter and Smith-Waterman kernels hold the GIL only in numba nopython
regions, and serialising them keeps a slow query from starving the event
loop's other work while still using one core per search, which is what
the published timings assume.

Each database may carry a display sidecar, ``<db>.meta.json`` written by
``scripts/build_metadata.py``: the token-aligned sequence of every chain
and, for PDB chains, the RCSB description, organism and resolution. Hits
are annotated from it and given an alignment (``align.describe``); without
it a database still searches, just with bare ids and no sequence lines.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import align, fullatom

ALIGN_MAX = int(os.environ.get("RFOLDSEARCH_ALIGN_MAX", "100"))
URS_ID = re.compile(r"^URS[0-9A-F]{10}_(\d+)")
# NCBI Taxonomy names for the taxon ids that occur in the NuFold set
# (app/nufold_taxa.json: id -> {"name", "common"}), so predicted models show
# an organism instead of a bare taxon id.
try:
    TAXA = json.loads((Path(__file__).with_name("nufold_taxa.json")).read_text())
except OSError:
    TAXA = {}


def organism_of(target: str) -> Optional[str]:
    """'Homo sapiens (human) · taxon 9606' for an RNAcentral-style model id."""
    m = URS_ID.match(target)
    if not m:
        return None
    t = TAXA.get(m.group(1))
    if not t or not t.get("name"):
        return f"taxon {m.group(1)}"
    common = f" ({t['common']})" if t.get("common") else ""
    return f"{t['name']}{common} \u00b7 taxon {m.group(1)}"
_ANNOTATION = ("pdb", "chain", "auth_asym", "desc", "organism", "title",
               "method", "resolution", "rfam", "plddt", "msa_depth")


def first_rna_chain(data: bytes, filename: str):
    """Keep only the first RNA chain of an uploaded structure.

    The encoder takes one chain, so a file with several chains is cut down
    to its first RNA chain in file order (first model; label_asym_id for
    mmCIF, as the encoder reads it). Return (data, notice): the notice
    names the chains when more than one was RNA, and is None otherwise.
    """
    from rfoldsearch.edges import RNA_RESIDUE_NAMES

    text = data.decode(errors="replace")
    if filename.lower().endswith((".cif", ".mmcif")):
        import io
        import numpy as np
        import biotite.structure.io.pdbx as pdbx
        f = pdbx.CIFFile.read(io.StringIO(text))
        block = f.block
        cat = block["atom_site"]
        col = lambda k: cat[k].as_array(str)
        model = col("pdbx_PDB_model_num") if "pdbx_PDB_model_num" in cat else None
        first = np.ones(len(col("label_asym_id")), bool) if model is None else model == model[0]
        label, comp = col("label_asym_id"), np.char.upper(col("label_comp_id"))
        auth = col("auth_asym_id") if "auth_asym_id" in cat else label
        rna = first & np.isin(comp, sorted(RNA_RESIDUE_NAMES))
        chains, names = [], []
        for lab, au in zip(label[rna], auth[rna]):
            if lab not in chains:
                chains.append(lab)
                names.append(au)
        if not chains or len(set(label[first])) <= 1:
            return data, None
        keep = label == chains[0]
        block["atom_site"] = pdbx.CIFCategory(
            {k: pdbx.CIFColumn(pdbx.CIFData(c.data.array[keep]),
                               None if c.mask is None else pdbx.CIFData(c.mask.array[keep]))
             for k, c in cat.items()})
        if "atom_site_anisotrop" in block:     # per-atom, often most of the file
            del block["atom_site_anisotrop"]
        out = io.StringIO()
        f.write(out)
        data = out.getvalue().encode()
    else:
        lines, chains, out, model_done = text.splitlines(), [], [], False
        for ln in lines:
            if ln.startswith("ENDMDL"):
                model_done = True
            if (ln.startswith(("ATOM", "HETATM")) and not model_done and len(ln) > 21
                    and ln[17:20].strip().upper() in RNA_RESIDUE_NAMES
                    and ln[21] not in chains):
                chains.append(ln[21])
        every = {ln[21] for ln in lines if ln.startswith(("ATOM", "HETATM")) and len(ln) > 21}
        if not chains or len(every) <= 1:
            return data, None
        for ln in lines:
            if ln.startswith(("ATOM", "HETATM", "ANISOU", "TER")) and (len(ln) <= 21 or ln[21] != chains[0]):
                continue
            out.append(ln)
        data = ("\n".join(out) + "\n").encode()
        names = chains
    if len(names) <= 1:                         # other chains were protein, ligands, ...
        return data, None
    shown = ", ".join(n.strip() or "(blank)" for n in names[:10]) + (", …" if len(names) > 10 else "")
    return data, (f"The structure has {len(names)} RNA chains ({shown}); "
                  f"only the first, chain {names[0].strip() or '(blank)'}, was searched.")


@dataclass
class DatabaseHandle:
    key: str
    label: str
    path: str
    searcher: object = None
    n_chains: int = 0
    meta: dict = field(default_factory=dict)       # short id -> annotation
    source: Optional[str] = None
    built: Optional[str] = None
    notice: Optional[str] = None
    url: Optional[str] = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def info(self) -> dict:
        return {"key": self.key, "label": self.label,
                "n_chains": self.n_chains, "source": self.source,
                "built": self.built,
                "notice": self.notice,
                "url": self.url,
                "evalue_calibrated": self.searcher.power_ev is not None}


class Engine:
    """Holds the tokeniser and one Searcher per configured database."""

    def __init__(self, spec: str, model: Optional[str] = None):
        self.databases: dict[str, DatabaseHandle] = {}
        self._model = None
        self._ckpt = None
        self._model_path = model
        self._spec = spec
        self.ready = False
        self.error: Optional[str] = None

    # ---- startup ---------------------------------------------------
    def load(self) -> None:
        """Parse the spec, load the tokeniser and every database."""
        import warnings

        from rfoldsearch.database import short_id
        from rfoldsearch.search import Searcher, asset
        from rfoldsearch.tokenizer import load_model

        t0 = time.time()
        path = self._model_path or asset("desc_vqvae_best.pt")
        self._model, self._ckpt, _, _ = load_model(path)
        print(f"[engine] tokeniser {Path(path).name} in {time.time()-t0:.1f}s",
              flush=True)

        # Entries are separated by "|" and fields by "=" and the LAST ":",
        # because labels carry commas ("83,156 chains") and paths carry
        # colons on some hosts; splitting on either of those loses entries.
        for item in self._spec.split("|"):
            item = item.strip()
            if not item:
                continue
            key, _, rest = item.partition("=")
            key = key.strip()
            label, sep, dbpath = rest.rpartition(":")
            if not sep:                         # key=path, no label given
                label, dbpath = key, rest
            label, dbpath = label.strip(), dbpath.strip()
            if not dbpath:
                raise ValueError(f"database entry {item!r} has no path; "
                                 f'expected key=Label:/path/to/db.pt')
            if not os.path.exists(dbpath):
                raise FileNotFoundError(f"database {key!r}: no such file "
                                        f"{dbpath!r}")
            h = DatabaseHandle(key=key, label=label, path=dbpath)
            t0 = time.time()
            with warnings.catch_warnings():
                # the uncalibrated-E-value warning is surfaced per database
                # through info()["evalue_calibrated"] instead of the log
                warnings.simplefilter("ignore")
                h.searcher = Searcher(dbpath)
            h.n_chains = len(h.searcher.toks)
            sidecar = os.path.splitext(dbpath)[0] + ".meta.json"
            if os.path.exists(sidecar):
                with open(sidecar) as fh:
                    side = json.load(fh)
                h.source, h.built = side.get("source"), side.get("built")
                h.notice = side.get("notice")
                h.url = side.get("url")
                h.meta = {short_id(k, i): v
                          for i, (k, v) in enumerate(side["entries"].items())}
                missing = sum(cid not in h.meta for cid in h.searcher.ids)
                if missing:
                    print(f"[engine] {key}: sidecar lacks {missing} chains",
                          flush=True)
            self.databases[key] = h
            print(f"[engine] {key}: {h.n_chains:,} chains in "
                  f"{time.time()-t0:.1f}s"
                  f"{'' if h.meta else ' (no sidecar)'}", flush=True)
        if not self.databases:
            raise RuntimeError("no databases configured")
        self.ready = True

    # ---- one search ------------------------------------------------
    def tokenise(self, data: bytes, filename: str) -> dict:
        """Structure bytes -> the encoded entry, plus its token-aligned seq."""
        import torch
        from rfoldsearch.database import structure_to_entry
        from rfoldsearch.tokenizer import encode_entry

        suffix = ".cif" if filename.lower().endswith((".cif", ".mmcif")) else ".pdb"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
            fh.write(data)
            tmp = fh.name
        try:
            try:
                entry = structure_to_entry(tmp)
            except Exception as exc:
                raise ValueError(
                    f"could not read an RNA chain from {filename!r} as "
                    f"{'mmCIF' if suffix == '.cif' else 'PDB'} "
                    f"({type(exc).__name__}: {exc})") from exc
            enc = encode_entry(self._model, entry, torch.device("cpu"),
                               self._ckpt.get("norm_mean"),
                               self._ckpt.get("norm_std"),
                               active_blocks=self._ckpt.get("active_blocks"))
        finally:
            os.unlink(tmp)
        if enc is None:
            raise ValueError("structure could not be encoded: no usable "
                             "RNA backbone found (needs C3', C4' and O4')")
        enc["seq"] = "".join(c for c, ok in zip(entry.get("seq", ""),
                                                 entry["mask"]) if ok) or None
        return enc

    def search(self, db_key: str, enc: dict, top_k: int, top_n: int,
               min_diag: int, rerank: str,
               tie_seed: Optional[int]) -> dict:
        import numpy as np
        h = self.databases[db_key]
        s = h.searcher
        qtok = np.asarray(enc["tokens"], np.int64)
        t0 = time.perf_counter()
        with h.lock:
            hits = s.search(
                qtok, top_k=top_k, top_n=top_n, min_diag=min_diag,
                rerank=rerank, query_entry=enc, tie_seed=tie_seed)
        seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        for n, hit in enumerate(hits):
            ann = h.meta.get(hit["target"], {})
            hit.update({k: ann[k] for k in _ANNOTATION if ann.get(k) is not None})
            if "organism" not in hit:
                org = organism_of(hit["target"])
                if org:
                    hit["organism"] = org
            if n < ALIGN_MAX:
                hit["aln"] = self.alignment(h, enc, hit["target"])
        return {
            "database": h.info(),
            "rerank": rerank,
            "seconds": round(seconds, 4),
            "align_seconds": round(time.perf_counter() - t1, 4),
            "n_aligned_hits": min(len(hits), ALIGN_MAX),
            "hits": hits,
        }

    def alignment(self, h: DatabaseHandle, enc: dict, target: str,
                  keep_superposition: bool = False) -> dict:
        s = h.searcher
        ti = s.id2i[target]
        d = align.describe(enc["tokens"], s.toks[ti], enc, s.ents[ti], s.sub,
                           enc.get("seq"), h.meta.get(target, {}).get("seq"))
        if not keep_superposition:
            d.pop("_superpose", None)
        return d

    def superposition_pdb(self, db_key: str, enc: dict, target: str,
                          query_file: Optional[tuple] = None) -> str:
        """Query (chain A) and the target superposed onto it (chain B).

        Each chain is written with all its atoms when they can be had (the
        uploaded query file; a PDB hit from the RCSB) and as the five
        stored atoms per nucleotide otherwise. A REMARK line names the
        chains that are full-atom, for the viewer to pick a style.
        """
        h = self.databases[db_key]
        s = h.searcher
        d = self.alignment(h, enc, target, keep_superposition=True)
        ann = h.meta.get(target, {})
        if "_superpose" not in d:
            raise ValueError("too few aligned residues to superpose")
        R, tr, pq, pt = d["_superpose"]
        tent = s.ents[s.id2i[target]]
        full = []

        def chain(entry, seq, letter, sources, rot=None, trans=None, mark=None, first=1):
            for source in sources:
                try:
                    arr = fullatom.load_atoms(*source())
                    txt, nxt = fullatom.chain_pdb(
                        arr, align.masked_coords(entry)[:, 1], letter, rot, trans, mark, first)
                    full.append(letter)
                    return txt, nxt
                except Exception as exc:          # next source, then the stored atoms
                    print(f"[engine] full-atom chain {letter} unavailable: "
                          f"{type(exc).__name__}: {exc}", flush=True)
            return align.backbone_pdb(entry, seq, letter, rot=rot, trans=trans,
                                      highlight=mark, first_serial=first)

        qsrc = [lambda: query_file] if query_file else []
        # PDB-RNA names chains by label_asym_id, RNAsolo by auth_asym_id
        tsrc = []
        if ann.get("pdb") and ann.get("chain"):
            tsrc.append(lambda: (fullatom.rcsb_chain(ann["pdb"], ann["chain"]), "chain.cif"))
            tsrc.append(lambda: (fullatom.rcsb_chain(ann["pdb"], ann.get("auth_asym") or ann["chain"],
                                                     "auth"), "chain.cif"))
        qtxt, nxt = chain(enc, enc.get("seq"), "A", qsrc, mark=pq)
        ttxt, _ = chain(tent, ann.get("seq"), "B", tsrc, R, tr, pt, nxt)
        head = f"REMARK   1 FULLATOM {' '.join(full)}\n" if full else ""
        return head + qtxt + ttxt + "END\n"
