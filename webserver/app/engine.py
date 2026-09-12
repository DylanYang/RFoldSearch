"""The search engine, loaded once and held resident.

RFoldseek's index is not memory-mapped: a database is a torch archive
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
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DatabaseHandle:
    key: str
    label: str
    path: str
    searcher: object = None
    n_chains: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


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
        import torch
        from rfoldseek.search import Searcher, asset
        from rfoldseek.tokenizer import load_model

        t0 = time.time()
        path = self._model_path or asset("desc_vqvae_best.pt")
        self._model, self._ckpt, _, _ = load_model(path)
        print(f"[engine] tokeniser {Path(path).name} in {time.time()-t0:.1f}s",
              flush=True)

        for item in self._spec.split(","):
            item = item.strip()
            if not item:
                continue
            key, _, rest = item.partition("=")
            label, _, dbpath = rest.partition(":")
            if not dbpath:                      # key=path, no label given
                label, dbpath = key, rest
            h = DatabaseHandle(key=key, label=label, path=dbpath)
            t0 = time.time()
            h.searcher = Searcher(dbpath)
            h.n_chains = len(h.searcher.toks)
            self.databases[key] = h
            print(f"[engine] {key}: {h.n_chains:,} chains in "
                  f"{time.time()-t0:.1f}s", flush=True)
        if not self.databases:
            raise RuntimeError("no databases configured")
        self.ready = True

    # ---- one search ------------------------------------------------
    def tokenise(self, data: bytes, filename: str) -> dict:
        """Structure bytes -> the encoded entry the searcher consumes."""
        import torch
        from rfoldseek.database import structure_to_entry
        from rfoldseek.tokenizer import encode_entry

        suffix = ".cif" if filename.lower().endswith((".cif", ".mmcif")) else ".pdb"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
            fh.write(data)
            tmp = fh.name
        try:
            entry = structure_to_entry(tmp)
            enc = encode_entry(self._model, entry, torch.device("cpu"),
                               self._ckpt.get("norm_mean"),
                               self._ckpt.get("norm_std"))
        finally:
            os.unlink(tmp)
        if enc is None:
            raise ValueError("structure could not be encoded: no usable "
                             "RNA backbone found (needs C3', C4' and O4')")
        return enc

    def search(self, db_key: str, enc: dict, top_k: int, top_n: int,
               min_diag: int, rerank: str,
               tie_seed: Optional[int]) -> dict:
        import numpy as np
        h = self.databases[db_key]
        qtok = np.asarray(enc["tokens"], np.int64)
        t0 = time.perf_counter()
        with h.lock:
            hits = h.searcher.search(
                qtok, top_k=top_k, top_n=top_n, min_diag=min_diag,
                rerank=rerank, query_entry=enc, tie_seed=tie_seed)
        return {
            "database": {"key": h.key, "label": h.label,
                         "n_chains": h.n_chains},
            "query": {"n_tokens": int(len(qtok)),
                      "n_residues": int(enc.get("n_valid", len(qtok)))},
            "rerank": rerank,
            "seconds": round(time.perf_counter() - t0, 4),
            "hits": hits,
        }
