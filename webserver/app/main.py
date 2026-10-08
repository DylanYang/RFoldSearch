"""RFoldseek search server.

A structure is uploaded (or pasted), tokenised with the deployed VQ-VAE and
searched against one or more resident databases. Each search is stored
under a short id, so ``/result/<id>`` is a shareable page and survives a
restart; the hit tables, alignments and superpositions are served from
that record by the JSON API and rendered by static/index.html.

Databases are declared in RFOLDSEEK_DBS as pipe-separated key=label:path
entries, so the same server can hold the experimental sets and a
half-million predicted one.

Searches run on a small thread pool and are bounded by a semaphore rather
than a queue with unbounded admission: a search holds a core for its
duration, so accepting more concurrent work than there are cores makes
every request slower without serving anyone sooner. Over the limit the
server says so with 503 instead of silently queueing.
"""
from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import re
import secrets
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import (FileResponse, JSONResponse, PlainTextResponse,
                               Response)
from fastapi.staticfiles import StaticFiles

from . import remote
from .engine import Engine

HERE = Path(__file__).resolve().parent
STATIC = HERE.parent / "static"
EXAMPLES = STATIC / "examples"

MAX_UPLOAD = int(os.environ.get("RFOLDSEEK_MAX_UPLOAD_MB", "32")) * 1024 * 1024
MAX_CONCURRENT = int(os.environ.get("RFOLDSEEK_MAX_CONCURRENT",
                                    str(max(1, (os.cpu_count() or 2) - 1))))
TOP_N_CAP = int(os.environ.get("RFOLDSEEK_TOP_N_CAP", "2000"))
JOBS_DIR = Path(os.environ.get("RFOLDSEEK_JOBS_DIR", HERE.parent / "jobs"))
JOB_CACHE = 200                         # records kept in memory
JOB_ID = re.compile(r"^[A-Za-z0-9_-]{8,32}$")

app = FastAPI(title="RFoldseek search", docs_url="/api/docs",
              openapi_url="/api/openapi.json")
engine = Engine(spec=os.environ.get("RFOLDSEEK_DBS", ""),
                model=os.environ.get("RFOLDSEEK_MODEL") or None)
pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT,
                          thread_name_prefix="rfsearch")
slots = asyncio.Semaphore(MAX_CONCURRENT)
_jobs: "OrderedDict[str, dict]" = OrderedDict()


# ---- job records -----------------------------------------------------
def _save_job(job: dict) -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = JOBS_DIR / f".{job['id']}.json"
    tmp.write_text(json.dumps(job))
    tmp.replace(JOBS_DIR / f"{job['id']}.json")
    _remember(job)


def _remember(job: dict) -> None:
    _jobs[job["id"]] = job
    _jobs.move_to_end(job["id"])
    while len(_jobs) > JOB_CACHE:
        _jobs.popitem(last=False)


def _load_job(job_id: str) -> dict:
    if not JOB_ID.match(job_id):
        raise HTTPException(404, "no such result")
    if job_id in _jobs:
        _jobs.move_to_end(job_id)
        return _jobs[job_id]
    path = JOBS_DIR / f"{job_id}.json"
    if not path.exists():
        raise HTTPException(404, "no such result (it may have expired)")
    job = json.loads(path.read_text())
    _remember(job)
    return job


def _query_entry(job: dict) -> dict:
    q = job["query"]
    return {"tokens": np.asarray(q["tokens"], np.int64),
            "coords": np.asarray(q["coords"], np.float32),
            "mask": np.asarray(q["mask"], bool), "seq": q.get("seq")}


def _public(job: dict) -> dict:
    """The record minus the query coordinates, which only the server uses."""
    out = {k: v for k, v in job.items() if k != "query"}
    out["query"] = {k: v for k, v in job["query"].items()
                    if k not in ("coords", "mask", "tokens")}
    return out


# ---- startup / health -------------------------------------------------
@app.on_event("startup")
def _startup():
    try:
        engine.load()
    except Exception as exc:                      # keep /api/health honest
        engine.error = f"{type(exc).__name__}: {exc}"
        print(f"[startup] FAILED {engine.error}", flush=True)


@app.get("/api/health")
def health():
    return {
        "ready": engine.ready,
        "error": engine.error,
        "databases": [h.info() for h in engine.databases.values()]
                     + [d for d in (remote.info(k) for k in remote.REMOTE) if d],
        "max_concurrent": MAX_CONCURRENT,
        "top_n_cap": TOP_N_CAP,
        "max_upload_mb": MAX_UPLOAD // (1024 * 1024),
        "usalign": bool(os.environ.get("USALIGN")),
    }


@app.get("/api/examples")
def examples():
    index = EXAMPLES / "index.json"
    return json.loads(index.read_text()) if index.exists() else []


# ---- search ------------------------------------------------------------
@app.post("/api/search")
async def search(
    structure: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    databases: Optional[str] = Form(None),
    database: Optional[str] = Form(None),
    top_k: int = Form(50),
    top_n: int = Form(200),
    min_diag: int = Form(2),
    rerank: str = Form("tm_a"),
    tie_seed: Optional[int] = Form(None),
):
    if not engine.ready:
        raise HTTPException(503, engine.error or "server still loading")
    keys = [k.strip() for k in (databases or database or "").split(",") if k.strip()]
    if not keys:
        keys = list(engine.databases)[:1]
    unknown = [k for k in keys if k not in engine.databases
               and not (k in remote.REMOTE and remote.info(k))]
    if unknown:
        raise HTTPException(400, f"unknown database {unknown[0]!r}; choose from "
                                 f"{sorted(engine.databases)}")
    if rerank not in ("tm_a", "sw", "tm"):
        raise HTTPException(400, "rerank must be 'tm_a', 'sw', or 'tm'")
    if not 1 <= top_k <= 1000:
        raise HTTPException(400, "top_k must be between 1 and 1000")
    if not 1 <= top_n <= TOP_N_CAP:
        raise HTTPException(400, f"top_n must be between 1 and {TOP_N_CAP}")
    if not 1 <= min_diag <= 8:
        raise HTTPException(400, "min_diag must be between 1 and 8")

    if structure is not None and structure.filename:
        data, name = await structure.read(), structure.filename
    elif text and text.strip():
        data = text.encode()
        name = "pasted.cif" if "_atom_site" in text else "pasted.pdb"
    else:
        raise HTTPException(400, "upload a structure file or paste one")
    if not data:
        raise HTTPException(400, "empty upload")
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, f"structure larger than "
                                 f"{MAX_UPLOAD // (1024 * 1024)} MB")

    if slots.locked():
        raise HTTPException(503, "all search slots busy; retry shortly")
    async with slots:
        loop = asyncio.get_running_loop()
        try:
            enc = await loop.run_in_executor(pool, engine.tokenise, data, name)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        except Exception as exc:
            raise HTTPException(422, f"could not parse the structure: "
                                     f"{type(exc).__name__}: {exc}")
        results = {}
        for key in keys:
            try:
                if key in remote.REMOTE:
                    results[key] = await loop.run_in_executor(
                        pool, remote.search, key, data, name, top_k,
                        min(top_n, TOP_N_CAP), min_diag, rerank, tie_seed)
                    continue
                results[key] = await loop.run_in_executor(
                    pool, engine.search, key, enc, top_k,
                    min(top_n, TOP_N_CAP), min_diag, rerank, tie_seed)
            except remote.Offline:
                raise HTTPException(502, f"the {key!r} database server is "
                                         f"offline; try again later")
            except remote.RemoteError as exc:
                raise HTTPException(exc.status if exc.status < 500 else 502, exc.detail)
            except FileNotFoundError as exc:      # US-align absent
                raise HTTPException(501, f"TM reranking unavailable on this "
                                         f"server: {exc}")
            except ValueError as exc:             # e.g. query too short for k-mer
                raise HTTPException(422, str(exc))

    job = {
        "id": secrets.token_urlsafe(9),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "params": {"databases": keys, "top_k": top_k, "top_n": top_n,
                   "min_diag": min_diag, "rerank": rerank, "tie_seed": tie_seed},
        "query": {"filename": name, "n_tokens": int(len(enc["tokens"])),
                  "seq": enc.get("seq"),
                  "tokens": np.asarray(enc["tokens"]).tolist(),
                  "coords": np.round(np.asarray(enc["coords"], float), 3).tolist(),
                  "mask": np.asarray(enc["mask"]).astype(bool).tolist()},
        "results": results,
    }
    _save_job(job)
    return JSONResponse(_public(job))


# ---- stored results -------------------------------------------------------
@app.get("/api/result/{job_id}")
def result(job_id: str):
    return JSONResponse(_public(_load_job(job_id)))


def _hit(job: dict, db: str, rank: int) -> dict:
    res = job["results"].get(db)
    if res is None or not 1 <= rank <= len(res["hits"]):
        raise HTTPException(404, "no such hit")
    return res["hits"][rank - 1]


def _remote_hit(job: dict, db: str, rank: int, what: str) -> bytes:
    rid = job["results"][db].get("remote_job")
    if not rid:
        raise HTTPException(410, "the remote result is no longer available")
    try:
        return remote.fetch(db, rid, rank, what)
    except remote.Offline:
        raise HTTPException(502, f"the {db!r} database server is offline")
    except remote.RemoteError as exc:
        raise HTTPException(exc.status if exc.status < 500 else 502, exc.detail)


@app.get("/api/result/{job_id}/{db}/{rank}/alignment")
def hit_alignment(job_id: str, db: str, rank: int):
    """Alignment for a hit beyond RFOLDSEEK_ALIGN_MAX (computed on demand)."""
    job = _load_job(job_id)
    hit = _hit(job, db, rank)
    if db in remote.REMOTE:
        return JSONResponse(json.loads(_remote_hit(job, db, rank, "alignment")))
    if db not in engine.databases:
        raise HTTPException(410, "database no longer served")
    return engine.alignment(engine.databases[db], _query_entry(job), hit["target"])


@app.get("/api/result/{job_id}/{db}/{rank}/structure",
         response_class=PlainTextResponse)
def hit_structure(job_id: str, db: str, rank: int):
    """PDB text: query as chain A, the hit superposed onto it as chain B."""
    job = _load_job(job_id)
    hit = _hit(job, db, rank)
    if db in remote.REMOTE:
        return _remote_hit(job, db, rank, "structure").decode()
    if db not in engine.databases:
        raise HTTPException(410, "database no longer served")
    try:
        return engine.superposition_pdb(db, _query_entry(job), hit["target"])
    except ValueError as exc:
        raise HTTPException(422, str(exc))


TSV_FIELDS = ["database", "rank", "target", "pdb", "chain", "auth_asym", "desc",
              "organism", "method", "resolution", "sw_score", "bits", "evalue",
              "raw_evalue", "structure_score", "tm_a", "aligned", "tm_mean",
              "tm_query", "tm_target", "q_start", "q_end",
              "t_start", "t_end", "q_len", "t_len", "n_aligned", "seq_identity",
              "rmsd", "tm_aln"]


@app.get("/api/result/{job_id}/download")
def download(job_id: str, format: str = "tsv"):
    job = _load_job(job_id)
    if format == "json":
        return Response(json.dumps(_public(job), indent=1),
                        media_type="application/json",
                        headers={"Content-Disposition":
                                 f'attachment; filename="rfoldseek_{job_id}.json"'})
    buf = io.StringIO()
    w = csv.DictWriter(buf, TSV_FIELDS, delimiter="\t", extrasaction="ignore")
    w.writeheader()
    for db, res in job["results"].items():
        for hit in res["hits"]:
            row = dict(hit, **(hit.get("aln") or {}), database=db)
            w.writerow({k: row.get(k, "") for k in TSV_FIELDS})
    return Response(buf.getvalue(), media_type="text/tab-separated-values",
                    headers={"Content-Disposition":
                             f'attachment; filename="rfoldseek_{job_id}.tsv"'})


# ---- pages ------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/tutorial")
def tutorial_page():
    return FileResponse(STATIC / "index.html")


@app.get("/result/{job_id}")
def result_page(job_id: str):
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
