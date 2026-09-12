"""RFoldseek search server.

A structure is uploaded, tokenised with the deployed VQ-VAE, and searched
against a resident database; the hit table comes back as JSON and is
rendered by static/index.html. Databases are declared in RFOLDSEEK_DBS as
comma-separated key=label:path entries, so the same image serves the
885-chain experimental set or a half-million predicted one.

Searches run on a small thread pool and are bounded by a semaphore rather
than a queue with unbounded admission: a search holds a core for its
duration, so accepting more concurrent work than there are cores makes
every request slower without serving anyone sooner. Over the limit the
server says so with 503 instead of silently queueing.
"""
from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .engine import Engine

HERE = Path(__file__).resolve().parent
STATIC = HERE.parent / "static"

MAX_UPLOAD = int(os.environ.get("RFOLDSEEK_MAX_UPLOAD_MB", "32")) * 1024 * 1024
MAX_CONCURRENT = int(os.environ.get("RFOLDSEEK_MAX_CONCURRENT",
                                    str(max(1, (os.cpu_count() or 2) - 1))))
TOP_N_CAP = int(os.environ.get("RFOLDSEEK_TOP_N_CAP", "2000"))

app = FastAPI(title="RFoldseek search", docs_url="/api/docs",
              openapi_url="/api/openapi.json")
engine = Engine(spec=os.environ.get("RFOLDSEEK_DBS", ""),
                model=os.environ.get("RFOLDSEEK_MODEL") or None)
pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT,
                          thread_name_prefix="rfsearch")
slots = asyncio.Semaphore(MAX_CONCURRENT)


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
        "databases": [{"key": h.key, "label": h.label, "n_chains": h.n_chains}
                      for h in engine.databases.values()],
        "max_concurrent": MAX_CONCURRENT,
        "top_n_cap": TOP_N_CAP,
        "max_upload_mb": MAX_UPLOAD // (1024 * 1024),
    }


@app.post("/api/search")
async def search(
    structure: UploadFile = File(...),
    database: str = Form(...),
    top_k: int = Form(10),
    top_n: int = Form(200),
    min_diag: int = Form(2),
    rerank: str = Form("sw"),
    tie_seed: Optional[int] = Form(None),
):
    if not engine.ready:
        raise HTTPException(503, engine.error or "server still loading")
    if database not in engine.databases:
        raise HTTPException(400, f"unknown database {database!r}; choose from "
                                 f"{sorted(engine.databases)}")
    if rerank not in ("sw", "tm"):
        raise HTTPException(400, "rerank must be 'sw' or 'tm'")
    if not 1 <= top_k <= 1000:
        raise HTTPException(400, "top_k must be between 1 and 1000")
    if not 1 <= top_n <= TOP_N_CAP:
        raise HTTPException(400, f"top_n must be between 1 and {TOP_N_CAP}")
    if not 1 <= min_diag <= 8:
        raise HTTPException(400, "min_diag must be between 1 and 8")

    data = await structure.read()
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
            enc = await loop.run_in_executor(
                pool, engine.tokenise, data, structure.filename or "query.pdb")
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        except Exception as exc:
            raise HTTPException(422, f"could not parse the structure: "
                                     f"{type(exc).__name__}: {exc}")
        try:
            result = await loop.run_in_executor(
                pool, engine.search, database, enc, top_k, min(top_n, TOP_N_CAP),
                min_diag, rerank, tie_seed)
        except FileNotFoundError as exc:          # US-align absent
            raise HTTPException(501, f"TM reranking unavailable on this "
                                     f"server: {exc}")
    result["filename"] = structure.filename
    return JSONResponse(result)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
