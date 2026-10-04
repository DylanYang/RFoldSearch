# RFoldseek search server

Upload or paste an RNA structure, get the ranked hits from one or more
resident RFoldseek databases, with a per-hit alignment (sequence and 3Dr
lines) and a 3D superposition. Every search is stored under a short id, so
`/result/<id>` is a shareable page that survives a restart;
`/result/<id>#<db>/<rank>` opens one hit. FastAPI backend, one static page
(`static/index.html` + `app.js`, 3Dmol.js from a CDN), no build step.

## Why not the Foldseek server's stack

`search.foldseek.com` runs [MMseqs2-App](https://github.com/soedinglab/MMseqs2-App),
whose `APP` setting selects `mmseqs` or `foldseek`, and which starts one
CLI process per job. That design works because those binaries memory-map
their index and are useful a millisecond after exec.

RFoldseek is not that shape. A database is a torch archive that has to be
deserialised, and the k-mer inverted index is then built in process —
about 20 s for half a million chains, on top of several GB resident. Per
request that is unusable, so this server loads each database once at
startup and serves every query from the resident objects. Adapting
MMseqs2-App would have meant writing a Go adapter plus a resident Python
worker behind it; the whole of this server is smaller than that adapter.

## Run it

```
pip install -r requirements.txt
export RFOLDSEEK_DBS="rnasolo=RNAsolo experimental (885 chains):/path/to/db.pt"
uvicorn app.main:app --port 8000
```

That binds to `127.0.0.1`, reachable only from the same machine. To let
other machines on the network reach it, bind to every interface and use
the host's own address:

```
uvicorn app.main:app --host 0.0.0.0 --port 8000     # http://<host-ip>:8000
```

On macOS the firewall may prompt the first time, or silently refuse
connections; allow incoming connections for the Python binary. Note that
there is no authentication: anyone who can route to the port can upload
a structure and start a search, and a search holds a core for its
duration. That is fine on a lab network and is not enough for the open
internet, which would want at least a rate limit and a proxy in front.

Or with the bundled compose file, which mounts `./databases` read-only:

```
mkdir -p databases && cp /path/to/db.pt databases/
docker compose up --build
```

`GET /api/health` reports whether the databases finished loading; a large
one takes minutes, which is why the healthcheck has a long start period.

## Configuration

| Variable | Meaning |
|---|---|
| `RFOLDSEEK_DBS` | Pipe-separated `key=Label:path` entries. The separator is `|`, not a comma, because labels carry commas; the path is taken from the last colon, so paths may contain colons too. Every entry is loaded at startup and appears in the database menu. |
| `RFOLDSEEK_MODEL` | Tokeniser checkpoint. Defaults to the one bundled with the package. |
| `RFOLDSEEK_MAX_CONCURRENT` | Searches allowed to run at once (default: cores − 1). Over the limit the API returns 503 rather than queueing. |
| `RFOLDSEEK_TOP_N_CAP` | Largest candidate depth a request may ask for (default 2000). |
| `RFOLDSEEK_MAX_UPLOAD_MB` | Upload size limit (default 32). |
| `USALIGN` | Path to the `USalign` binary. Without it `rerank=tm` returns 501 and the rest of the server works normally. |
| `RFOLDSEEK_JOBS_DIR` | Where search records are stored as JSON (default `./jobs`). Nothing expires them yet. |
| `RFOLDSEEK_ALIGN_MAX` | Hits per database that get an alignment and superposition up front (default 100); later hits are aligned on demand. |

### Display sidecar

The result page shows sequences and PDB annotation, which a database `.pt`
does not store. `scripts/build_metadata.py` writes `<db>.meta.json` beside
the database: the token-aligned sequence of every chain (from the
preprocessed JSONL the database was built from) and, with `--rcsb`, the
RCSB description, organism, method and resolution, plus the date the
database was built. Chains are matched to RCSB RNA entities by label chain
id or author chain id, whichever names an RNA whose sequence matches. A
database without a sidecar still works: hits show their ids and the
target line of the alignment shows placeholders.

Run **one** worker process. Each would load its own copy of every
database; concurrency is meant to come from `RFOLDSEEK_MAX_CONCURRENT`
threads inside a single process.

## API

```
GET  /api/health
POST /api/search      multipart form
       structure   file        .pdb / .cif, one RNA chain   } one of
       text        string      pasted PDB or mmCIF          }
       databases   string      comma-separated keys from /api/health
       top_k       int = 50    hits returned per database
       top_n       int = 200   candidate depth; 200 is the deployed default
       min_diag    int = 2     k-mer hits on a common diagonal; 1 is more sensitive
       rerank      tm_a | sw | tm   structure score (default), token SW, or US-align
       tie_seed    int         break score ties by a hash of the chain id
GET  /api/result/{id}                          the stored record
GET  /api/result/{id}/{db}/{rank}/alignment    alignment of one hit
GET  /api/result/{id}/{db}/{rank}/structure    PDB: query (A) + superposed hit (B)
GET  /api/result/{id}/download?format=tsv|json
GET  /api/examples
```

```
curl -sF structure=@query.cif -F databases=pdb-rna,rnasolo -F top_k=5 \
     localhost:8000/api/search | python -m json.tool
```

`tm_a` hits carry `structure_score`, `tm_a`, `aligned`, `sw_score`, `bits`,
and `evalue`; the E-value is derived from `sw_score`, not `structure_score`.
`sw` hits carry `sw_score`, `bits` and `evalue`; `tm` hits carry
`tm_mean`, `tm_query` and `tm_target` and no E-value, because TM-scores
have no Karlin-Altschul statistics. Hits within `RFOLDSEEK_ALIGN_MAX` carry
`aln`: aligned ranges, sequence and 3Dr alignment lines, sequence and 3Dr
identity, and the C4' RMSD and TM-score of the superposition over the
aligned pairs. The alignment is recomputed with a full traceback
(`app/align.py`) and its score equals `sw_score` exactly. Interactive docs
at `/api/docs`.

## Sizing

Resident memory is dominated by the database. The 885-chain experimental
set is tens of MB; the 531,888-chain NuFold set is several GB and needs
minutes to load and index at startup. A search holds one core for its
duration — a median NuFold query is tens of milliseconds, the slowest
a few seconds — so `RFOLDSEEK_MAX_CONCURRENT` is what bounds tail
latency under load, not the queue depth.

`rerank=tm` is far more expensive than the default: it runs up to `top_n`
coordinate alignments, which at the published settings costs seconds to
tens of seconds per query. Offer it only with `RFOLDSEEK_MAX_CONCURRENT`
set deliberately low, or not at all.

## Status

Demo running on the lab 2080 (LAN only) with PDB-RNA, RNAsolo 2.0 and
NuFold. Before a public deployment: E-values are not yet calibrated for
these databases (the page says so), the NuFold database predates the
PR #13 tokeniser and has no sequence sidecar, and the server has no
authentication or rate limit.

## Verified

End-to-end against the 885-chain experimental database on 12 Sep 2026:
the 4,444-token 23S rRNA chain `6HRM_1_1` returns itself at rank 1 with
E = 0 followed by other large-subunit rRNAs, in 5.2 s; a 24-token query
returns in 0.5 s. Malformed uploads, unknown database keys, out-of-range
parameters and a missing `USalign` binary each return a useful message
rather than a traceback.

One packaging note found while testing: on Python 3.9, `pip install
rfoldseek` resolves `biotite` to 0.40, which is built against numpy 1.x,
while `torch` pulls numpy 2. The import then fails inside biotite. The
`python:3.11-slim` base image in the Dockerfile avoids this; a 3.9
install needs `numpy<2` pinned. The package declares `>=3.9`, so this is
worth either pinning or raising in `pyproject.toml`.
