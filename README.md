# <img src=".github/logo.PNG" height="40" align="left" style="margin-right:12px" /> RFoldseek

RFoldseek enables fast RNA tertiary-structure search with **3Dr**, a learned
discrete structural alphabet. Each nucleotide's local geometry is quantised into
one of 80 states by a VQ-VAE; chains become token strings searched with an
inverted-index k-mer prefilter and reranked by affine-gap token Smith-Waterman,
with a Karlin-Altschul database E-value per hit. The deployed tokenizer,
Structure-BLOSUM matrix and E-value calibration are bundled — search works
straight after installation.

<p align="center"><img src=".github/fig1.jpeg" width="720" /></p>

## Table of Contents

- [RFoldseek](#rfoldseek)
  * [Installation](#installation)
  * [Hardware and memory requirements](#hardware-and-memory-requirements)
  * [Quick start](#quick-start)
    + [Preprocess structures](#preprocess-structures)
    + [Build a database](#build-a-database)
    + [Search](#search)
    + [Python API](#python-api)
  * [Main modules](#main-modules)
  * [Prefilter implementation](#prefilter-implementation)
  * [Reproducing the paper's search results](#reproducing-the-papers-search-results)
  * [Repository layout](#repository-layout)
  * [Third-party components](#third-party-components)
  * [License](#license)

## Installation

```
pip install .
```

Requires Python >= 3.9. Runtime dependencies: `numpy`, `torch`, `numba`,
`biotite`.

## Hardware and memory requirements

CPU-only; a single core is sufficient. Tokenization uses the bundled VQ-VAE
checkpoint on CPU by default and moves to CUDA automatically if a GPU is
visible to `torch`. Memory scales with database size — an 83,156-chain
predicted-RNA database fits comfortably in a few GB of RAM.

## Quick start

### Preprocess structures

Convert a directory of `.pdb` / `.cif` files into a backbone-coordinate JSONL:

```
rfoldseek preprocess --pdb_dir my_structures/ --output my.jsonl
```

| Option      | Description                                            |
| ----------- | ------------------------------------------------------ |
| `--max_len` | Skip chains longer than this many nucleotides           |

### Build a database

Tokenize the JSONL and write a searchable token database:

```
rfoldseek build --data my.jsonl --output my_db.pt
```

| Option         | Description                                                     |
| -------------- | --------------------------------------------------------------- |
| `--min_len`    | Skip chains shorter than this (default: 20, as used in the paper) |
| `--checkpoint` | VQ-VAE tokenizer to encode with (default: the bundled one)       |

### Search

Search a chain that is already in the database against it:

```
rfoldseek search --query my_chain --db my_db.pt --top_k 10
```

Or search a chain from a separate query database:

```
rfoldseek search --query queries_db.pt --query_id chainA --db my_db.pt
```

Both commands run the same two-stage pipeline — a k-mer prefilter over the 3Dr
tokens, then a reranking of the surviving candidates. What differs is how the
candidates are reranked, chosen with `--rerank`:

- **`sw`** (default) — affine-gap token Smith-Waterman, with a Karlin-Altschul
  E-value per hit. Fast, self-contained, and what the rest of this README
  benchmarks.
- **`tm`** — **RFoldseek-TM**: the candidate pool is handed to
  [US-align](https://github.com/pylelab/USalign) and ranked by the mean of its
  two length-normalised TM-scores. This measures structural similarity directly
  instead of scoring a token alignment, at a large cost in speed. It is the
  Foldseek-style coordinate-reranking variant used in the paper's
  remote-homologue analysis.

```
rfoldseek search --query my_chain --db my_db.pt --rerank tm
```

TM mode is the one part of RFoldseek that does **not** work straight after
installation: it needs a `USalign` binary, which is neither bundled nor a pip
dependency. Build it once from source and either put it on your `PATH`, set
`$USALIGN`, or pass `--usalign`:

```
git clone https://github.com/pylelab/USalign && cd USalign && g++ -O3 -o USalign USalign.cpp
```

No structure files are needed at search time. US-align uses C3' as the
representative atom for RNA, and the token database already stores C3' for
every residue, so RFoldseek writes the coordinates it needs straight out of the
`.pt` file. Existing databases work unchanged, and the original PDB/mmCIF
inputs do not have to still exist. This is verified rather than assumed:
scoring a C3'-only file and a file carrying all five stored backbone atoms
returns TM-scores identical to five decimal places.

#### Output format

A comment line naming the query and the database, then one row per hit. Under
the default `--rerank sw`, hits are ranked by Smith-Waterman score:

```
# query: chainA  (128 tokens)   database: my_db.pt  (885 chains)
rank  target                    SW_score    bits    E-value    len
   1  4V9F_A                      412.00   118.3    2.1e-30    129
   2  6XU8_B                      287.50    83.1    5.4e-19    121
   3  7QR4_A                      231.00    67.4    8.8e-14    134
```

Under `--rerank tm` the comment line names the backend and the score columns
change:

```
# query: PDB_00007O81_1_AJ  (74 tokens)   database: db.pt  (885 chains)   rerank: US-align TM
rank  target                    TM_mean  TM_query  TM_target    len
   1  PDB_00008B2L_1_W2          0.7076    0.7143     0.7008     76
   2  PDB_00008KD9_1_M           0.6724    0.6674     0.6773     72
   3  PDB_00008AW3_1_1           0.6688    0.6662     0.6715     73
```

`len` is the target's token length. The SW table's `bits` and `E-value` appear
when E-value parameters are loaded, which they are by default. The TM table has
no such columns at all: the Karlin-Altschul parameters are calibrated against
Smith-Waterman scores and say nothing about TM-scores, so RFoldseek prints
nothing rather than a meaningless number. When nothing survives the prefilter,
a single comment line says so and suggests lowering `--min_diag`.

#### Important search parameters

| Option       | Category    | Description                                                                        |
| ------------ | ----------- | ---------------------------------------------------------------------------------- |
| `--top_k`    | Output      | Number of top hits to return (default: 10)                                         |
| `--rerank`   | Scoring     | `sw` (default) or `tm`                                                             |
| `--top_n`    | Sensitivity | Candidate-pool depth passed to the reranker (default: 200, selected on the validation split) |
| `--min_diag` | Sensitivity | k-mer hits required on a common diagonal to admit a target (default: 2, Foldseek's double-match rule) |
| `--query_id` | Input       | Chain id to look up inside `--query` when it is a database file                    |
| `--model`    | Input       | VQ-VAE tokenizer for the query (default: the bundled one)                          |
| `--matrix`   | Scoring     | Substitution matrix (default: the bundled Structure-BLOSUM)                        |
| `--usalign`  | Scoring     | Path to the `USalign` binary, `--rerank tm` only (default: `$USALIGN`, then `PATH`) |

#### Choosing the parameters

**`--rerank` is the choice that matters.** Everything else is a smaller
adjustment on top of it.

Under `sw`, reranking is negligible next to the prefilter, so the search is
effectively as fast as the prefilter: a mean of 1.12 s per query against an
83,156-chain database. Under `tm`, reranking dominates completely. The cost of
one US-align superposition grows steeply with chain length, and `--top_n`
of them run per query — measured single-core, on 2,700-3,100 nt queries with a
full pool of 200 candidates, one query takes **20-23 minutes**. Short chains
are cheap by comparison: a 74 nt query against the same 200-candidate pool
finishes in well under a second. Cost is driven by the length of the chains
involved, not by database size.

Two consequences worth planning around. Use `sw` for interactive work and for
anything that needs an E-value; reach for `tm` when ranking quality on a
specific query matters more than latency. And if you are running `tm` over a
whole query set, expect a batch job rather than an interactive one, and budget
by the length of your longest chains rather than by the number of queries.

**`--top_n` behaves very differently under the two backends.** Under `sw`,
raising it costs almost nothing: the prefilter often returns fewer than `top_n`
candidates for queries from small families, so the deeper pool is free for
exactly those queries that cannot fill it. Under `tm`, `--top_n` is the direct
multiplier on runtime — halving it roughly halves the query's cost. The default
of 200 was selected on the validation split for `sw` reranking; it is kept
under `tm` so the two backends rank the same candidate pool and stay
comparable, which is what the paper's RFoldseek-TM does. Lower it when you want
`tm` results on long chains without waiting.

**`--min_diag` trades sensitivity against pool size** and applies to both
backends, since it acts before reranking. The default of 2 is Foldseek's
double-match rule: a target is admitted only when two k-mer hits share a
diagonal. Lowering it to 1 admits far more candidates — worth trying when a
query returns nothing, but under `tm` it also multiplies the number of
superpositions.

**`--top_k` is output-only.** It slices the ranking that has already been
computed, so it has no effect on runtime or on the ranking itself.

**`--model` and `--matrix` default to the bundled deployed artifacts** and
should be left alone unless you are reproducing an ablation. The tokenizer in
particular must match the one the database was built with, or the token strings
are not comparable.

### Python API

```python
from rfoldseek import Searcher

s = Searcher("my_db.pt")
hits = s.search_id("my_chain", top_k=10)
# [{'rank': 1, 'target': …, 'sw_score': …, 'bits': …, 'evalue': …}, …]
```

`search_id` forwards keyword arguments to `search`, so `top_n`, `min_diag` and
`rerank` are available there too. `bits` and `evalue` are present when E-value
parameters are loaded, which they are by default.

## Main modules

- `rfoldseek preprocess` — parse `.pdb` / `.cif` structures into a backbone JSONL
- `rfoldseek build` — tokenize a JSONL into a searchable token database
- `rfoldseek search` — prefilter + Smith-Waterman (or US-align TM) search against a database

## Prefilter implementation

The prefilter in this release computes per-target best-diagonal hit counts in
three linear compiled passes (counting-sort by target, diagonal scatter,
per-segment bincount) instead of materialising and sorting every posting.
**Its output is identical to the reference implementation** — candidate set,
ranking and scores — which ships in `rfoldseek/prefilter_reference.py` and is
enforced by `tests/test_prefilter_equivalence.py`; it was additionally
verified query-for-query against the paper pipeline at database sizes of
1,000 / 10,000 / 83,156 (20/20 identical at each).

Single-core cost against an 83,156-structure predicted-RNA database
(mean / median per query): **1.12 s / 0.037 s**, with a 3.3 s one-off index
build. The reference implementation costs 49 s / 0.78 s on the same
benchmark.

## Reproducing the paper's search results

`tests/test_paper_db.py` checks this package against the analysis repository's
bundled 885-chain database:

```
RFOLDSEEK_PAPER_DB=/path/to/RFoldseek-paper-code/files/models/db.pt pytest tests/
```

The top hits and Smith-Waterman scores match the paper pipeline exactly.

## Repository layout

```
rfoldseek/
  backbone.py, edges.py       structure -> backbone coords + pair/stack edges
  descriptors/, vqvae.py      17-D descriptor and the VQ-VAE tokenizer
  tokenizer.py, database.py   checkpoint loading, encoding, database build/load
  prefilter.py                release prefilter (compiled O(n) kernel)
  prefilter_reference.py      reference implementation (semantic ground truth)
  smith_waterman.py           numba token Smith-Waterman
  usalign.py                  US-align TM-score reranking (RFoldseek-TM)
  evalue.py                   Karlin-Altschul bit score / E-value
  search.py, cli.py           pipeline and command line
  assets/                     deployed tokenizer, matrix, E-value params
```

## Third-party components

No third-party source is vendored. `backbone.py`, `tokenizer.py` and
`prefilter_reference.py` are ported from the authors' own
`RFoldseek-paper-code`, with imports adjusted; the ported files say so in their
module docstrings.

The prefilter's double-match rule — admitting a target only on two k-mer hits
sharing a diagonal — follows Foldseek (van Kempen et al., *Nat. Biotechnol.*
2024). Runtime dependencies are `numpy`, `torch`, `numba` and `biotite`, each
under its own license.

`--rerank tm` calls US-align (Zhang et al., *Nat. Methods* 2022) as an external
program. Its source is not vendored and it is not installed by `pip`; users
supply their own binary.

## License

Apache License 2.0. See [LICENSE](LICENSE).
