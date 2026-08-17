# <img src=".github/logo.PNG" height="40" align="left" style="margin-right:12px" /> RFoldseek

RFoldseek enables fast RNA tertiary-structure search with **3Dr**, a learned
discrete structural alphabet. Each nucleotide's local geometry is quantised
into one of 80 states by a VQ-VAE; chains become token strings searched with
an inverted-index k-mer prefilter and reranked by affine-gap token Smith-Waterman, with a Karlin-Altschul database E-value per hit. The deployed tokenizer, Structure-BLOSUM matrix and E-value calibration are bundled — search works straight after installation.

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

### Build a database

Tokenize the JSONL and write a searchable token database:

```
rfoldseek build --data my.jsonl --output my_db.pt
```

### Search

Search a chain that is already in the database against it:

```
rfoldseek search --query my_chain --db my_db.pt --top_k 10
```

Or search a chain from a separate query database:

```
rfoldseek search --query queries_db.pt --query_id chainA --db my_db.pt
```

#### Important search parameters

| Option       | Category    | Description                                                       |
| ------------ | ----------- | ----------------------------------------------------------------- |
| `--top_k`    | Output      | Number of top hits to return (default: 10)                        |
| `--evalue`   | Sensitivity | Report matches at or below this E-value                           |
| `--query_id` | Input       | Chain id to look up inside `--query` when it is a database file   |

### Python API

```python
from rfoldseek import Searcher

s = Searcher("my_db.pt")
hits = s.search_id("my_chain", top_k=10)
# [{'target': …, 'sw_score': …, 'bits': …, 'evalue': …}, …]
```

## Main modules

- `rfoldseek preprocess` — parse `.pdb` / `.cif` structures into a backbone JSONL
- `rfoldseek build` — tokenize a JSONL into a searchable token database
- `rfoldseek search` — prefilter + Smith-Waterman search against a database

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
  evalue.py                   Karlin-Altschul bit score / E-value
  search.py, cli.py           pipeline and command line
  assets/                     deployed tokenizer, matrix, E-value params
```

