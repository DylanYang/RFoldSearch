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

#### Output format

A comment line naming the query and the database, then one row per hit, ranked
by Smith-Waterman score:

```
# query: chainA  (128 tokens)   database: my_db.pt  (885 chains)
rank  target                    SW_score    bits    E-value    len
   1  4V9F_A                      412.00   118.3    2.1e-30    129
   2  6XU8_B                      287.50    83.1    5.4e-19    121
   3  7QR4_A                      231.00    67.4    8.8e-14    134
```

Values above are illustrative; the column layout is what `rfoldseek search`
prints. `len` is the target's token length. `bits` and `E-value` appear when
E-value parameters are loaded, which they are by default. When nothing
survives the prefilter, a single comment line says so and suggests lowering
`--min_diag`.

#### Important search parameters

| Option       | Category    | Description                                                                        |
| ------------ | ----------- | ---------------------------------------------------------------------------------- |
| `--top_k`    | Output      | Number of top hits to return (default: 10)                                         |
| `--top_n`    | Sensitivity | Candidate-pool depth reranked by Smith-Waterman (default: 200, selected on the validation split) |
| `--min_diag` | Sensitivity | k-mer hits required on a common diagonal to admit a target (default: 2, Foldseek's double-match rule) |
| `--query_id` | Input       | Chain id to look up inside `--query` when it is a database file                    |
| `--model`    | Input       | VQ-VAE tokenizer for the query (default: the bundled one)                          |
| `--matrix`   | Scoring     | Substitution matrix (default: the bundled Structure-BLOSUM)                        |

Raising `--top_n` costs little: the prefilter often returns fewer than `top_n`
candidates for queries from small families, so the deeper pool is free for
exactly those queries that cannot fill it.

### Python API

```python
from rfoldseek import Searcher

s = Searcher("my_db.pt")
hits = s.search_id("my_chain", top_k=10)
# [{'rank': 1, 'target': …, 'sw_score': …, 'bits': …, 'evalue': …}, …]
```

`search_id` forwards keyword arguments to `search`, so `top_n` and `min_diag`
are available there too. `bits` and `evalue` are present when E-value
parameters are loaded, which they are by default.

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

## Third-party components

No third-party source is vendored. `backbone.py`, `tokenizer.py` and
`prefilter_reference.py` are ported from the authors' own
`RFoldseek-paper-code`, with imports adjusted; the ported files say so in their
module docstrings.

The prefilter's double-match rule — admitting a target only on two k-mer hits
sharing a diagonal — follows Foldseek (van Kempen et al., *Nat. Biotechnol.*
2024). Runtime dependencies are `numpy`, `torch`, `numba` and `biotite`, each
under its own license.

## License

Apache License 2.0. See [LICENSE](LICENSE).
