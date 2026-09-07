<h1><a name="rfoldseek"></a><img src=".github/header.PNG" alt="RFoldseek" height="88" /></h1>

RFoldseek enables fast RNA tertiary-structure search with **3Dr**, a learned discrete structural alphabet. Each nucleotide's local geometry is quantised into one of 80 states by a VQ-VAE, converting RNA chains into token strings that are searched with an inverted-index k-mer prefilter. The standard RFoldseek mode reranks prefilter hits by affine-gap token Smith-Waterman alignment and assigns a Karlin-Altschul database E-value to each hit. We also provide RFoldseek-TM, an accuracy-oriented mode in which the same prefilter candidates are reranked by coordinate alignment with US-align using the mean of the two length-normalised TM-scores.

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
  * [Repository layout](#repository-layout)
  * [Third-party components](#third-party-components)
  * [License](#license)

## Installation

```
pip install .
```

Requires Python >= 3.9. Runtime dependencies: `numpy`, `torch`, `numba`, `biotite`.

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
| `--clusters`   | CD-HIT `.clstr` (or id/cluster TSV) grouping near-duplicate chains |

E-values count the search space in sequence clusters, so pass `--clusters` for
any database that holds families more than once. Produce it with

```
cd-hit-est -i targets.fa -o nr80 -c 0.8 -n 5 -aS 0.8 -aL 0.8 -g 1
rfoldseek build --data my.jsonl --output my_db.pt --clusters nr80.clstr
```

Without it every chain counts as its own cluster, which is correct for a
non-redundant database and reports E-values that are too small for a redundant
one — on a 3717-chain database averaging 7.8 chains per cluster, roughly two
orders of magnitude too small at E=0.1.

### Search

Search a chain that is already in the database against it:

```
rfoldseek search --query my_chain --db my_db.pt --top_k 10
```

Or search a chain from a separate query database:

```
rfoldseek search --query queries_db.pt --query_id chainA --db my_db.pt
```

Both commands run the same two-stage pipeline — a k-mer prefilter over the 3Dr tokens, 
then a reranking of the surviving candidates. What differs is how the candidates are reranked, chosen with `--rerank`:

- **`sw`** (default) — affine-gap token Smith-Waterman, with a Karlin-Altschul
  E-value per hit. Fast, self-contained, and what the rest of this README
  benchmarks.
- **`tm`** — RFoldseek-TM: the candidate pool is handed to
  [US-align](https://github.com/pylelab/USalign) and ranked by the mean of its two length-normalised TM-scores. This measures structural similarity directly instead of scoring a token alignment, at a large cost in speed.

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

#### Output format

A comment line naming the query and the database, then one row per hit. Under the default mode, hits are ranked by Smith-Waterman score:

```
# query: chainA  (128 tokens)   database: my_db.pt
rank  target                    SW_score    bits    E-value    len
   1  4V9F_A                      412.00   118.3    2.1e-30    129
   2  6XU8_B                      287.50    83.1    5.4e-19    121
   3  7QR4_A                      231.00    67.4    8.8e-14    134
```

Values above are illustrative; the column layout is what `rfoldseek search`
prints. `len` is the target's token length. `bits` and `E-value` appear when
E-value parameters are loaded, which they are by default. The E-value is the
expected number of database *clusters* reaching this score by chance, and the
bit score is `(S - mu) / (beta ln 2)`, so `E = N_clusters * 2**(-bits)`. When
nothing survives the prefilter, a single comment line says so and suggests
lowering `--min_diag`.

Under `--rerank tm` the comment line names the backend and the score columns change:

```
# query: PDB_00007O81_1_AJ  (74 tokens)   database: db.pt  (885 chains)   rerank: US-align TM
rank  target                    TM_mean  TM_query  TM_target    len
   1  PDB_00008B2L_1_W2          0.7076    0.7143     0.7008     76
   2  PDB_00008KD9_1_M           0.6724    0.6674     0.6773     72
   3  PDB_00008AW3_1_1           0.6688    0.6662     0.6715     73
```

#### Important search parameters

| Option       | Category    | Description                                                                        |
| ------------ | ----------- | ---------------------------------------------------------------------------------- |
| `--top_k`    | Output      | Number of top hits to return (default: 10)                                         |
| `--rerank`   | Scoring     | `sw` (default) or `tm`                                                             |
| `--top_n`    | Sensitivity | Candidate-pool depth passed to the reranker (default: 200) |
| `--min_diag` | Sensitivity | k-mer hits required on a common diagonal to admit a target (default: 2, Foldseek's double-match rule) |
| `--query_id` | Input       | Chain id to look up inside `--query` when it is a database file                    |
| `--model`    | Input       | VQ-VAE tokenizer for the query (default: the bundled one)                          |
| `--matrix`   | Scoring     | Substitution matrix (default: the bundled Structure-BLOSUM)                        |
| `--usalign`  | Scoring     | Path to the `USalign` binary, `--rerank tm` only (default: `$USALIGN`, then `PATH`) |

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
  evalue.py                   cluster-max bit score / E-value
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
