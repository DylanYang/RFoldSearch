<h1><a name="rfoldseek"></a><img src=".github/header.PNG" alt="RFoldseek" height="88" /></h1>

RFoldseek enables fast RNA tertiary-structure search with **3Dr**, a learned discrete structural alphabet. Each nucleotide's local geometry is quantised into one of 80 states by a VQ-VAE, converting RNA chains into token strings that are searched with an inverted-index k-mer prefilter. The standard RFoldseek mode reranks prefilter hits by affine-gap token Smith-Waterman alignment and assigns a Karlin-Altschul database E-value to each hit, counting the search space in CD-HIT-EST sequence clusters rather than in chains so that a family stored many times poses one question of the database. We also provide RFoldseek-TM, an accuracy-oriented mode in which the same prefilter candidates are reranked by coordinate alignment with US-align using the mean of the two length-normalised TM-scores.

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
      - [Tokenize a structure into 3Dr](#tokenize-a-structure-into-3dr)
  * [Third-party components](#third-party-components)
  * [License](#license)

## Installation

```
pip install .
```

Requires Python >= 3.9. Runtime dependencies: `numpy`, `torch`, `numba`, `biotite`.

## Hardware and memory requirements

A single core CPU is sufficient. Tokenization uses the bundled VQ-VAE checkpoint on CPU by default 
and moves to CUDA automatically if a GPU is available.

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

Cluster the target sequences at 80% identity, then tokenize the JSONL and
write a searchable token database:

```
cd-hit-est -i targets.fa -o nr80 -c 0.8 -n 5 -aS 0.8 -aL 0.8 -g 1
rfoldseek build --data my.jsonl --output my_db.pt --clusters nr80.clstr
```

`cd-hit-est` is an external program, neither bundled nor installed by `pip`.
Build it once from source if your platform has no package for it:

```
curl -L -O https://github.com/weizhongli/cdhit/releases/download/V4.8.1/cd-hit-v4.8.1-2019-0228.tar.gz
tar xzf cd-hit-v4.8.1-2019-0228.tar.gz && cd cd-hit-v4.8.1-2019-0228 && make openmp=no
```

| Option         | Description                                                     |
| -------------- | --------------------------------------------------------------- |
| `--min_len`    | Skip chains shorter than this (default: 20, as used in the paper) |
| `--checkpoint` | VQ-VAE tokenizer to encode with (default: the bundled one)       |
| `--clusters`   | CD-HIT `.clstr` (or id/cluster TSV) grouping near-duplicate chains |

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
then a reranking of the candidates. What differs is how the candidates are reranked, chosen with `--rerank`:

- **`sw`** (default) — affine-gap token Smith-Waterman, with a Karlin-Altschul
  E-value per hit. Fast, self-contained.
- **`tm`** — RFoldseek-TM: the candidate pool is handed to
  [US-align](https://github.com/pylelab/USalign) and ranked by the mean of its two length-normalised TM-scores. 
  This measures structural similarity directly instead of scoring a token alignment, at a large cost in speed.

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

#### Tokenize a structure into 3Dr

The alphabet is the primitive everything else is built on: `encode_entry`
turns one chain into a token per nucleotide, each an integer in `0..79`. There
is no letter form — 80 states do not fit the Latin alphabet — so the tokens are
an integer array, and a chain is a string over that alphabet.

```python
import torch
from rfoldseek import structure_to_entry
from rfoldseek.search import asset
from rfoldseek.tokenizer import encode_entry, load_model

model, ckpt, n_codes, _ = load_model(asset("desc_vqvae_best.pt"))   # or your own
enc = encode_entry(model, structure_to_entry("query.cif"), torch.device("cpu"),
                   ckpt.get("norm_mean"), ckpt.get("norm_std"))

enc["tokens"]     # int16, one 3Dr state per resolved nucleotide, e.g. [22, 53, 53, 44, …]
enc["n_valid"]    # how many nucleotides were resolved well enough to tokenize
enc["coords"]     # (n_valid, 5, 3) backbone coordinates, kept for US-align
```

`encode_entry` returns `None` when a chain has no resolved nucleotides at all,
or when its edge arrays disagree with its mask; there is no minimum length here,
`--min_len` is applied by `rfoldseek build`. `structure_to_entry` reads both
`.cif` and `.pdb`, and `load_model` puts the model on whatever device you pass,
so `torch.device("cuda")` runs the tokenizer on a GPU.

#### Search

A query already in the database needs no tokenizer, since the database holds
its tokens:

```python
from rfoldseek import Searcher

s = Searcher("my_db.pt")
hits = s.search_id("my_chain", top_k=10)
# [{'rank': 1, 'target': …, 'sw_score': …, 'bits': …, 'evalue': …}, …]
```

Anything else is searched by handing `search` the tokens from above:

```python
import numpy as np

hits = s.search(np.asarray(enc["tokens"], np.int64), query_entry=enc, top_k=10)
```

`query_entry` is read only by `rerank="tm"`, which needs the query coordinates
for US-align; plain Smith-Waterman search can leave it out. `search_id`
forwards its keyword arguments to `search`, so `top_n`, `min_diag`, `rerank`
and `tie_seed` are available from both. `bits` and `evalue` are present when
E-value parameters are loaded, which they are by default.

`Searcher` deliberately takes no tokenizer — searching happens on tokens, not
structures. It does accept the two files that interpret those tokens,
`Searcher(db, matrix_path=…, evalue_params_path=…)`, which is how to search
under a different Structure-BLOSUM matrix or a calibration of your own. On the
command line the same choices are `rfoldseek search --model / --matrix` and
`rfoldseek build --checkpoint`; all default to the bundled files.

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
program, and `--clusters` consumes the output of CD-HIT-EST (Fu et al.,
*Bioinformatics* 2012). Neither source is vendored and neither is installed by
`pip`; users supply their own binaries.

## License

Apache License 2.0. See [LICENSE](LICENSE).
