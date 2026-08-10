# RFoldseek

Fast RNA tertiary-structure search with **3Dr**, a learned discrete structural
alphabet. Each nucleotide's local tertiary geometry is quantised into one of 80
states by a VQ-VAE; chains become token strings searched with an inverted-index
k-mer prefilter (similar-k-mer expansion under a Structure-BLOSUM matrix,
Foldseek's double-match rule) and reranked by affine-gap token Smith-Waterman,
with a Karlin-Altschul database E-value per hit.

The deployed tokenizer, Structure-BLOSUM matrix and E-value calibration are
bundled — search works straight after installation.

## Install

```bash
pip install .
```

Dependencies: numpy, torch, numba, biotite (Python >= 3.9).

## Quickstart

```bash
# 1. structures -> backbone JSONL  (a directory of .pdb / .cif files)
rfoldseek preprocess --pdb_dir my_structures/ --output my.jsonl

# 2. JSONL -> searchable token database
rfoldseek build --data my.jsonl --output my_db.pt

# 3. search a chain already in the database against it
rfoldseek search --query my_chain --db my_db.pt --top_k 10

# ... or a chain from a separate query database
rfoldseek search --query queries_db.pt --query_id chainA --db my_db.pt
```

Python API:

```python
from rfoldseek import Searcher

s = Searcher("my_db.pt")
hits = s.search_id("my_chain", top_k=10)   # [{'target':…, 'sw_score':…, 'bits':…, 'evalue':…}, …]
```

## Implementation note

The prefilter in this release computes per-target best-diagonal hit counts in
three linear compiled passes (counting-sort by target, diagonal scatter,
per-segment bincount) instead of materialising and sorting every posting.
**Its output is identical to the reference implementation** — candidate set,
ranking and scores — which ships in `rfoldseek/prefilter_reference.py` and is
enforced by `tests/test_prefilter_equivalence.py`; it was additionally verified
query-for-query against the paper pipeline at database sizes of 1,000 / 10,000
/ 83,156 (20/20 identical at each).

Single-core cost against an 83,156-structure predicted-RNA database
(mean / median per query): **1.12 s / 0.037 s**, with a 3.3 s one-off index
build. The reference implementation costs 49 s / 0.78 s on the same benchmark.

## Reproducing the paper's search results

`tests/test_paper_db.py` checks this package against the analysis repository's
bundled 885-chain database:

```bash
RFOLDSEEK_PAPER_DB=/path/to/RFoldseek-paper-code/files/models/db.pt pytest tests/
```

The top hits and Smith-Waterman scores match the paper pipeline exactly.

## Layout

```
rfoldseek/
  backbone.py, edges.py     structure -> backbone coords + pair/stack edges
  descriptors/, vqvae.py    17-D descriptor and the VQ-VAE tokenizer
  tokenizer.py, database.py checkpoint loading, encoding, database build/load
  prefilter.py              release prefilter (compiled O(n) kernel)
  prefilter_reference.py    reference implementation (semantic ground truth)
  smith_waterman.py         numba token Smith-Waterman
  evalue.py                 Karlin-Altschul bit score / E-value
  search.py, cli.py         pipeline and command line
  assets/                   deployed tokenizer, matrix, E-value params
```

Core model, descriptor and alignment code is ported unchanged from
`RFoldseek-paper-code`; provenance is noted at the top of each ported file.
