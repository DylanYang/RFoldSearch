<h1><a name="rfoldseek"></a><img src=".github/header.PNG" alt="RFoldseek" height="88" /></h1>

RFoldseek enables fast RNA tertiary-structure search with **3Dr**, a learned discrete structural alphabet. Each nucleotide's local geometry is quantised into one of 80 states by a VQ-VAE, converting RNA chains into token strings that are searched with an inverted-index k-mer prefilter. The default mode aligns candidates by affine-gap token Smith-Waterman, superposes the aligned C4′ atoms, and ranks by the structure score **R<sub>str</sub> = S<sub>SW</sub> × TM<sub>a</sub>**. The reported E-value remains the E-value of S<sub>SW</sub> and counts the search space in CD-HIT-EST sequence clusters. We also provide the legacy token-only `sw` ranking and RFoldseek-TM, an accuracy-oriented mode that reranks candidates by coordinate alignment with US-align.

<p align="center"><img src=".github/fig1.jpeg" width="720" /></p>

## Table of Contents

- [RFoldseek](#rfoldseek)
  * [Installation](#installation)
  * [Hardware and memory requirements](#hardware-and-memory-requirements)
  * [Quick start](#quick-start)
    + [Preprocess structures](#preprocess-structures)
    + [Build a database](#build-a-database)
    + [Search](#search)
    + [E-values](#e-values)
    + [Python API](#python-api)
      - [Tokenize a structure into 3Dr](#tokenize-a-structure-into-3dr)
      - [Search](#search) 
  * [Third-party components](#third-party-components)
  * [License](#license)

## Installation

```
pip install .
```

Requires Python >= 3.9. Runtime dependencies: `numpy`, `torch`, `numba`, `biotite`, `scipy`.

## Hardware and memory requirements

A single core CPU is sufficient. Tokenization uses the bundled VQ-VAE checkpoint on CPU by default 
and moves to CUDA automatically if a GPU is available.

## Quick start

### Preprocess structures

Convert the target `.pdb` / `.cif` files into a backbone-coordinate JSONL:

```
rfoldseek preprocess \
  --pdb_dir target_structures/ \
  --output target_structures.jsonl
```

| Option      | Description                                            |
| ----------- | ------------------------------------------------------ |
| `--max_len` | Skip chains longer than this many nucleotides           |

### Build a database

Cluster the target sequences at 80% identity, then tokenize the JSONL and
write a searchable token database:

```
cd-hit-est -i target_sequences.fasta -o target_nr80 \
  -c 0.8 -n 5 -aS 0.8 -aL 0.8 -g 1

rfoldseek build \
  --data target_structures.jsonl \
  --clusters target_nr80.clstr \
  --output target_db.pt
```

`cd-hit-est` is an external program, neither bundled nor installed by `pip`.
Build it once from source if your platform has no package for it:

```
curl -L -O https://github.com/weizhongli/cdhit/releases/download/V4.8.1/cd-hit-v4.8.1-2019-0228.tar.gz
tar xzf cd-hit-v4.8.1-2019-0228.tar.gz && cd cd-hit-v4.8.1-2019-0228 && make openmp=no
```

| Option         | Description                                                     |
| -------------- | --------------------------------------------------------------- |
| `--min_len`    | Skip chains shorter than this many valid residues. No limit by default; the paper's 885-chain corpus used `--min_len 20` |
| `--checkpoint` | VQ-VAE tokenizer to encode with (default: the bundled one)       |
| `--clusters`   | CD-HIT `.clstr` (or id/cluster TSV) grouping near-duplicate chains |

### Search

Search a chain that is already in the database against it:

```
rfoldseek search --query target_chain_id --db target_db.pt --top_k 10
```

Or search a chain from a separate query database:

```
rfoldseek search --query query_db.pt --query_id query_chain_id --db target_db.pt
```

Both commands run the same two-stage pipeline — a k-mer prefilter over the 3Dr tokens, 
then a reranking of the candidates. What differs is how the candidates are reranked, chosen with `--rerank`:

- **`tm_a`** (default): affine-gap token Smith-Waterman defines the alignment,
  aligned C4′ atoms are superposed, and candidates are ranked by
  **R<sub>str</sub> = S<sub>SW</sub> × TM<sub>a</sub>**. `TM_a` is the mean of
  query- and target-length-normalised TM terms for the fixed token alignment.
  The E-value is still calculated from `S_SW`, not from `R_str`.
- **`sw`**: legacy token-only Smith-Waterman ranking. It can search databases
  without coordinates and reports the same Smith-Waterman E-value.
- **`tm`** (RFoldseek-TM): the candidate pool is handed to
  [US-align](https://github.com/pylelab/USalign) and ranked by the mean of its two length-normalised TM-scores. 
  This measures structural similarity directly instead of scoring a token alignment, at a large cost in speed.

```
rfoldseek search --query target_chain_id --db target_db.pt --rerank tm
```

TM mode is the one part of RFoldseek that does **not** work straight after
installation: it needs a `USalign` binary, which is neither bundled nor a pip
dependency. Build it once from source and either put it on your `PATH`, set
`$USALIGN`, or pass `--usalign`:

```
git clone https://github.com/pylelab/USalign && cd USalign && g++ -O3 -o USalign USalign.cpp
```

#### Output format

A comment line names the query, database, and ranking mode, followed by one
row per hit. Under the default mode, hits are ranked by `R_str`:

```
# query: query_chain_id  (128 tokens)   database: target_db.pt   rerank: R_str = S_SW x TM_a
rank  target                       R_str     TM_a      S_SW    E-value   aligned    len
   1  4V9F_A                       358.44   0.8700    412.00    2.1e-30       124    129
   2  6XU8_B                       221.38   0.7700    287.50    5.4e-19       111    121
   3  7QR4_A                       159.39   0.6900    231.00    8.8e-14       103    134
```

Under `--rerank tm` — RFoldseek-TM — the comment line names the mode and the score columns change:

```
# query: query_chain_id  (74 tokens)   database: target_db.pt   rerank: RFoldseek-TM
rank  target                    TM_mean  TM_query  TM_target    len
   1  PDB_00008B2L_1_W2          0.7076    0.7143     0.7008     76
   2  PDB_00008KD9_1_M           0.6724    0.6674     0.6773     72
   3  PDB_00008AW3_1_1           0.6688    0.6662     0.6715     73
```

#### Important search parameters

| Option       | Category    | Description                                                                        |
| ------------ | ----------- | ---------------------------------------------------------------------------------- |
| `--top_k`    | Output      | Number of top hits to return (default: 10)                                         |
| `--rerank`   | Scoring     | `tm_a` (default), `sw`, or `tm`                                                    |
| `--top_n`    | Sensitivity | Candidate-pool depth passed to the reranker (default: 200) |
| `--min_diag` | Sensitivity | k-mer hits required on a common diagonal to admit a target (default: 2, Foldseek's double-match rule) |
| `--query_id` | Input       | Chain id to look up inside `--query` when it is a database file                    |
| `--model`    | Input       | VQ-VAE tokenizer for the query (default: the bundled one)                          |
| `--matrix`   | Scoring     | Substitution matrix (default: the bundled Structure-BLOSUM)                        |
| `--evalue_params` | Scoring | Database-specific E-value parameter file for this search |
| `--usalign`  | Scoring     | Path to the `USalign` binary, `--rerank tm` only (default: `$USALIGN`, then `PATH`) |

### E-values

RFoldseek reports Smith--Waterman E-values in `tm_a` and `sw` modes when the
database embeds matching parameters or matching parameters are supplied with
`--evalue_params`. It never falls back to a bundled calibration. The
E-value is the expected number of unrelated database clusters (CD-HIT-EST,
80% identity) reaching a hit's token-alignment score; it is not a significance
estimate for the structure score `R_str`. It is computed in two steps:

1. A target-aware Gumbel null models the length-normalized score
   `Y_qt = S_SW / min(L_q, L_t)` for every query--target pair, where `L_t` is
   the target-chain length. Its predictors are
   query length, target-cluster median length, their log-length difference,
   and query-token entropy. The initial value is
   `E0 = sum_c P(max_{t in c} Y_qt >= y)`.
2. A database-specific positive power law,
   `E_cal = a_D * E0**b_D` within the fitted range, calibrates `E0` to the
   observed number of structurally unrelated clusters per query. Above the
   fitted maximum it extends proportionally and is capped at the cluster count.

Gumbel fitting supports two explicitly selected likelihoods: left-censored
and uncensored. The selected mode is stored with the fitted model. Optional
auxiliary queries can contribute to Gumbel fitting with query weights supplied
in the calibration table; they are not used for power calibration. Fitting
uses the complete target-cluster length distribution, and no query--target
length-ratio filter is applied during scoring. Parameters are bound to the
database tokens, clusters, substitution matrix, and gap costs by fingerprints.

#### Fit and export parameters

Fit parameters separately for each target database from a tab-separated file
containing one row per query and eligible target cluster. Each score must be
the maximum `S_SW/min(Lq,Lt)` among that cluster's eligible members, after the
chosen family/clan and structural-background exclusions. The fitter does not
generate pairwise scores or perform the US-align audit; those are inputs to
this fitting step.

Required columns are:

| Column | Meaning |
| --- | --- |
| `query_id` | Calibration-query identifier |
| `role` | `gumbel`, `auxiliary`, or `power` |
| `query_length`, `query_entropy` | Query token length and Shannon entropy |
| `family`, `clan` | Semicolon-separated labels; required for `power` rows |
| `query_weight` | Total Gumbel likelihood weight for this query, repeated on its rows |
| `expansion` | Background expansion factor for this query, repeated on its rows |
| `cluster_id` | Cluster label used in the encoded database |
| `cluster_median_length` | Median token length of that target cluster |
| `normalized_score` | Eligible cluster-maximum normalized SW score |

Use disjoint query IDs for Gumbel fitting (`gumbel` and `auxiliary`) and power
calibration (`power`). Select the Gumbel likelihood explicitly:

```bash
python scripts/fit_target_aware_evalue_params.py --db target_db.pt \
  --observations audited_cluster_scores.tsv --censoring left-censored \
  --gumbel-model target_gumbel_model.json --result target_power_fit.json
```

Use `--censoring none` for an uncensored fit. The fitter writes the fitted
Gumbel model and power-law result. It leaves deployment validation pending
unless both `--structural-audit-complete` and `--validation-accepted` are
provided. Those flags assert that the input background has passed structural
adjudication and that nested family/clan validation has been reviewed; they do
not perform either procedure.

After the audit and validation are complete, export the database-specific
asset:

```bash
python scripts/export_target_aware_evalue_params.py --db target_db.pt \
  --gumbel-model target_gumbel_model.json --result target_power_fit.json \
  --output target_evalue_params.json
```

The exporter reads the fitted censoring mode from the model and checks it
against the refit protocol. It cannot change the fitting mode.

2. Attach the parameters to their matching database:

```bash
rfoldseek attach-calibration --db target_db.pt \
  --clusters target_nr80.clstr --evalue_params target_evalue_params.json \
  --output calibrated_db.pt
```

Every cluster leader within the fitted E0 range must have a US-align audit
result. Structurally related leaders are excluded and replaced by the next
eligible member of the same cluster. Calibration should be assessed by nested
leave-one-Rfam-family-out validation. Since `a_D > 0` and `b_D > 0`, the
correction is monotone and preserves the token-score ordering for a fixed
query.

### Python API

#### Tokenize a structure into 3Dr

The **3Dr** alphabet is built on: `encode_entry` turns one chain into a token per nucleotide, each an integer in `0..79`. 
There is no letter form — 80 states do not fit the Latin alphabet — so the tokens are an integer array, and a chain is a string over that alphabet.

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

A query already in the database needs no tokenizer, since the database holds its tokens:

```python
from rfoldseek import Searcher

s = Searcher("target_db.pt")
hits = s.search_id("target_chain_id", top_k=10)
# [{'rank': 1, 'target': …, 'structure_score': …, 'tm_a': …,
#   'aligned': …, 'sw_score': …, 'normalized_sw_score': …, 'evalue': …}, …]
```

Anything else is searched by handing `search` the tokens from above:

```python
import numpy as np

hits = s.search(np.asarray(enc["tokens"], np.int64), query_entry=enc, top_k=10)
```

`query_entry` is required by the default `rerank="tm_a"` and by `rerank="tm"`,
because both use query coordinates. Plain `rerank="sw"` can leave it out.
`search_id`
forwards its keyword arguments to `search`, so `top_n`, `min_diag`, `rerank`
and `tie_seed` are available from both. `evalue` is present when matching
database-specific E-value parameters are loaded. The target-aware model reports
`normalized_sw_score` rather than a single bit score, because its null varies
with target-chain length.

`Searcher` deliberately takes no tokenizer — searching happens on tokens, not
structures. It does accept the two files that interpret those tokens,
`Searcher(db, matrix_path=…, evalue_params_path=…)`, which is how to search
under a different Structure-BLOSUM matrix or a calibration of your own. On the
command line, the corresponding options are `--model`, `--matrix`,
`--checkpoint`, and `--evalue_params`.

## Third-party components

No third-party source is vendored. Runtime dependencies are `numpy`, `torch`, `numba` and `biotite`, each
under its own license. `--rerank tm` calls US-align (Zhang et al., *Nat. Methods* 2022) as an external
program, and `--clusters` consumes the output of CD-HIT-EST (Fu et al.,
*Bioinformatics* 2012). Neither source is vendored and neither is installed by
`pip`; users supply their own binaries.

## License

Apache License 2.0. See [LICENSE](LICENSE).
