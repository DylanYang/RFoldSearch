<h1><a name="rfoldseek"></a><img src=".github/header.PNG" alt="RFoldseek" height="88" /></h1>

RFoldseek enables fast RNA tertiary-structure search with **3Dr**, a learned discrete structural alphabet. Each nucleotide is described by a 17-D vector: the rigid-body transform to its nearest non-sequential spatial neighbour (distance, direction and rotation) and its local context (sequence separations to its neighbour, base-pair and stacking partners, cis/trans base-pair orientation, and contact density within 8 and 12 Å). A VQ-VAE quantises this vector into one of 80 states, converting RNA chains into token strings (**B**). Targets sharing at least two token k-mers on a common diagonal with the query pass the prefilter, and the top N are aligned by affine-gap token Smith-Waterman, which gives the alignment score S_SW and the nucleotide correspondence. The aligned C4′ atoms are then superposed without changing the alignment, and hits are ranked by the structure score R_str = S_SW × aTM, where aTM is the alignment TM-score; each hit carries a cluster-maximum Gumbel E-value computed from S_SW (**A**). RFoldseek-TM, an accuracy-oriented mode, instead reranks the same top N candidates with US-align by the mean of the two length-normalised TM-scores.

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

Requires Python >= 3.9. Runtime dependencies: `numpy`, `torch`, `numba`, `scipy` (needed by numba's linear algebra in the superposition), `biotite`.

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

- **`atm`** (default): affine-gap token Smith-Waterman gives the alignment score S_SW and,
  in the same dynamic-programming pass, the alignment. The C4′ atoms of the
  aligned nucleotides are then superposed without re-optimising the alignment,
  and the alignment TM-score aTM = (T_Q + T_T) / 2 is taken, where T_X sums
  1 / (1 + (r_i / d0(L_X))²) over the aligned pairs, divides by the query
  (X = Q) or target (X = T) length, and d0(L) = max(0.5, 0.6 √(L − 0.5) − 2.5) Å.
  Normalising by both lengths penalises partial coverage. The superposition is
  fitted by iterative Kabsch refinement from two starts, all aligned pairs and
  the highest-scoring gapless segment of the alignment; each iteration refits
  on the pairs closer than min(d0(L_Q), d0(L_T)) + 1 Å, for up to five
  iterations or until that set stops changing, and the best aTM over both starts
  and all iterations is kept. Hits are ranked by the structure score R_str = S_SW × aTM, so
  a hit must match locally *and* in the arrangement of its aligned segments;
  equal structure scores are ordered as in the prefilter. The E-value is that
  of S_SW.
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

A comment line naming the query and the database, then one row per hit. Under the default mode, hits are ranked by the structure score:

```
# query: PDB_00001FFZ_1_A  (496 tokens)   database: db.pt  (885 chains, 882 clusters)
rank  target                   structure_score    aTM    E-value  aligned    len
   1  PDB_00001FFK_1_0                  118.59  0.583    8.3e-32      489   2706
   2  PDB_00005AN9_1_N                  117.27  0.651    1.1e-27      468   1162
   3  PDB_00004V9F_1_0                   94.04  0.559    1.2e-25      485   2803
```

- `aTM`: alignment TM-score, 0 to 1: the TM-score of the token alignment after superposition, averaged over normalisation by the query and by the target length.
- `structure_score`: the ranking score R_str = S_SW × aTM, where S_SW is the token Smith-Waterman score.
- `E-value`: expected number of hits at least this good by chance, computed from S_SW alone; it therefore need not increase down the list.
- `aligned`: number of aligned nucleotide pairs.
- `len`: target length.

The Python interface also returns `sw_score` (S_SW) and `bits`, the length- and composition-normalised form of S_SW from which the E-value is computed.

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
| `--rerank`   | Scoring     | `atm` (default) or `tm`                                                            |
| `--top_n`    | Sensitivity | Candidate-pool depth passed to the reranker (default: 200) |
| `--min_diag` | Sensitivity | k-mer hits required on a common diagonal to admit a target (default: 2, Foldseek's double-match rule) |
| `--query_id` | Input       | Chain id to look up inside `--query` when it is a database file                    |
| `--model`    | Input       | VQ-VAE tokenizer for the query (default: the bundled one)                          |
| `--matrix`   | Scoring     | Substitution matrix (default: the bundled Structure-BLOSUM)                        |
| `--evalue_params` | Scoring | Database-specific E-value parameter file for this search |
| `--usalign`  | Scoring     | Path to the `USalign` binary, `--rerank tm` only (default: `$USALIGN`, then `PATH`) |

### E-values

RFoldseek reports E-values for token Smith--Waterman searches. The initial
E-value uses a Gumbel distribution fitted to per-query cluster-maximum
background scores; the search space is the number of CD-HIT-EST clusters.
Both experimental and predicted databases can store their own linear ridge
model of Gumbel location and log scale using log query length and 3Dr-token
entropy. These predictors are clipped to the fitted training range before
evaluation; queries outside that range emit a warning because the boundary
estimate is not independently validated there. Each database
also has its own empirical mapping from
the initial E-value to the family-weighted mean number of unrelated
background clusters per query. E-values are computed from S_SW alone, while
hits are ranked by the structure score R_str = S_SW × aTM, so E-values need not
increase down the list.

The bundled experimental-database parameter asset contains the all-family
query model and the empirical mapping used for the current manuscript figure.
The latter is marked provisional because a later fixed-point audit found
previously unreviewed backfilled cluster leaders; RFoldseek therefore reports
the matching initial parametric value but does not deploy that empirical map.
The asset must be regenerated by the workflow below before final release.
For any other database (including NuFold), attach its own parameter JSON.
RFoldseek binds fitted parameters to a fingerprint of the token database,
cluster membership, substitution matrix and gap costs, rather than accepting
a database merely because it has the same number of clusters.

The numbered workflow below fits the deployed linear query-conditional model
directly from each database. For every query in a pre-defined, balanced query
panel it searches the full target database, takes one eligible maximum per
sequence cluster, fits a per-query Gumbel distribution, and then fits
`[1, z(log query length), z(3Dr-token entropy)]` to `mu` and `log(beta)` by
ridge regression. No query--target length-ratio filter is used. The historical
length-grid model remains available only as an explicit compatibility fallback.

The files in the calibration pipeline are:

| File | Produced by |
| ---- | ----------- |
| `target_structures.jsonl` | `rfoldseek preprocess` on the target structures |
| `target_nr80.clstr` | CD-HIT-EST on `target_sequences.fasta` |
| `target_db.pt` | `rfoldseek build` from the target JSONL and cluster file |
| `labels.tsv` | Step 0 below |
| `query_panel.tsv` | Balanced calibration-query selection fixed before fitting |
| `adjudication.tsv` | US-align review of candidate structural relatives |
| `target_clustermax_params.json` | Step 1 below |
| `calibration_queries.pt` | Step 2 below |
| `target_background.tsv` | Step 3 below |
| `target_background.queries.tsv` | Step 3 query denominators and audit provenance |
| `target_evalue_params.json` | Step 4 below |

`target_sequences.fasta` and `calibration_sequences.fasta` are maintained by
the web-server ingestion pipeline and use the same chain identifiers as the
corresponding structure JSONL files. End users of the web server do not create
any of the calibration files below.

0. Annotate the target and independent calibration sequences against the same
Rfam release. The server obtains `Rfam.cm` and `Rfam.clanin` from Rfam, runs
`cmpress Rfam.cm` once, and then runs Infernal with Rfam gathering thresholds:

```bash
cmscan --cut_ga --rfam --fmt 2 --clanin Rfam.clanin \
  --tblout target_rfam.tbl Rfam.cm target_sequences.fasta

cmscan --cut_ga --rfam --fmt 2 --clanin Rfam.clanin \
  --tblout calibration_rfam.tbl Rfam.cm calibration_sequences.fasta

python scripts/build_rfam_labels.py \
  --fasta target_sequences.fasta \
  --fasta calibration_sequences.fasta \
  --tblout target_rfam.tbl \
  --tblout calibration_rfam.tbl \
  --output labels.tsv
```

`labels.tsv` is therefore required server-side calibration metadata, not a
user-supplied search input. It contains `chain_id`, `family`, and `clan`; the
last two fields are empty when Rfam reports no included hit. The conversion
script keeps the best included hit for each chain. The calibration scripts
also accept the synonymous columns `entry_id`/`id`, `rfam_acc`, and
`clan_acc`.

1. Fit the initial query-conditional cluster-max Gumbel model for
`target_db.pt`. The query panel must cover the intended query-length and token-
entropy domain and should be balanced by Rfam family rather than sampled in
proportion to family size. Structural exclusions are applied before the
cluster maximum is selected, so an excluded cluster leader is replaced by the
best remaining eligible member:

```bash
python scripts/rebuild_evalue_params.py \
  --db target_db.pt \
  --labels labels.tsv \
  --query-panel query_panel.tsv \
  --related-pairs adjudication.tsv \
  --query-fits-output per_query_gumbel_fits.tsv \
  --output target_clustermax_params.json
```

Omitting `--query-panel` uses every chain and is practical only for small
databases. `--model length-grid` reproduces the legacy fallback; it is not the
model used for the shipped experimental or predicted databases. The command
requires a US-align result for every retained cluster leader with
$E_0\leq100$. When coverage is incomplete it writes
`target_clustermax_params.missing-audit.tsv` and does not write deployable
parameters. Run US-align for those pairs, append both related and unrelated
results to `adjudication.tsv`, and rerun. A related leader is removed and the
next eligible member is backfilled, so this loop must continue until the
command succeeds with no missing leaders. `--allow-unreviewed` exists only to
write explicitly provisional exploratory parameters; the search code refuses
to report E-values from them.

2. Encode an independent calibration panel:

```bash
rfoldseek preprocess \
  --pdb_dir calibration_structures/ \
  --output calibration_queries.jsonl

rfoldseek build \
  --data calibration_queries.jsonl \
  --output calibration_queries.pt
```

3. Generate one observation per eligible query--cluster pair:

```bash
python scripts/build_empirical_evalue_background.py \
  --db target_db.pt \
  --queries calibration_queries.pt \
  --params target_clustermax_params.json \
  --labels labels.tsv \
  --related-pairs adjudication.tsv \
  --background-scope annotated \
  --audit-ceiling 100 \
  --output target_background.tsv
```

The example above is the experimental-database workflow: it evaluates only
Rfam-annotated background clusters so that the observed counts can later be
expanded by the per-query factor $N_D/B_q$. For the predicted database use
`--background-scope all` and do not apply that expansion. In both cases the
generator performs a full-database token Smith--Waterman scan, excludes
the query's own cluster and chains from the same annotated family or clan, and
retains only the highest-scoring eligible member of each CD-HIT cluster. Pairs
marked as structurally related by the US-align audit, or whose mean query- and
target-normalized TM-score is at least 0.45, are excluded before the cluster
maximum is selected; the next eligible member
therefore backfills that cluster. The adjudication TSV may
provide `tm_query_normalized` and `tm_target_normalized` or their mean as
`tm_mean`. As in Step 1, every retained leader at $E_0\leq100$ must have an
explicit related or unrelated result. Missing leaders are written to
`target_background.missing-audit.tsv`; adjudicate and rerun until the command
succeeds. Only then are `target_background.tsv` and the one-row-per-query
`target_background.queries.tsv` published.

4. Fit the empirical mapping. The script reads the CD-HIT cluster assignments
stored in `target_db.pt`, so no cluster count is entered manually:

```bash
python scripts/fit_empirical_evalue_calibration.py \
  --params target_clustermax_params.json \
  --background target_background.tsv \
  --query-panel target_background.queries.tsv \
  --background-scaling annotated-to-full \
  --raw-evalue-ceiling 100 \
  --db target_db.pt \
  --output target_evalue_params.json
```

For the predicted database, use `--background-scaling none` instead. The query
metadata records
the background scope, $N_D$, $B_q$, and structural-audit ceiling. The fitting
command rejects `annotated-to-full` unless the metadata came from
`--background-scope annotated`; it calculates the family/query-balanced
weight $[|\mathcal F|\,|Q_f|]^{-1}N_D/B_q$ itself. Do not precompute or combine
this factor with an explicit row weight.

Model assessment must keep whole Rfam families out of each test fold. In every
leave-one-family-out fold, refit both the query-conditional Gumbel regression
and the empirical mapping using only the remaining families. After that
validation is complete, rerun steps 1--4 with all calibration families to
produce the deployment JSON; do not deploy a fold-specific parameter file.

`target_background.tsv` contains the required `raw_evalue` and `query_id`
columns. It may also contain an explicit
`weight` column; otherwise `query_id` is required and equal-query weighting is
used. If a `family` column is also present, families are weighted equally.
Rows marked `structurally_related=true` are excluded from the empirical null.
For a bounded predicted-database fit, supply `--query-panel panel.tsv` with
every calibration `query_id` and `family`, including queries with no candidate
below the fitted threshold, together with `--raw-evalue-ceiling 100`. This
retains the correct family and query denominators. Experimental calibrations
use the generated query metadata and `--background-scaling annotated-to-full`;
predicted full-background calibrations use
`--background-scaling none`.
The fitted map is stored thinned: adjacent knots are merged into segments that
report the largest calibrated value they contain, so a stored E-value is never
smaller than the full map's and at most 1% larger (`--max-relative-step`,
default 0.01).

5. Search the target database with its matching calibrated parameters:

```bash
rfoldseek search \
  --query query_db.pt \
  --query_id query_chain_id \
  --db target_db.pt \
  --evalue_params target_evalue_params.json
```

With empirical calibration enabled, `evalue` is the calibrated value and
`raw_evalue` retains the initial Gumbel cluster-max value in the Python API.
Below the smallest observed calibration background E-value, calibrated values
share a finite-sample floor; raw E-values, bit scores, and alignment scores
still distinguish these strong hits, but a smaller calibrated E-value is not
claimed without supporting background observations. The bit score remains
derived from the initial Gumbel model. RFoldseek checks the logical database
and scoring-context fingerprints as well as the recorded cluster count, and
refuses parameters fitted for another target even when both targets happen to
contain the same number of clusters. A predicted-database mapping fitted only
for initial E-values up to a recorded ceiling reports the raw E-value beyond
that range.

To deploy an already encoded database with matching calibration parameters,
assign its CD-HIT clusters and attach the parameter JSON without re-encoding:

```bash
rfoldseek attach-calibration \
  --db existing_db.pt \
  --clusters target_nr80.clstr \
  --evalue_params target_evalue_params.json \
  --output calibrated_db.pt
```

When CD-HIT used positional FASTA identifiers such as `t0`, `t1`, ... for
database entries in their original order, add `--index-prefix t`. Every entry
must have a cluster assignment, and the cluster count must match the parameter
file. `rfoldseek build` also accepts `--evalue_params` alongside `--clusters`.

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
# [{'rank': 1, 'target': …, 'structure_score': …, 'atm': …, 'aligned': …,
#   'sw_score': …, 'bits': …, 'evalue': …, 'length': …}, …]
```

Anything else is searched by handing `search` the tokens from above:

```python
import numpy as np

hits = s.search(np.asarray(enc["tokens"], np.int64), query_entry=enc, top_k=10)
```

`query_entry` supplies the query coordinates, which both rerankers need; a
chain already in the database brings its own. `search_id`
forwards its keyword arguments to `search`, so `top_n`, `min_diag`, `rerank`
and `tie_seed` are available from both. `bits` and `evalue` are present when
E-value parameters are loaded, which they are by default.

`Searcher` deliberately takes no tokenizer — searching happens on tokens, not
structures. It does accept the two files that interpret those tokens,
`Searcher(db, matrix_path=…, evalue_params_path=…)`, which is how to search
under a different Structure-BLOSUM matrix or a calibration of your own. On the
command line, the corresponding options are `--model`, `--matrix`,
`--checkpoint`, and `--evalue_params`.

## Third-party components

No third-party source is vendored. Runtime dependencies are `numpy`, `torch`, `numba`, `scipy` and `biotite`, each
under its own license. `--rerank tm` calls US-align (Zhang et al., *Nat. Methods* 2022) as an external
program, and `--clusters` consumes the output of CD-HIT-EST (Fu et al.,
*Bioinformatics* 2012).

## License

Apache License 2.0. See [LICENSE](LICENSE).
