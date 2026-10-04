<h1><a name="rfoldseek"></a><img src=".github/header.PNG" alt="RFoldseek" height="88" /></h1>

RFoldseek enables fast RNA tertiary-structure search with **3Dr**, a learned discrete structural alphabet. A VQ-VAE quantises each nucleotide's local geometry into one of 80 states (**B**), turning RNA chains into token strings. A k-mer prefilter selects candidates, which are aligned by token Smith-Waterman and ranked by the structure score S_SW × aTM, the alignment score times the TM-score of the aligned C4′ atoms (**A**). Each hit carries a cluster-level E-value. RFoldseek-TM, an accuracy-oriented mode, reranks the same candidates with US-align.

<p align="center"><img src=".github/fig1.jpeg" width="720" /></p>

## Table of Contents

- [Installation](#installation)
- [Quick start](#quick-start)
  * [Build a database](#build-a-database)
  * [Search](#search)
  * [Output](#output)
  * [Search parameters](#search-parameters)
- [E-values](#e-values)
  * [Calibrating a database](#calibrating-a-database)
- [Python API](#python-api)
- [Third-party components](#third-party-components)
- [License](#license)

## Installation

```
pip install .
```

Requires Python >= 3.9 with `numpy`, `torch`, `numba`, `scipy` and `biotite`, installed by `pip`. A single CPU core is enough; tokenization uses a GPU automatically when one is available.

Two external programs are optional and not installed by `pip`:

- [CD-HIT-EST](https://github.com/weizhongli/cdhit), to cluster a database (recommended, needed for E-values).
- [US-align](https://github.com/pylelab/USalign), for `--rerank tm`. Put `USalign` on your `PATH`, set `$USALIGN`, or pass `--usalign`.

## Quick start

### Build a database

```
rfoldseek preprocess --pdb_dir target_structures/ --output target_structures.jsonl

cd-hit-est -i target_sequences.fasta -o target_nr80 -c 0.8 -n 5 -aS 0.8 -aL 0.8 -g 1

rfoldseek build --data target_structures.jsonl --clusters target_nr80.clstr --output target_db.pt
```

`preprocess` reads `.pdb` and `.cif` files. `build` options: `--clusters` (CD-HIT `.clstr` or an id/cluster TSV), `--min_len` (skip shorter chains; no limit by default, the paper used 20), `--max_len` on `preprocess`, and `--checkpoint` (tokenizer; default: bundled).

### Search

```
rfoldseek search --query target_chain_id --db target_db.pt                              # a chain in the database
rfoldseek search --query query_db.pt --query_id query_chain_id --db target_db.pt        # a chain from another database
rfoldseek search --query target_chain_id --db target_db.pt --rerank tm                  # RFoldseek-TM
```

Both modes share the k-mer prefilter and differ in how the candidates are reranked:

- **`atm`** (default): token Smith-Waterman gives the score S_SW and the alignment; the aligned C4′ atoms are superposed without changing the alignment, and hits are ranked by S_SW × aTM, where aTM is the alignment TM-score averaged over normalisation by the query and by the target length. A hit must therefore match locally *and* in the arrangement of its aligned segments.
- **`tm`** (RFoldseek-TM): candidates are ranked by the mean of US-align's two length-normalised TM-scores. More accurate, much slower, and without E-values.

### Output

```
# query: PDB_00001FFZ_1_A  (496 tokens)   database: db.pt  (885 chains, 882 clusters)
rank  target                   structure_score    aTM    E-value  aligned    len
   1  PDB_00001FFK_1_0                  118.59  0.583    8.3e-32      489   2706
   2  PDB_00005AN9_1_N                  117.27  0.651    1.1e-27      468   1162
```

- `structure_score`: the ranking score, S_SW × aTM.
- `aTM`: alignment TM-score, 0 to 1.
- `E-value`: expected number of database clusters scoring this well by chance. It is computed from S_SW alone, so it need not increase down the list.
- `aligned`, `len`: aligned nucleotide pairs, target length.

Under `--rerank tm` the score columns are `TM_mean`, `TM_query` and `TM_target`.

### Search parameters

| Option | Description |
| ------ | ----------- |
| `--top_k` | Hits to report (default 10) |
| `--top_n` | Candidates passed from the prefilter to the reranker (default 200) |
| `--min_diag` | k-mer hits required on a common diagonal (default 2, Foldseek's double-match rule) |
| `--rerank` | `atm` (default) or `tm` |
| `--evalue_params` | E-value parameters fitted for this database |
| `--matrix`, `--model` | Substitution matrix and query tokenizer (default: bundled) |
| `--usalign` | Path to `USalign`, `--rerank tm` only |

## E-values

The E-value is the expected number of unrelated database **clusters** (CD-HIT, 80% identity) reaching a hit's score, so a family held hundreds of times counts once. It is computed in two steps:

1. A Gumbel null of per-cluster maximum scores, whose location and scale are predicted from the query's length and 3Dr-token entropy, gives an initial E-value.
2. A database-specific empirical map converts it into the observed number of structurally unrelated clusters per query (US-align TM-score < 0.45).

Parameters belong to one database. RFoldseek checks a fingerprint of the tokens, clusters, substitution matrix and gap costs, and does not apply parameters fitted for another database:

- **Your own database**: without its own parameters, searches run normally but report no E-value. Fit them as below.
- **The bundled 885-chain experimental database**: the packaged parameters report the initial E-value only; the empirical map is provisional and will be refitted before release.

### Calibrating a database

Calibration is done once per database, offline. All commands are in `scripts/`.

| Input | Contents |
| ----- | -------- |
| `target_db.pt` | Clustered database (`rfoldseek build --clusters`) |
| `labels.tsv` | `chain_id`, `family`, `clan` (Rfam; empty if unannotated) |
| `query_panel.tsv` | Calibration queries, one column `query_id`, fixed before fitting and balanced by Rfam family |
| `calibration_queries.pt` | The panel's structures as a database; chains taken from the target must keep their ids |
| `adjudication.tsv` | US-align results: `query_id`, `target_id`, `tm_mean` (or both normalised TM-scores) |

Labels can come from Infernal (`cmscan --cut_ga --rfam` against one Rfam release, converted by `build_rfam_labels.py`) or from the database's own Rfam annotation.

**1. Gumbel model.** For each panel query, the full database is scored, one maximum is taken per cluster, and a Gumbel is fitted; μ and log β are then regressed on z(log length) and z(token entropy). Same-family, same-clan and structurally related targets are excluded before the maximum, so the next member of the cluster takes its place.

```bash
python scripts/rebuild_evalue_params.py --db target_db.pt --labels labels.tsv \
  --query-panel query_panel.tsv --related-pairs adjudication.tsv --audit-ceiling 100 \
  --output target_clustermax_params.json
```

**2. Background.** One observation per query and cluster.

```bash
python scripts/build_empirical_evalue_background.py --db target_db.pt \
  --queries calibration_queries.pt --params target_clustermax_params.json \
  --labels labels.tsv --related-pairs adjudication.tsv --audit-ceiling 100 \
  --background-scope all --output target_background.tsv
```

**3. Empirical map.**

```bash
python scripts/fit_empirical_evalue_calibration.py --params target_clustermax_params.json \
  --background target_background.tsv --query-panel target_background.queries.tsv \
  --background-scaling none --raw-evalue-ceiling 100 --db target_db.pt \
  --output target_evalue_params.json
```

**4. Deploy**, then search with the result:

```bash
rfoldseek attach-calibration --db target_db.pt --clusters target_nr80.clstr \
  --evalue_params target_evalue_params.json --output calibrated_db.pt
```

Rules that matter:

- **Audit to completion.** Steps 1 and 2 need a US-align result for every cluster leader with initial E-value ≤ 100. If any is missing they stop and write `*.missing-audit.tsv`. Run US-align on those pairs, append the results to `adjudication.tsv` and rerun; a related leader is replaced by the next member, which may need its own audit, so repeat until both steps succeed.
- **Predicted vs experimental databases.** Predicted databases use `--background-scope all` and `--background-scaling none`, as above. Experimental databases use `--background-scope annotated` and `--background-scaling annotated-to-full`, which counts only Rfam-annotated background clusters and rescales them to the whole database.
- **Validate by family.** Assess the calibration leave-one-Rfam-family-out, refitting steps 1–3 without the held-out family in each fold; deploy the fit on all families.
- The stored map is thinned into conservative segments: a stored E-value is never smaller than the full map's and at most 1% larger (`--max-relative-step`).

When CD-HIT was given positional ids (`t0`, `t1`, …), add `--index-prefix t` to `attach-calibration`.

## Python API

Tokenize a structure (tokens are integers 0–79, one per resolved nucleotide):

```python
import torch
from rfoldseek import structure_to_entry
from rfoldseek.search import asset
from rfoldseek.tokenizer import encode_entry, load_model

model, ckpt, _, _ = load_model(asset("desc_vqvae_best.pt"))
enc = encode_entry(model, structure_to_entry("query.cif"), torch.device("cpu"),
                   ckpt.get("norm_mean"), ckpt.get("norm_std"))
enc["tokens"], enc["coords"]
```

Search:

```python
import numpy as np
from rfoldseek import Searcher

s = Searcher("target_db.pt")                     # evalue_params_path=..., matrix_path=... optional
hits = s.search_id("target_chain_id", top_k=10)  # a chain in the database
hits = s.search(np.asarray(enc["tokens"], np.int64), query_entry=enc, top_k=10)
# [{'rank': 1, 'target': …, 'structure_score': …, 'atm': …, 'aligned': …,
#   'sw_score': …, 'bits': …, 'evalue': …, 'length': …}, …]
```

`query_entry` supplies the query coordinates needed for reranking. `top_n`, `min_diag`, `rerank` and `tie_seed` are accepted by both calls. `evalue` (and `bits`) are present when E-value parameters apply to the database; with an empirical map, `raw_evalue` holds the initial value.

## Third-party components

No third-party source is vendored. Runtime dependencies are `numpy`, `torch`, `numba`, `scipy` and `biotite`, each under its own license. `--rerank tm` calls US-align (Zhang et al., *Nat. Methods* 2022), and `--clusters` uses the output of CD-HIT-EST (Fu et al., *Bioinformatics* 2012).

## License

Apache License 2.0. See [LICENSE](LICENSE).
