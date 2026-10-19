# Ruyi's and Hongxiang's E-value schemes: leave-one-family-out test (2026-10-18/19)

Schemes under test:
- **Ruyi:** raw SW score + target-aware null + one global power-law
  correction, fitted on a background cleaned with US-align. Summary in Gitea
  issue #2, comment 243.
- **Hongxiang** (main `c25c6a5`, `rfoldsearch/experimental_evalue.py`): the
  same target-aware null followed by a family-balanced empirical monotone map
  (`fit_empirical`, ceiling 100). Added 2026-10-19 as `hx_emp_*`; summary in
  Gitea issue #2.

## Contents

| Path | What |
|---|---|
| `scripts/lofo.py` | LOFO test of five E-value methods (below); writes a JSON report |
| `scripts/list_bg_pairs.py` | lists every background pair and those not yet in `adjudication.tsv` |
| `scripts/audit_missing.py` | runs US-align on the listed pairs and appends them to `adjudication.tsv` |
| `results/rnasolo_lofo.json`, `.log` | RNAsolo, background fully audited |
| `results/rnasolo_lofo_partial_audit.json` | RNAsolo, original adjudication only (high-scoring pairs) |
| `results/pdbrna_lofo.json`, `.log` | PDB-RNA, original adjudication only |
| `results/{rnasolo,pdbrna}_lofo_v3.json`, `.log` | rerun with all seven methods incl. `hx_emp_*` (RNAsolo fully audited) |
| `data/<db>/labels.tsv` | Rfam family and clan per chain |
| `data/<db>/adjudication.tsv` | US-align TM-scores per (query, target) pair |

`data/rnasolo/adjudication.tsv` is complete: all 94,320 background pairs, 1,321
related. `data/pdbrna/adjudication.tsv` covers only the pairs audited during the
original calibration, because the full set is 1.83 M pairs. On RNAsolo the
complete and partial audits give almost the same result.

## Methods compared

| Key | Score | Null model | Power law |
|---|---|---|---|
| `old_deployed` | raw SW | cluster-max Gumbel (deployed params) | deployed, not refitted |
| `old_lofo` | raw SW | cluster-max Gumbel (deployed params) | refitted per fold |
| `ta_norm_left` | SW / min(Lq, Lt) | target-aware, left-censored | per fold (PR #25 as merged) |
| `ta_raw_left` | raw SW | target-aware, left-censored | per fold (**Ruyi's scheme**) |
| `ta_raw_none` | raw SW | target-aware, uncensored | per fold |
| `hx_emp_left` | raw SW | Hongxiang's target-aware fitter, left-censored | empirical map, per fold (**Hongxiang's scheme**) |
| `hx_emp_none` | raw SW | as above, uncensored | empirical map, per fold |

`hx_emp_*` fit the null and the map on all training families of a fold, as
Hongxiang's fitter does, on the same annotated, US-align-cleaned background.

- **Target-aware fitter:** `scripts/fit_target_aware_evalue_params.py` from PR #25, commit `98f650c`.
  - Covariates: log query length, log cluster median length, their absolute difference, and token entropy.
  - Left-censoring at the 95th percentile, 3 bins.
  - Power law: `fit_max` 100.
  - In each fold, alternating training families fit the null and the power law.
- **Background:** annotated targets of another Rfam family and clan, minus pairs with mean TM ≥ 0.45. Counts are scaled by N_D / B_q.
- **True positives:** same-family clusters.
- **Report fields:** for each E threshold t:
  - `fp_per_query`: background clusters per query at E ≤ t, averaged over families
  - `ratio`: `fp_per_query` divided by t
  - `families_over_3x`: families whose `fp_per_query` exceeds 3t
  - `tp_fraction`: share of true-positive clusters found

## Running

Inputs: the database and queries from the PR #14 calibration run. These files
are not in git:

| File | sha256 |
|---|---|
| `rnasolo/rnasolo_calibrated.pt` | `9ed421ce2e89…997a` |
| `rnasolo/calibration_queries.pt` | `00a3af7fa135…a9c6` |
| `pdbrna/pdbrna_calibrated.pt` | `1de0d976b908…eb6f` |
| `pdbrna/calibration_queries.pt` | `07a23f6caff4…fce6` |

Put each `.pt` file in the same directory as that database's `labels.tsv`.
The installed `rfoldsearch` package must be the pre-PR #25 version, for
example main `cbf966d`.

```bash
git worktree add /tmp/pr25 98f650c          # PR #25 code: evalue.py and the fitter
python lofo.py /tmp/pr25 out.json data/rnasolo rnasolo_calibrated.pt data/rnasolo/adjudication.tsv
python lofo.py /tmp/pr25 out.json data/pdbrna  pdbrna_calibrated.pt  data/pdbrna/adjudication.tsv
```

The `hx_emp_*` methods load `rfoldsearch.experimental_evalue`; set
`HX_EVALUE=/path/to/experimental_evalue.py` to use a file from another checkout.
An optional sixth argument names a pickle that caches the scored rows
(default `<out>.rows.pkl`), so reruns skip the Smith-Waterman pass.

RNAsolo takes about 40 minutes and PDB-RNA about 1.5 hours, on 1 core.

`audit_missing.py` imports `rfoldseek.usalign.score_pool` from the PR #14 code
(branch `query-conditional-evalues`), at the path set in its `sys.path` line.
To audit the background completely:

```bash
python list_bg_pairs.py <dir> <db.pt> missing.tsv
python audit_missing.py --db <dir>/<db.pt> --queries <dir>/calibration_queries.pt \
    --missing missing.tsv --adjudication <dir>/adjudication.tsv --expand 1 --workers 24
```
