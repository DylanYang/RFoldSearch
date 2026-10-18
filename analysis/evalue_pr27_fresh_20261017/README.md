# Independent experimental E-value comparison

This folder shares the exploratory 885-chain comparison with collaborators. The report and figures are snapshots, not a validated calibration or deployable parameter set. In particular, the US-align exclusions were **not** fed back into Gumbel/power fitting. No database-specific coefficients are installed in `rfoldsearch` by this change.

## Inputs and replay

Run from the repository root. Supply the same 885 mmCIF files (one chain ID per filename), Rfam `labels.tsv` (columns `id`, `family`, `clan`), a working `cd-hit-est`, `USalign`, and the repository's VQ-VAE checkpoint and substitution matrix. The report and `score_manifest.json` record input hashes; the raw CIFs, labels, score arrays, database, and large audit tables are not uploaded. Use a fresh output directory; do not mix runs.

```bash
python -m scripts.fresh_experimental_evalue_preprocess --source /path/to/cifs --out /path/to/run --workers 4
cd-hit-est -i /path/to/run/sequences.fasta -o /path/to/run/clusters_nr80.fasta -c 0.8 -n 5 -aS 0.8 -aL 0.8 -g 1
rfoldsearch build --data /path/to/run/structures.jsonl --clusters /path/to/run/clusters_nr80.fasta.clstr --min_len 20 --output /path/to/run/db_nr80.pt
python -m scripts.fresh_experimental_evalue_scores --db /path/to/run/db_nr80.pt --labels /path/to/labels.tsv --matrix rfoldsearch/assets/sub_matrix.pt --out /path/to/run --workers 4
python -m scripts.fresh_experimental_evalue_compare --run /path/to/run --labels /path/to/labels.tsv --exploratory-unaudited
python -m scripts.check_fresh_experimental_evalue_run --run /path/to/run
python -m scripts.fresh_experimental_evalue_usalign --run /path/to/run --usalign /path/to/USalign --fast
python -m scripts.recheck_fresh_usalign_boundary --run /path/to/run --usalign /path/to/USalign
python -m scripts.summarize_fresh_evalue_usalign --run /path/to/run
python -m scripts.write_fresh_experimental_evalue_report --run /path/to/run --labels /path/to/labels.tsv
```

The comparison script also accepts `--structural-audit /path/to/complete-member-audit.tsv` for a cleaned fitting run; that is **not** the mode used for this snapshot. A complete member-level audit can be generated independently with `python -m scripts.audit_clean_experimental_background --source /path/to/run --run /path/to/audited-run --labels /path/to/labels.tsv --usalign /path/to/USalign`; it does not treat missing alignment output as unrelated. `--uncensored` selects the uncensored alternative; the default fit is left-censored. Audit every candidate, including long queries, before interpreting structural-truth panels. US-align is the slowest stage. The scripts require the repository's Python dependencies plus `matplotlib` for report plots; run `python -m pytest -q tests/test_fresh_experimental_evalue_compare.py` for focused tests.

The follow-up full-range check is `python -m scripts.recheck_fresh_experimental_evalue_range --run /path/to/run --labels /path/to/labels.tsv --audit /path/to/structural_members.tsv --output /path/to/full_range_recheck.json`. It replays every held-out background count from the frozen score matrices and fold models, checks all 27 grid levels, and reports conservative bounds using the available member-level audit. It does **not** refit the null distribution. See `full_range_recheck.md` for the current method-comparison interval E=0.01–10 and the separate E=100 extrapolation diagnostic.

The bundled `cv_folds.tsv`, `score_manifest.json`, `qa_checks.json`, and PNGs help compare a local replay against this snapshot. The generated `comparison.json` contains fitted research coefficients and is intentionally not bundled; collaborators should fit their own database. The original run used Python 3.12.14, NumPy 2.5.3, SciPy 1.18.1, PyTorch 2.14.1+cpu, Numba 0.68.0, Biotite 1.7.1 and CD-HIT-EST 4.8.1.
