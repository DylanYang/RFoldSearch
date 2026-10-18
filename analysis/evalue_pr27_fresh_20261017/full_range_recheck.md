# Follow-up: experimental E-value range 0.01–100

**Status: exploratory, fixed-fit recheck; not a structurally cleaned refit.** This check replays all 516 annotated held-out queries from the frozen 885-chain/878-cluster database, score matrices and cross-validation fold models. All 27 threshold counts, for four score modes and both power corrections, match the saved per-query and family-balanced results. The original candidate Panel A already plots 0.01–100; only its *separate US-align-audited panel* stopped at E=1. The two panels must not be conflated.

## Whole-range calibration

Entries are family-balanced Rfam-negative cluster counts per query, expanded by the original $N_D/B_q$ factor. An ideally calibrated E-value would give 0.01, 0.1, 1, 10 and 100 in the five columns. Rfam-negative does not imply structurally unrelated.

| Score / correction | E=0.01 | E=0.1 | E=1 | E=10 | E=100 |
|---|---:|---:|---:|---:|---:|
| Raw SW / global | 0.063 | 0.077 | 0.539 | 476.3 | 878.0 |
| Raw SW / four-bin | 0.057 | 0.071 | 0.209 | 329.2 | 878.0 |
| SW/min / global | 0.009 | 0.085 | 1.174 | 10.08 | 253.3 |
| SW/min / four-bin | 0.010 | 0.062 | 1.144 | 12.82 | 250.3 |
| Comp-bias raw / global | 0.090 | 0.100 | 0.379 | 414.8 | 878.0 |
| Comp-bias raw / four-bin | 0.085 | 0.099 | 0.714 | 397.7 | 877.7 |
| Comp-bias SW/min / global | 0.011 | 0.132 | 0.916 | 9.11 | 347.4 |
| Comp-bias SW/min / four-bin | 0.011 | 0.087 | 0.657 | 7.78 | 210.2 |

The family-block bootstrap 95% intervals at E=10 are 442.9–510.4 for raw/global, 8.60–11.73 for SW/min/global, and 6.21–9.57 for comp-bias SW/min/four-bin. At E=100 they are 220.9–288.0 for SW/min/global and 168.6–252.7 for comp-bias SW/min/four-bin. The apparent near-perfect behavior of SW/min/global at E=10 is therefore real for this *proxy* dataset, but it does not extend to E=100.

## Structural check and its limit

The separate, database-matched directed US-align audit contains **496,828 of 509,687** required member pairs (97.5%); 12,859 remain unaudited. Its status was paused, and this recheck did not resume the expensive alignments. At each threshold we classified the saved, out-of-fold cluster leaders as structurally unrelated (mean TM<0.45), related, or unknown. The conservative upper bound additionally allows backfill when a related leader belongs to a multi-member cluster. The following are **bounds conditional on the frozen, not structurally refitted models**, not confidence intervals:

| Score / correction | Audited unrelated at E=10 | At E=100 |
|---|---:|---:|
| Raw SW / global | 468.6–474.3 | 869.7–875.7 |
| SW/min / global | 9.73–9.75 | 251.49–251.57 |
| Comp-bias SW/min / four-bin | 7.67–7.70 | 209.00–209.06 |

Thus the high-E discrepancy is not explained by the known structural relatives in this frozen comparison. The earlier E≤1 candidate audit remains the complete operational audit for that narrower candidate set; the near-complete member audit above supports only these bounded full-range diagnostics. A complete audit and structural exclusion *before both Gumbel and power refitting* are still required for definitive calibration.

## Extrapolation and saturation

Power parameters were fitted only for initial $E_0\leq100$. The median calibrated value at that boundary is 3.45 for raw/global, 28.95 for SW/min/global, and 45.79 for comp-bias SW/min/four-bin. Accordingly, E=10 already requires upper-tail extrapolation for **all** raw/global queries (86% for raw/four-bin), whereas E=100 requires it for **every** compared method. With the current linear extension above $E_0=100$ and final cap at $N_D=878$, raw/global cannot even output E=100: its median maximum attainable E is 30.3. All its annotated background clusters therefore fall below reported E=100, giving the artificial 878 plateau. At E=0.01, 76% of raw/global queries are below the fitted lower boundary. These threshold-specific support limits must accompany any 0.01–100 figure.

**Conclusion.** Selecting a method from E≤1 alone would be misleading. SW/min/global closely tracks the expected count at E=10, while comp-bias SW/min/global is closer at E=1; no candidate is acceptably calibrated across the entire requested 0.01–100 range. Do not deploy a new parameter set or claim Fig. S16A is validated from this run. Complete the remaining audit, refit both stages with structural exclusions/backfill, and repeat family/clan-blocked validation over the supported reported-E range. If E=100 remains beyond $E_0$ support, extend the *training/validation design* rather than treating the current extrapolation as validated.

Machine-readable replay and audit bounds: `full_range_recheck.json`. Reproduction code: `scripts/recheck_fresh_experimental_evalue_range.py`. This reuses the frozen score arrays and E-value functions; it is a cross-check of every saved count, not a new mmCIF extraction, tokenization or independent E-value implementation.
