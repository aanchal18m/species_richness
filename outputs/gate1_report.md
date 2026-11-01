# Gate 1: where is the data, and where is it not?

Download: GBIF.org (5 October 2026) GBIF Occurrence Download https://doi.org/10.15468/dl.3k5dhd
(28,449,817 records). Sampling unit: (cell, point, day), pooling observers (rule C). Dedup:
same observer only. Winters 2015/16-2023/24. Tables in `outputs/tables/gate1_*.csv`; figures in
`outputs/figures/gate1_*.png`. **The threshold below is a recommendation; the team decides.**

## Pipeline counts

| Step | Records |
|---|---|
| Download | 28,449,817 |
| After cleaning, whole winters and grid assignment | 24,010,074 |
| After cross-chunk same-observer dedup (N = 1) | 21,791,750 |
| Sampling units (point-days) | 775,499 |
| Cells with records: N = 1 / 3 / 5 / 10 | 16,684 / 15,598 / 14,894 / 13,079 (of 32,676) |

## Coverage distribution (cells with records)

| N | p10 | p25 | median | p75 | exactly 1.0 |
|---|---|---|---|---|---|
| 1 | 0.319 | 0.613 | 0.872 | 0.994 | 3,617 (median T = 1) |
| 3 | 0.415 | 0.669 | 0.896 | 0.997 | 3,596 |
| 5 | 0.470 | 0.702 | 0.908 | 0.998 | 3,576 |
| 10 | 0.565 | 0.757 | 0.932 | 1.000 | 3,431 |

**The guard is not optional.** With T = 1, `observed_coverage` multiplies the singleton
term by (T - 1) = 0 and returns 1.0 for any list: all 3,582 cells with T < 2 report
coverage 1.0. Without a guard, 9,778 cells "pass" 0.80 at N = 1; with T >= 2, 6,196.

## Cells clearing each threshold (guard T >= 20 point-day units)

| N | 0.80 | 0.90 | 0.95 |
|---|---|---|---|
| 1 | 3,900 | 3,401 | 2,573 |
| 3 | 3,520 | 3,255 | 2,548 |
| 5 | 3,296 | 3,116 | 2,500 |
| 10 | 2,734 | 2,691 | 2,299 |

Other guards are in `gate1_threshold_counts.csv`. Counts below record guards (N = 1):
records < 5: 1,359 cells; < 20: 4,018; < 100: 8,986. A guard of 20 units implies at least
20 records, so a separate record guard is redundant.

## How far the split moves with N (Jaccard similarity to N = 1, guard T >= 20)

| Threshold | N = 3 | N = 5 | N = 10 | cells lost by N = 10 |
|---|---|---|---|---|
| 0.80 | 0.903 | 0.845 | 0.701 | 1,166 |
| 0.90 | 0.950 | 0.912 | 0.791 | 711 |
| 0.95 | 0.976 | 0.963 | 0.886 | 284 |

Raising N almost only removes cells (no more than 19 are gained at any setting).

## Overlap with flagged cells (guard T >= 20, N = 1)

| Threshold | well-sampled | Andaman & Nicobar | bio12 saturated (any / > half) |
|---|---|---|---|
| 0.80 | 3,900 | 22 | 21 / 11 |
| 0.90 | 3,401 | 20 | 17 / 9 |
| 0.95 | 2,573 | 17 | 11 / 5 |

The Andaman & Nicobar cells are excluded from training via `train_eligible`. Saturated
cells stay in, with bio16 preferred over bio12 at VIF.

## Guard simulation

120 richly sampled real cells per N (T >= 200, coverage >= 0.97). Subsample k of their
point-day units 25 times and compare the coverage estimate with the coverage k units
actually achieve, computed exactly from the full sample (`gate1_guard_simulation.csv`).

| k units | median bias (N = 1) | MAE (N = 1) | p90 abs error, N = 1 / 3 / 5 / 10 |
|---|---|---|---|
| 2 | -0.050 | 0.206 | 0.39 / 0.36 / 0.34 / 0.31 |
| 5 | -0.012 | 0.125 | 0.27 / 0.18 / 0.15 / 0.12 |
| 10 | -0.001 | 0.058 | 0.12 / 0.085 / 0.071 / 0.056 |
| 20 | 0.000 | 0.027 | 0.058 / 0.036 / 0.034 / 0.023 |
| 30 | 0.001 | 0.016 | 0.035 / 0.024 / 0.021 / 0.015 |

The gross false-pass rate (reported >= 0.90 while truly more than 0.05 below it) is 2-5% at
every k: the estimator rarely passes a badly short sample, and the error *spread* is what a
guard must control. Caveat: units from rich cells hold more species than units from sparse
cells, so this is optimistic for the sparsest cells.

## Recommendation (for the team to decide)

**Guard: T >= 20 point-day units.** At 20 units the 90th-percentile coverage error is
0.023-0.058 across N, about half the 0.10 gap between a 0.90 threshold and 1. At 10 units
it is 0.056-0.12 (at N = 1, larger than that gap). The guard also removes the T = 1
formula artefact (3,582 cells) and the noisy T = 2-4 range.

**Threshold: 0.90**, with these reasons:
1. The pool is comfortably above the plan's ~1,500-cell floor at every N (3,401 at N = 1,
   2,691 at N = 10).
2. Above that floor it is the most stable option: Jaccard 0.95 / 0.91 / 0.79 across N,
   against 0.90 / 0.85 / 0.70 at 0.80.
3. The estimator's error at T >= 20 (MAE ~0.03, p90 0.06 at N = 1) sits inside the 0.10
   margin at 0.90. At 0.95 the margin is 0.05, the same size as the error, so membership
   near the threshold is largely noise.
4. Richness rarefied to 0.90 coverage has a median of ~106 species in the pool: a substantive
   inventory, not a fragment.

Alternative: **0.95** if stability across N and nearer-complete inventories matter more than
pool size (2,573 to 2,299 cells; Jaccard 0.98 / 0.96 / 0.89). **0.80 is not recommended**: it
is the least stable across N and admits the least complete inventories.

## Findings to carry forward

- **The training pool is spatially concentrated.** 36.4% of candidate cells (0.90, T >= 20,
  N = 1) lie south of 15N, against 10.5% of the grid: Kerala, Karnataka and Tamil Nadu plus
  city clusters. Central India is mostly poorly sampled; Ladakh and the high Himalaya have
  almost no records. Expect this to drive C3 (area of applicability) and the spatial-CV block
  size.
- **One pathological cell at N = 1** (47 units, 2 species, coverage 1.0): with one singleton
  and no doubletons, `richness.py`'s bias-corrected form returns coverage 1.0. N >= 3 removes it.
- **Effort grows more than five-fold** from winter 2015/16 to 2023/24 (1.04M to 5.60M
  records). Pooled winters are dominated by recent years.
