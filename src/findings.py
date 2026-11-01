"""Part 9: outputs/findings.md -- the material for the report, not the report.

Every number is read from the output tables at run time, so the document cannot
drift from the results. Narrative is limited to bullets; the team writes the
prose (assignment cap on AI-generated content). The decision log is copied
verbatim from CLAUDE.md.

Run: python src/findings.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_models as fm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "outputs" / "tables"
P = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "findings.md"


def md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def claude_sections(names: list[str]) -> str:
    text = (ROOT / "CLAUDE.md").read_text()
    parts = re.split(r"(?m)^## ", text)
    keep = [("## " + p).rstrip() for p in parts[1:] if any(p.startswith(n) for n in names)]
    return "\n\n".join(keep)


def build() -> str:
    # keep_default_na=False: pandas otherwise reads the model name "null" as missing
    cv = pd.read_csv(T / "cv_results.csv", keep_default_na=False, na_values=[""])
    c2 = pd.read_csv(T / "c2_recall.csv", keep_default_na=False, na_values=[""])
    stress = pd.read_csv(T / "stress_downsample_summary.csv")
    aoa = json.loads((T / "aoa_report.json").read_text())
    aoa_b = pd.read_csv(T / "aoa_breakdown.csv")
    lo = aoa_b.group.str.extract(r"\[(-?\d+)")[0].astype(float)
    aoa_b = aoa_b.assign(_k=aoa_b.breakdown.map({"record_status": 0}).fillna(1), _lo=lo.fillna(-1e9)) \
        .sort_values(["breakdown", "_lo"], kind="stable").drop(columns=["_k", "_lo"])
    imp = pd.read_csv(T / "importance_comparison.csv", index_col=0)
    interp = json.loads((T / "interpretation_report.json").read_text())
    preds = json.loads((P / "predictors.json").read_text())
    corr = pd.read_csv(T / "response_correlogram.csv")
    audit = pd.read_csv(T / "cleaning_audit.csv")
    clean = json.loads((T / "clean_report.json").read_text())["incidence"]
    dl = json.loads((T / "gbif_download_summary.json").read_text())
    env = json.loads((T / "env_report.json").read_text())

    table = pd.read_parquet(P / "model_table.parquet")
    train = table[table.train_eligible].reset_index(drop=True)
    oof = pd.read_parquet(P / "oof_predictions.parquet")
    rf_resid = (oof.observed - oof.spatial_residual_rf).to_numpy()
    rf_corr = fm.correlogram(train[["x", "y"]].to_numpy(float), rf_resid)
    corr["rf_oof_residual"] = rf_corr["autocorrelation"].to_numpy()
    S = pd.read_parquet(P / "shap_train.parquet")
    bands = pd.cut(train.elev_mean, [0, 200, 500, 1000, 1500, 2000, 3000, 6000], right=False)
    elev = pd.DataFrame({"band": bands.astype(str), "shap": S.elev_mean, "obs": train.richness_c090}) \
        .groupby("band", sort=False).agg(training_cells=("shap", "size"), shap_elev_median=("shap", "median"),
                                         observed_richness_median=("obs", "median")).reset_index()
    elev["order"] = elev.band.str.extract(r"\[(\d+)").astype(int)
    elev = elev.sort_values("order").drop(columns="order")

    pred_all = pd.read_parquet(P / "predictions.parquet")
    an = pred_all[pred_all.andaman_nicobar.fillna(False).astype(bool)]
    flags = pd.read_parquet(ROOT / "data" / "interim" / "cell_flags.parquet")
    sat_any, sat_half = int((flags.bio12_saturated_frac > 0).sum()), int((flags.bio12_saturated_frac > 0.5).sum())
    above_1500 = int((train.elev_mean >= 1500).sum())
    dep = pd.read_csv(T / "dependence_curves.csv").query("predictor == 'elev_mean'")
    top_bin = dep.sort_values("bin").iloc[-1]
    eb = aoa_b[aoa_b.breakdown.str.startswith("elev_band")].copy()
    eb["_lo"] = eb.group.str.extract(r"\[(-?\d+)")[0].astype(float)
    hi_elev = eb[eb._lo >= 3000].share_inside
    south_train = float((train.lat < 15).mean())
    south_grid = float((table.lat < 15).mean())

    rel = json.loads((T / "reliability_summary.json").read_text())
    rs = rel["summary"]
    rel_tab = pd.DataFrame(rel["per_split"])
    pools = pd.DataFrame([{"pool": k.replace("pool_", "").replace("_", " "), **v} for k, v in rel.items()
                          if k.startswith("pool_")])
    nsens_path = T / "n_sensitivity.csv"
    nsens = pd.read_csv(nsens_path) if nsens_path.exists() else None

    g = lambda s, m: cv.query("scheme == @s and model == @m").iloc[0]
    prim, resp, rnd = g("spatial_residual", "rf"), g("spatial_response", "rf"), g("random", "rf")
    c2p = c2.query("scheme == 'spatial_residual' and model == 'rf'").iloc[0]
    c2r = c2.query("scheme == 'random' and model == 'rf'").iloc[0]
    c2q = c2.query("scheme == 'spatial_response' and model == 'rf'").iloc[0]
    s50 = stress.query("level == 50").iloc[0]
    s100 = stress.query("level == 100").iloc[0]
    s10 = stress.query("level == 10").iloc[0]
    eff = interp["effort"]
    k = int(round(0.2 * len(oof)))
    obs, pr = oof.observed.to_numpy(), oof.spatial_residual_rf.to_numpy()
    hits = len(set(np.argsort(-obs, kind="stable")[:k]) & set(np.argsort(-pr, kind="stable")[:k]))

    criteria = pd.DataFrame([
        ["C1 pattern", "Spearman rho, predicted vs withheld observed, spatially blocked CV", ">= 0.60",
         f"{prim.spearman_rho:.3f} (100 km residual-based blocks)",
         f"PASS -- contingent: {resp.spearman_rho:.3f} (FAIL) under the original 400 km response-based "
         f"rule, revised after the first run; weakest fold {prim.spearman_rho_fold_min:.3f}"],
        ["C2 hotspots", "Recall of true top-quintile cells in predicted top quintile", ">= 0.70",
         f"{c2p.c2_recall:.3f} ({hits} of {k})",
         f"FAIL at every level: 400 km {c2q.c2_recall:.3f}, random CV {c2r.c2_recall:.3f}"],
        ["C3 reach", "Share of poorly sampled cells inside the area of applicability", ">= 0.50",
         f"{aoa['c3']:.3f} ({aoa['non_eligible_inside']:,} of {aoa['non_eligible_cells']:,})", "PASS"],
    ], columns=["Criterion", "Measure", "Bar", "Result", "Verdict"])

    cv_tab = cv.assign(block_km=cv.block_km.fillna(0).astype(int).astype(str).replace("0", "--"),
                       n_blocks=cv.n_blocks.fillna(0).astype(int).astype(str).replace("0", "--"))[
        ["scheme", "model", "block_km", "n_blocks", "spearman_rho", "spearman_rho_fold_mean",
         "spearman_rho_fold_min", "rmse", "mae", "r2"]]
    local = audit[audit.stage == "local"][["filter", "dropped", "kept"]]
    gbif_ev = dl["event_fields"]
    om = env["grid_omissions"]

    lines = f"""# Findings: material for the report

Generated by `src/findings.py` from the output tables; regenerate after any rerun.
Not report prose -- tables, captions, bullets and the decision log for the team to
write from. No discussion section by design.

Data citation: GBIF.org (5 October 2026) GBIF Occurrence Download
https://doi.org/10.15468/dl.3k5dhd. Other sources and citations: `data/raw/README.md`.

## 1. Pre-registered criteria

{md(criteria)}

- C1 is judged under spatially blocked CV (team decision, 2026-10-06; proposal
  section 3 governs). C2 is read at the same level, by assumption -- confirm.
- Chance level for C2 is 0.20.

## 2. Data and pipeline numbers

| Step | Count |
|---|---|
| GBIF records downloaded (Aves, India, 2015-2024, Nov-Feb) | {dl['rows']:,} |
| Distinct species (speciesKey) in download | {dl['distinct_species']:,} |
| Records with usable recordedBy AND single-day eventDate | {gbif_ev['recordedBy_and_single_day']:,} ({100 * gbif_ev['recordedBy_and_single_day_frac']:.2f}%) |
| Records carrying eventID | {gbif_ev['eventID_present']:,} |
| Records after cleaning, whole winters, grid assignment | {int(local.kept.iloc[-1]):,} |
| Same-observer duplicates removed across chunks | {clean['cross_chunk_duplicates']:,} |
| Sampling units (cell, point, day) | {clean['N1']['units_in']:,} |
| Grid cells (10 km equal-area) | {len(table):,} |
| Cells with records (N = 3) | {int(table.has_records.sum()):,} |
| Training-eligible cells (coverage >= 0.90, T >= 20, N = 3, not A&N) | {int(table.train_eligible.sum()):,} |
| Prediction targets (non-eligible) | {int((~table.train_eligible).sum()):,} |
| Predictors after VIF screening | {len(preds['selected'])} |

Cleaning audit (local filters; GBIF server-side filters in `outputs/tables/cleaning_audit.csv`):

{md(local)}

Predictors retained (22): {", ".join(preds['selected'])}. Land-cover reference class:
cropland. Final VIF max {max(preds['final_vif'].values()):.1f} (elev_mean, protected).

## 3. Cross-validation (Part 5)

{md(cv_tab)}

## 4. Hotspot recall, C2 (Part 6)

{md(c2[['scheme', 'model', 'c2_recall', 'c2_recall_fold_mean', 'c2_recall_fold_min']])}

## 5. Training-set-size stress test (Part 6, pre-specified)

{md(stress.assign(mean_train_cells=stress.mean_train_cells.round().astype(int)))}

Held-out folds identical at every level; 100% row is the Part 5 primary run.

## 6. Area of applicability, C3 (Part 7)

C3 = {aoa['c3']:.4f}; threshold DI = {aoa['threshold_di']:.3f}; training cells inside by
their own CV DI = {aoa['training_cells_inside_by_own_di']:.3f}.

{md(aoa_b[['breakdown', 'group', 'cells', 'inside_aoa', 'share_inside', 'median_di']])}

## 7. Interpretation (Part 8)

Importance (share of total) -- SHAP and permutation; rank agreement Spearman
{interp['rank_agreement_spearman']:.2f}:

{md(imp[['shap_share', 'perm_share', 'shap_rank', 'perm_rank']].reset_index().rename(columns={'index': 'predictor'}))}

Elevation effect by fixed band (training cells):

{md(elev, '{:.2f}')}

Pre-set hump test (decile bins, fixed before results): interior_peak =
{interp['peaks']['elev_mean']['interior_peak']}; rise {interp['peaks']['elev_mean']['rise_from_low_end']:.2f},
fall {interp['peaks']['elev_mean']['fall_to_high_end']:.2f}, threshold {interp['interior_peak_test_min_drop']:.2f} species.

Effort check (Spearman with T, point-day units; 95% CI from a 100 km block bootstrap):

| Pair | rho | 95% CI |
|---|---|---|
| Observed richness vs T | {eff['observed_vs_T']['rho']:.3f} | {eff['observed_vs_T']['ci95'][0]:.3f} to {eff['observed_vs_T']['ci95'][1]:.3f} |
| Out-of-fold RF residual vs T | {eff['residual_vs_T']['rho']:.3f} | {eff['residual_vs_T']['ci95'][0]:.3f} to {eff['residual_vs_T']['ci95'][1]:.3f} |
| Built-up fraction vs T | {eff['lc_built_vs_T']['rho']:.3f} | {eff['lc_built_vs_T']['ci95'][0]:.3f} to {eff['lc_built_vs_T']['ci95'][1]:.3f} |
| SHAP of built-up vs T | {eff['shap_lc_built_vs_T']['rho']:.3f} | {eff['shap_lc_built_vs_T']['ci95'][0]:.3f} to {eff['shap_lc_built_vs_T']['ci95'][1]:.3f} |

Spatial autocorrelation by distance band (response; linear-model residuals, all
training cells; RF out-of-fold residuals):

{md(corr)}

## 7b. Response reliability: split-half (Part 10, pre-specified)

Each training cell's point-day units split at random into halves; each half
rarefied to coverage 0.80; cells kept where both halves attain it. 5 splits.

{md(rel_tab, '{:.4f}')}

| Summary (mean; range) | Value |
|---|---|
| Half-depth reliability r_half | {rs['r_half']['mean']:.3f} ({rs['r_half']['min']:.3f}-{rs['r_half']['max']:.3f}) |
| Full-depth reliability (Spearman-Brown) | {rs['r_full']['mean']:.3f} ({rs['r_full']['min']:.3f}-{rs['r_full']['max']:.3f}) |
| C1 ceiling sqrt(r_full) | {rs['c1_ceiling']['mean']:.3f} -- model achieves {prim.spearman_rho:.3f} |
| Raw half-half top-quintile agreement | {rs['c2_half_agreement']['mean']:.3f} ({rs['c2_half_agreement']['min']:.3f}-{rs['c2_half_agreement']['max']:.3f}) |
| Normal-model check: simulated agreement at r_half | {rs['c2_sim_at_r_half']['mean']:.3f} (vs observed {rs['c2_half_agreement']['mean']:.3f}) |
| C2 ceiling, simulated at full depth | {rs['c2_ceiling_sim']['mean']:.3f} ({rs['c2_ceiling_sim']['min']:.3f}-{rs['c2_ceiling_sim']['max']:.3f}) -- model achieves {c2p.c2_recall:.3f} |

Kept vs dropped cells (dropped = failed 0.80 in at least one split):

{md(pools, '{:.3f}')}

Caveats: halves share observers and sites, so reliability and both ceilings are
optimistic; it measures sampling noise in the richness estimate, not all
measurement error. Halves use coverage 0.80; the response uses 0.90. Dropped
cells are the thinly sampled ones, so reliability describes the better-sampled
part of the pool.
""" + ("" if nsens is None else f"""
## 7c. N sensitivity (fixed decision 7; Part 10, pre-specified)

RF, nested tuning, 100 km blocks held fixed, frozen N = 3 predictor list.

{md(nsens)}
""") + f"""
## 8. Figure captions (factual)

- **fig1_predicted_richness.png.** Predicted winter bird richness at 0.90 sample
  coverage for the {int(table.shape[0] - (~pd.read_parquet(P / 'predictions.parquet').aoa).sum()):,} grid cells inside the
  area of applicability; seven classes with equal numbers of cells. Hatched cells
  ({int((~pd.read_parquet(P / 'predictions.parquet').aoa).sum()):,}) are outside the area of applicability and carry no prediction.
- **fig2_aoa.png.** Training cells ({int(table.train_eligible.sum()):,}), predicted cells inside
  the area of applicability ({aoa['non_eligible_inside']:,}), and cells outside it
  ({aoa['non_eligible_cells'] - aoa['non_eligible_inside']:,}). Right: share of poorly sampled cells inside, by mean elevation band.
- **fig3_observed_vs_predicted.png.** Observed vs out-of-fold predicted richness for
  {len(oof):,} training cells under 100 km blocked 5-fold CV; Spearman rho
  {prim.spearman_rho:.3f}, RMSE {prim.rmse:.1f} species, R^2 {prim.r2:.2f}. Predictions span
  {pr.min():.0f}-{pr.max():.0f} species against observed {obs.min():.0f}-{obs.max():.0f}: the
  extremes are compressed.
- **fig4_hotspots.png.** True vs predicted top-quintile training cells (k = {k}) under
  100 km blocked CV: {hits} hits, {k - hits} misses, {k - hits} false alarms (equal by
  construction); recall {hits / k:.3f}.
- **fig5_stress.png.** Pooled Spearman rho and C2 recall when each fold's training
  side is thinned to 75/50/25/10% (5 replicates; band = min-max); held-out folds unchanged.
- **shap_importance_comparison.png.** Share of total importance per predictor:
  mean |SHAP| and permutation RMSE increase, training cells.
- **shap_dependence.png.** SHAP value against predictor value for the top six
  predictors and mean elevation; line = median in 5% bins (1% at the ends).
- **effort_check.png.** Left: out-of-fold residual against T (log). Right: SHAP of
  built-up fraction against built-up fraction, coloured by T.
- **gate1_split_map.png, gate1_coverage_ecdf.png.** Gate 1 split and coverage
  distribution for N = 1, 3, 5, 10.

## 9. The argument, in bullets

- Question: can environmental variables predict winter bird richness in poorly
  sampled India well enough to identify biodiversity patterns?
- Broad pattern: moderately, yes -- C1 passes (rho {prim.spearman_rho:.3f} under 100 km
  blocking) but only under the revised block rule; C3 passes widely ({aoa['c3']:.1%}).
- Hotspots: no -- C2 fails at every validation level, including random CV
  ({c2r.c2_recall:.3f}); about half of true hotspots are missed. Predictions compress
  the extremes, so the richest cells are under-predicted.
- The RF earns its complexity: beats linear and null under every scheme (spatial
  rho {prim.spearman_rho:.3f} vs {g('spatial_residual', 'linear').spearman_rho:.3f} vs undefined; R^2
  {prim.r2:.2f} vs {g('spatial_residual', 'linear').r2:.2f}).
- Validation decline (S1): rho {rnd.spearman_rho:.3f} random -> {prim.spearman_rho:.3f}
  100 km -> {resp.spearman_rho:.3f} 400 km; C2 {c2r.c2_recall:.3f} -> {c2p.c2_recall:.3f} -> {c2q.c2_recall:.3f}.
- Ecology (S3): built-up land strongly negative (urbanisation), open water positive
  (winter waterbirds), rainfall seasonality positive; mid-elevation hump visible
  but not confirmed (section 10).
- "The limit is the predictors, not the data" -- three lines, with their strength:
  1. Quantity: halving the training data barely matters (rho {s50.rho_mean:.3f} at 50% vs
     {s100.rho_mean:.3f} at 100%); more data of the same kind is not the bottleneck.
     Strong. (It still falls to {s10.rho_mean:.3f} at 10%.)
  2. Coverage of environmental space: {aoa['c3']:.1%} of poorly sampled cells resemble the
     training cells; failure is not extrapolation. Strong.
  3. Unexplained spatial structure: RF out-of-fold residuals autocorrelate at short
     range ({corr.rf_oof_residual.iloc[0]:.2f} within 10 km, {corr.rf_oof_residual.iloc[1]:.2f} at 10-25 km, < 0.1 by 25-50 km).
     MODERATE: consistent with missing local drivers, but equally with shared local
     sampling effects (same observers in neighbouring cells).
  4. Noise in the response is not the limit (split-half reliability,
     section 7b): full-depth reliability {rs['r_full']['mean']:.3f}, so a perfect model could
     reach rho ~{rs['c1_ceiling']['mean']:.2f} (achieved {prim.spearman_rho:.3f}) and hotspot recall ~{rs['c2_ceiling_sim']['mean']:.2f}
     (achieved {c2p.c2_recall:.3f}). The 0.70 C2 bar was attainable in principle. Strong for
     sampling noise; optimistic because halves share observers and sites, so error
     shared across a cell's observers remains unmeasured.
  - Net: data quantity, environmental coverage and sampling noise are each ruled
    out as the main limit; what remains is the predictor set (and any error shared
    across a cell's observers, which no test here can see).
- Where the model may not be used: high Himalaya (>= 3,000 m: {hi_elev.min():.0%}-{hi_elev.max():.0%} inside),
  Andaman & Nicobar ({int(an.aoa.sum())} of {len(an)} inside), Thar/Kutch and NE hills fringes.

## 10. Limitations (all)

- **Block-size rule revised after the first result.** First run: response-based
  400 km blocks, RF rho 0.576 (fail; first-run value, recorded in CLAUDE.md). Revised to residual-based 100 km blocks
  (Roberts et al. 2017): rho {prim.spearman_rho:.3f} (pass); the original rule now gives
  {resp.spearman_rho:.3f} (fail). The C1 verdict depends on this post-hoc choice; report both rows.
- **RF grid widened after the first run** (max_features 0.2 added; tuning had hit the
  grid corner in all 10 folds).
- **Effort leaks slightly:** out-of-fold residual vs T rho {eff['residual_vs_T']['rho']:.2f}
  (CI {eff['residual_vs_T']['ci95'][0]:.2f}-{eff['residual_vs_T']['ci95'][1]:.2f}). Direction ambiguous: effort passing the coverage
  standardisation, or observers choosing places richer than their environment predicts.
- **Mid-elevation hump not confirmed by the pre-set test** (decile bins merged
  {top_bin.x_low:,.0f}-{top_bin.x_high:,.0f} m); visible in fixed bands; only {above_1500} training cells above 1,500 m;
  elevation shares signal with temperature predictors (elev_mean VIF 13.5).
- **Proposal 6.5 Level 3 not run.** The stress test thins training cells (data
  quantity); the proposal's version thins sampling units within cells (shallow
  response). Different question.
""" + ("""- **Fixed decision 7 not met for models.** N was to be reported at 1, 3, 5, 10;
  Gate 1 was, but models (Parts 5-8) were run at N = 3 only. Either rerun or amend.""" if nsens is None else
f"""- **N sensitivity covers C1 and C2 only** (section 7c): random CV, 400 km blocks,
  SHAP and AOA were run at N = 3 only; predictors were not re-screened per N.""") + f"""
- **bio12 clipped** at 6,553.5 mm in {sat_any} cells ({sat_half} more than half clipped); bio12
  and bio16 both dropped by VIF; bio13 carries wet-season rainfall.
- **Training pool concentrated in the south:** {south_train:.1%} of training-eligible cells lie
  south of 15N vs {south_grid:.1%} of all grid cells.
- **GBIF route:** eBird records carry no eventID or completeness flag; units are
  (cell, point, day) pooling observers (rule C), which merges independent same-day
  hotspot visits and slightly understates coverage. eBird Basic Dataset not used.
- **Coverage standardisation guard:** T >= 20 units; the guard simulation is
  optimistic for sparse cells (units from rich cells hold more species).
- **CHELSA artefacts:** Andaman & Nicobar temperature range is marine ({len(an)} cells,
  excluded from training); coastal diurnal range attenuated; bio3/bio4 rescaled and
  bio16/bio18 unwrapped (verified by identities).
- **Islands lost from the grid:** all of Lakshadweep, small Andaman/Nicobar and
  Sundarbans islands ({om['their_area_km2']:.0f} km^2); {om['india_land_outside_kept_cells_km2']:,} km^2 of land outside kept cells.
- **Temporal mismatch:** climate 1981-2010, land cover 2021, records 2015/16-2023/24;
  effort grew >5x over the period.
- **AOA weights** use in-sample permutation importance (approximating CAST's OOB
  importance); AOA and spatial CV not validated against R/CAST (R not installed).
- **Inside the AOA is not "accurate"**: accuracy there is what C1/C2 measured.
- **Predicted extremes compressed:** predictions {pr.min():.0f}-{pr.max():.0f} vs observed
  {obs.min():.0f}-{obs.max():.0f} species.
- **Copernicus DEM is a surface model**; open-pit mines give elevation minima of -93 m.
- **C2 level by assumption** (same as C1); not separately decided.
- **Reliability is optimistic** (shared observers and sites between halves); the
  ceilings bound sampling noise only.

## Appendix: decision log (verbatim from CLAUDE.md)

{claude_sections(['Route decision', 'Decisions that are FIXED', 'Traps already discovered', 'Parts'])}
"""
    return lines


def main():
    OUT.write_text(build())
    print(f"{OUT.relative_to(ROOT)}: {len(OUT.read_text().splitlines())} lines")


if __name__ == "__main__":
    main()
