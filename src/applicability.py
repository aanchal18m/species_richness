"""Part 7: final RF, predictions for every cell, area of applicability, C3.

Design pre-specified in CLAUDE.md (Part 7) before C3 was computed:
  * final RF on all training-eligible cells, Part 5 tuning (inner blocked
    4-fold on the 100 km residual blocks), saved to models/rf_final.joblib
  * predictions for all cells -> data/processed/predictions.parquet (the frozen
    model table is read, never written)
  * AOA weights = permutation importance of the final RF on the training cells
    (RMSE increase, 10 repeats, seed 0; negatives -> 0)
  * training DI cross-validated with the primary 100 km folds; threshold =
    Tukey upper whisker (aoa.di_threshold), not tuned
  * C3 = share of the non-eligible cells inside the AOA; bar >= 0.50

Writes outputs/tables/rf_permutation_importance.csv, aoa_breakdown.csv,
       aoa_report.json

Run: python src/applicability.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aoa  # noqa: E402
import fit_models as fm  # noqa: E402
import spatial_cv as sc  # noqa: E402
from model_table import FROZEN, verify_frozen  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
MODELS = ROOT / "models"
PREDICTIONS = PROCESSED / "predictions.parquet"
C3_BAR = 0.50
LAT_BANDS = [5, 10, 15, 20, 25, 30, 35, 40]
ELEV_BANDS = [-100, 200, 500, 1000, 2000, 3000, 4000, 9000]


def record_status(t: pd.DataFrame) -> pd.Series:
    s = pd.Series("no records", index=t.index)
    s[t["has_records"]] = "records, poorly sampled"
    s[t["has_records"] & t["andaman_nicobar"].fillna(False).astype(bool)
      & (t["coverage_full"] >= 0.90) & (t["T"] >= 20)] = "Andaman & Nicobar (excluded)"
    s[t["train_eligible"]] = "training-eligible"
    return s


def c3_fraction(inside: pd.Series, eligible: pd.Series) -> float:
    """Share of NON-eligible cells inside the AOA."""
    target = ~eligible.astype(bool)
    return float(inside[target].mean())


def breakdown(t: pd.DataFrame, by: str) -> pd.DataFrame:
    g = t.groupby(by, observed=True)
    return pd.DataFrame({"cells": g.size(), "inside_aoa": g["aoa"].sum(),
                         "share_inside": g["aoa"].mean().round(4),
                         "median_di": g["di"].median().round(3)}).reset_index().rename(columns={by: "group"})


def main():
    import joblib

    if FROZEN.exists():
        verify_frozen()
    table = pd.read_parquet(PROCESSED / "model_table.parquet")
    info = json.loads((PROCESSED / "predictors.json").read_text())
    preds = info["selected"]
    train = table[table.train_eligible].reset_index(drop=True)
    X_tr, y_tr = train[preds].to_numpy(float), train[info["response"]].to_numpy(float)
    X_all = table[preds].to_numpy(float)
    if np.isnan(X_all).any():
        raise AssertionError("missing predictor values in the model table")

    oof = pd.read_parquet(PROCESSED / "oof_predictions.parquet")
    if not (oof["cell_id"].to_numpy() == train["cell_id"].to_numpy()).all():
        raise AssertionError("OOF fold assignments do not line up with the training cells")
    folds = oof["fold_spatial_residual"].to_numpy()
    meta = json.loads((ROOT / "data" / "interim" / "grid_meta.json").read_text())
    block_m = 1000 * float(pd.read_csv(TABLES / "cv_results.csv")
                           .query("scheme == 'spatial_residual'")["block_km"].iloc[0])
    blocks = sc.block_ids(train[["x", "y"]].to_numpy(float), block_m, origin=(meta["x0"], meta["y0"]))

    final = fm.TunedRF(blocks).fit(X_tr, y_tr, np.arange(len(y_tr)))
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final.model_, "params": final.params_, "predictors": preds,
                 "response": info["response"]}, MODELS / "rf_final.joblib")
    print(f"   final RF params {final.params_}", flush=True)

    imp = aoa.permutation_importance(final.predict, X_tr, y_tr, n_repeats=10, seed=0)
    w = aoa.importance_weights(imp)
    pd.DataFrame({"predictor": preds, "importance_rmse_increase": imp, "aoa_weight": w}) \
        .sort_values("importance_rmse_increase", ascending=False) \
        .to_csv(TABLES / "rf_permutation_importance.csv", index=False)

    res = aoa.area_of_applicability(X_tr, X_all, w, folds=folds)
    out = table[["cell_id", "lon", "lat", "elev_mean", "has_records", "train_eligible",
                 "andaman_nicobar", "coverage_full", "T"]].copy()
    out["pred_richness_c090"] = final.predict(X_all)
    out["di"], out["aoa"] = res.di, res.aoa
    out["record_status"] = record_status(out)
    out.drop(columns=["coverage_full", "T"]).to_parquet(PREDICTIONS, index=False)

    c3 = c3_fraction(out["aoa"], out["train_eligible"])
    out["lat_band"] = pd.cut(out["lat"], LAT_BANDS, right=False).astype(str)
    out["elev_band"] = pd.cut(out["elev_mean"], ELEV_BANDS, right=False).astype(str)
    target = out[~out.train_eligible]
    parts = []
    for by in ("record_status", "lat_band", "elev_band"):
        b = breakdown(target if by != "record_status" else out, by)
        b.insert(0, "breakdown", by + ("" if by == "record_status" else " (non-eligible cells)"))
        parts.append(b)
    bd = pd.concat(parts, ignore_index=True)
    bd.to_csv(TABLES / "aoa_breakdown.csv", index=False)
    rep = {"c3": round(c3, 4), "c3_bar": C3_BAR, "meets_bar": bool(c3 >= C3_BAR),
           "non_eligible_cells": int((~out.train_eligible).sum()),
           "non_eligible_inside": int(target["aoa"].sum()),
           "threshold_di": res.threshold, "dbar": res.dbar,
           "training_di_quantiles": {q: float(np.quantile(res.train_di, q)) for q in (0.25, 0.5, 0.75, 0.95)},
           "training_cells_inside_by_own_di": float(np.mean(res.train_di <= res.threshold)),
           "final_rf_params": final.params_}
    (TABLES / "aoa_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(bd.to_string(index=False))


if __name__ == "__main__":
    main()
