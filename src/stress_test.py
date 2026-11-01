"""Part 6: C2 hotspot recall, and the training-set-size stress test.

Stages
  c2          C2 recall from data/processed/oof_predictions.parquet, for the RF
              at both block sizes and under random CV, plus the baselines.
              -> outputs/tables/c2_recall.csv
  downsample  the pre-specified stress test (CLAUDE.md, Part 6): keep 75 / 50 /
              25 / 10% of the TRAINING side of each primary 100 km fold, 5
              replicates each, nested-tuned RF, scored on the unchanged
              held-out folds. -> outputs/tables/stress_downsample.csv
              (one row per replicate) and stress_downsample_summary.csv

This is training-set-size sensitivity, not the proposal's section 6.5 Level 3
(downsampling sampling units within cells); see CLAUDE.md.

Run: python src/stress_test.py c2 | downsample
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_models as fm  # noqa: E402
import spatial_cv as sc  # noqa: E402
from model_table import FROZEN, verify_frozen  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
LEVELS = (75, 50, 25, 10)
REPLICATES = 5
SEED_BASE = 20261006
C2_BAR = 0.70


def stage_c2():
    o = pd.read_parquet(PROCESSED / "oof_predictions.parquet")
    rows = []
    for scheme in ("spatial_residual", "spatial_response", "random"):
        for model in ("rf", "linear"):
            col = f"{scheme}_{model}"
            per_fold = [fm.hotspot_recall(d["observed"], d[col]) for _, d in o.groupby(f"fold_{scheme}")]
            rows.append({"scheme": scheme, "model": model,
                         "c2_recall": fm.hotspot_recall(o["observed"], o[col]),
                         "c2_recall_fold_mean": float(np.mean(per_fold)),
                         "c2_recall_fold_min": float(np.min(per_fold)),
                         "n_cells": len(o), "top_k": int(round(0.2 * len(o)))})
    df = pd.DataFrame(rows)
    df["meets_bar"] = df["c2_recall"] >= C2_BAR
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / "c2_recall.csv", index=False)
    print(df.round(3).to_string(index=False))


def downsampled_cv(X, y, folds, blocks, frac: float, level: int, replicate: int, make=None):
    """Out-of-fold predictions with each fold's TRAINING rows thinned to `frac`;
    the held-out rows are untouched. Seeds per (level, replicate, fold)."""
    make = make or (lambda: fm.TunedRF(blocks))
    oof = np.full(len(y), np.nan)
    used = {}
    for f, tr, te in sc.iter_folds(folds):
        rng = np.random.default_rng([SEED_BASE, level, replicate, int(f)])
        sub = np.sort(rng.choice(tr, size=int(round(frac * len(tr))), replace=False))
        used[int(f)] = sub
        oof[te] = make().fit(X[sub], y[sub], sub).predict(X[te])
    return oof, used


def stage_downsample():
    if FROZEN.exists():
        verify_frozen()
    table = pd.read_parquet(PROCESSED / "model_table.parquet")
    info = json.loads((PROCESSED / "predictors.json").read_text())
    train = table[table.train_eligible].reset_index(drop=True)
    X, y = train[info["selected"]].to_numpy(float), train[info["response"]].to_numpy(float)
    o = pd.read_parquet(PROCESSED / "oof_predictions.parquet")
    if not (o["cell_id"].to_numpy() == train["cell_id"].to_numpy()).all():
        raise AssertionError("OOF rows do not line up with the frozen training cells")
    folds = o["fold_spatial_residual"].to_numpy()
    meta = json.loads((ROOT / "data" / "interim" / "grid_meta.json").read_text())
    block_m = 1000 * float(pd.read_csv(TABLES / "cv_results.csv")
                           .query("scheme == 'spatial_residual'")["block_km"].iloc[0])
    blocks = sc.block_ids(train[["x", "y"]].to_numpy(float), block_m, origin=(meta["x0"], meta["y0"]))

    rows = [{"level": 100, "replicate": 0, "train_frac": 1.0,
             "spearman_rho": float(spearmanr(y, o["spatial_residual_rf"])[0]),
             "c2_recall": fm.hotspot_recall(y, o["spatial_residual_rf"]),
             "mean_train_cells": float(np.mean([len(tr) for _, tr, _ in sc.iter_folds(folds)]))}]
    for level in LEVELS:
        for rep in range(REPLICATES):
            oof, used = downsampled_cv(X, y, folds, blocks, level / 100, level, rep)
            rows.append({"level": level, "replicate": rep, "train_frac": level / 100,
                         "spearman_rho": float(spearmanr(y, oof)[0]),
                         "c2_recall": fm.hotspot_recall(y, oof),
                         "mean_train_cells": float(np.mean([len(u) for u in used.values()]))})
            r = rows[-1]
            print(f"   {level:3d}% rep {rep}: rho {r['spearman_rho']:.3f}  C2 {r['c2_recall']:.3f}  "
                  f"(~{r['mean_train_cells']:.0f} training cells per fold)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "stress_downsample.csv", index=False)
    summ = df.groupby("level").agg(
        replicates=("replicate", "size"), mean_train_cells=("mean_train_cells", "mean"),
        rho_mean=("spearman_rho", "mean"), rho_min=("spearman_rho", "min"), rho_max=("spearman_rho", "max"),
        c2_mean=("c2_recall", "mean"), c2_min=("c2_recall", "min"), c2_max=("c2_recall", "max"),
    ).sort_index(ascending=False)
    summ.to_csv(TABLES / "stress_downsample_summary.csv")
    print(summ.round(3).to_string())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["c2", "downsample"])
    a = p.parse_args(argv)
    {"c2": stage_c2, "downsample": stage_downsample}[a.stage]()


if __name__ == "__main__":
    main()
