"""Part 10, step 2: N sensitivity for fixed decision 7 (design pre-specified in CLAUDE.md).

C1 and C2 only, at N = 1, 5, 10, beside the frozen N = 3 result: RF, nested
tuning as Part 5, primary blocking with the block size held at 100 km. The
frozen N = 3 predictor list is reused (not re-screened), so only the response
and the training pool change. Tables go to data/processed/sensitivity/; the
frozen model table is never written.

Run: python src/n_sensitivity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_models as fm  # noqa: E402
import spatial_cv as sc  # noqa: E402
from model_table import FROZEN, PROCESSED, assemble, verify_frozen  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SENS = PROCESSED / "sensitivity"
TABLES = ROOT / "outputs" / "tables"
N_VALUES = (1, 5, 10)
BLOCK_M = 100_000


def evaluate(table: pd.DataFrame, preds: list[str], response: str, origin) -> dict:
    train = table[table.train_eligible].reset_index(drop=True)
    X, y = train[preds].to_numpy(float), train[response].to_numpy(float)
    blocks = sc.block_ids(train[["x", "y"]].to_numpy(float), BLOCK_M, origin=origin)
    folds = sc.blocked_folds(blocks, fm.K_OUTER, seed=fm.SEED)
    oof, _ = fm.cross_validate(lambda: fm.TunedRF(blocks), X, y, folds, needs_rows=True)
    return {"train_cells": int(len(y)), "n_blocks": int(len(np.unique(blocks))),
            "spearman_rho": float(spearmanr(y, oof)[0]), "c2_recall": fm.hotspot_recall(y, oof),
            "rmse": fm.rmse(y, oof)}


def main():
    if FROZEN.exists():
        verify_frozen()
    info = json.loads((PROCESSED / "predictors.json").read_text())
    meta = json.loads((ROOT / "data" / "interim" / "grid_meta.json").read_text())
    origin = (meta["x0"], meta["y0"])
    SENS.mkdir(parents=True, exist_ok=True)
    cv = pd.read_csv(TABLES / "cv_results.csv", keep_default_na=False, na_values=[""])
    c2 = pd.read_csv(TABLES / "c2_recall.csv")
    frozen = pd.read_parquet(PROCESSED / "model_table.parquet")
    rows = [{"N": 3, "train_cells": int(frozen.train_eligible.sum()),
             "n_blocks": int(cv.query("scheme == 'spatial_residual' and model == 'rf'").n_blocks.iloc[0]),
             "spearman_rho": float(cv.query("scheme == 'spatial_residual' and model == 'rf'").spearman_rho.iloc[0]),
             "c2_recall": float(c2.query("scheme == 'spatial_residual' and model == 'rf'").c2_recall.iloc[0]),
             "rmse": float(cv.query("scheme == 'spatial_residual' and model == 'rf'").rmse.iloc[0]),
             "source": "frozen Part 5 run"}]
    for n in N_VALUES:
        t = assemble(n)
        t.to_parquet(SENS / f"model_table_N{n}.parquet", index=False)
        r = evaluate(t, info["selected"], info["response"], origin)
        rows.append({"N": n, **r, "source": "sensitivity"})
        print(f"   N={n}: {r}", flush=True)
    if FROZEN.exists():
        verify_frozen()            # the frozen table must be untouched
    df = pd.DataFrame(rows).sort_values("N")
    df["meets_c1"], df["meets_c2"] = df.spearman_rho >= 0.60, df.c2_recall >= 0.70
    df.to_csv(TABLES / "n_sensitivity.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
