"""Part 5, steps 3-4: three models, evaluated by random and spatially blocked CV.

Models (no more -- the brief asks for interpretation, not algorithmic novelty):
  null     predicts the training-fold mean. Anything that cannot beat it has
           learned nothing.
  linear   ordinary least squares on standardised predictors.
  rf       random forest; a small grid tuned INSIDE each outer training set by
           an inner blocked CV, so the outer score never sees the tuning data.

All three get the same VIF-screened predictors (data/processed/predictors.json)
and the same folds, so the comparison says whether the forest earns its
complexity.

CV schemes
  spatial_residual  blocked k-fold; block size = range of the RESIDUAL
                    autocorrelation of the linear model fitted to all training
                    cells (Roberts et al. 2017): first distance band whose
                    autocorrelation < 0.1, rounded up to 50 km. PRIMARY --
                    C1 is judged here (proposal section 3).
  spatial_response  the same rule on the response itself (400 km). This was the
                    rule of the first run; revised because richness is
                    autocorrelated largely because the environment is, and
                    explaining that is the model's job. Kept as a sensitivity row.
  random            plain k-fold (fixed decision 5, level 1).

Writes outputs/tables/cv_results.csv, cv_folds.csv, rf_tuning.csv,
       response_correlogram.csv; data/processed/oof_predictions.parquet

Run: python src/fit_models.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spatial_cv as sc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
K_OUTER, K_INNER = 5, 4
RF_GRID = {"max_features": [0.2, 0.33, 0.66], "min_samples_leaf": [1, 5, 20]}
RF_TREES = 500
SEED = 0
CORR_EDGES_KM = [0, 10, 25, 50, 75, 100, 150, 200, 300, 400, 600]


# --- spatial structure ------------------------------------------------------
def correlogram(xy: np.ndarray, v: np.ndarray, edges_km=CORR_EDGES_KM) -> pd.DataFrame:
    """Moran-type autocorrelation by distance band: mean product of
    standardised values over pairs whose separation falls in the band."""
    z = (v - v.mean()) / v.std()
    d = pdist(xy) / 1000
    i, j = np.triu_indices(len(v), k=1)
    prod = z[i] * z[j]
    rows = []
    for lo, hi in zip(edges_km[:-1], edges_km[1:]):
        m = (d > lo) & (d <= hi)
        rows.append({"from_km": lo, "to_km": hi, "pairs": int(m.sum()),
                     "autocorrelation": float(prod[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def block_size_from(corr: pd.DataFrame, below: float = 0.1, step_km: int = 50) -> float:
    """First band upper edge where autocorrelation < `below`, rounded up to
    `step_km`, in metres."""
    hit = corr[corr["autocorrelation"] < below]
    edge = float(hit["to_km"].iloc[0]) if len(hit) else float(corr["to_km"].iloc[-1])
    return float(np.ceil(edge / step_km) * step_km * 1000)


def linear_residuals(X, y) -> np.ndarray:
    return np.asarray(y) - linear_model().fit(X, y).predict(X)


def residual_block_size(X, y, xy) -> tuple[float, pd.DataFrame]:
    """Block size from the autocorrelation the environment does NOT explain."""
    corr = correlogram(xy, linear_residuals(X, y))
    return block_size_from(corr), corr


# --- models -----------------------------------------------------------------
class NullModel:
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def linear_model():
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LinearRegression())


def rf_model(**params):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=RF_TREES, n_jobs=-1, random_state=SEED, **params)


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


class TunedRF:
    """Random forest whose grid is chosen by an inner blocked CV run on the
    training data it is given -- and nothing else."""

    def __init__(self, blocks_of_rows, k_inner=K_INNER, grid=RF_GRID, make=rf_model):
        self.blocks_of_rows, self.k_inner, self.grid, self.make = blocks_of_rows, k_inner, grid, make
        self.tuning_ = []

    def fit(self, X, y, rows):
        X, y = np.asarray(X), np.asarray(y)
        inner = sc.blocked_folds(self.blocks_of_rows[rows], self.k_inner, seed=SEED + 1)
        best, best_score = None, np.inf
        for values in itertools.product(*self.grid.values()):
            params = dict(zip(self.grid.keys(), values))
            pred = np.empty(len(y))
            for _, tr, te in sc.iter_folds(inner):
                pred[te] = self.make(**params).fit(X[tr], y[tr]).predict(X[te])
            score = rmse(y, pred)
            self.tuning_.append({**params, "inner_rmse": score})
            if score < best_score:
                best, best_score = params, score
        self.params_ = best
        self.model_ = self.make(**best).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X))


# --- cross-validation -------------------------------------------------------
def cross_validate(make_model, X, y, folds, needs_rows=False):
    """Out-of-fold predictions. Each fold's model is built fresh and fitted on
    that fold's training rows only."""
    X, y = np.asarray(X), np.asarray(y)
    oof = np.full(len(y), np.nan)
    fitted = []
    for f, tr, te in sc.iter_folds(folds):
        m = make_model()
        m.fit(X[tr], y[tr], tr) if needs_rows else m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
        fitted.append((f, m))
    return oof, fitted


def hotspot_recall(observed, predicted, top: float = 0.2) -> float:
    """C2: share of the truly top-quintile cells that are also in the predicted
    top quintile. Top sets are the k = round(top * n) highest values; ties are
    broken by position (stable), which matters only for exactly tied values."""
    o, p = np.asarray(observed, float), np.asarray(predicted, float)
    k = int(round(top * len(o)))
    true_top = set(np.argsort(-o, kind="stable")[:k])
    pred_top = set(np.argsort(-p, kind="stable")[:k])
    return len(true_top & pred_top) / k


def metrics(y, yhat, folds) -> dict:
    y, yhat = np.asarray(y), np.asarray(yhat)
    per_fold = []
    for _, _, te in sc.iter_folds(folds):
        per_fold.append(np.nan if np.ptp(yhat[te]) == 0 else spearmanr(y[te], yhat[te])[0])
    return {
        "spearman_rho": float(spearmanr(y, yhat)[0]) if np.ptp(yhat) > 0 else np.nan,
        "spearman_rho_fold_mean": float(np.nanmean(per_fold)) if not np.all(np.isnan(per_fold)) else np.nan,
        "spearman_rho_fold_min": float(np.nanmin(per_fold)) if not np.all(np.isnan(per_fold)) else np.nan,
        "rmse": rmse(y, yhat),
        "mae": float(np.mean(np.abs(y - yhat))),
        "r2": float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)),
    }


def main():
    from model_table import FROZEN, verify_frozen
    if FROZEN.exists():
        verify_frozen()
    table = pd.read_parquet(PROCESSED / "model_table.parquet")
    info = json.loads((PROCESSED / "predictors.json").read_text())
    preds, response = info["selected"], info["response"]
    train = table[table.train_eligible].reset_index(drop=True)
    X, y = train[preds].to_numpy(float), train[response].to_numpy(float)
    xy = train[["x", "y"]].to_numpy(float)

    meta = json.loads((ROOT / "data" / "interim" / "grid_meta.json").read_text())
    origin = (meta["x0"], meta["y0"])
    corr_resp = correlogram(xy, y)
    block_resp = block_size_from(corr_resp)
    block_res, corr_res = residual_block_size(X, y, xy)
    blocks_res = sc.block_ids(xy, block_res, origin=origin)
    blocks_resp = sc.block_ids(xy, block_resp, origin=origin)
    # (scheme, outer folds, blocks used for inner RF tuning, block size m)
    schemes = {
        "spatial_residual": (sc.blocked_folds(blocks_res, K_OUTER, seed=SEED), blocks_res, block_res),
        "spatial_response": (sc.blocked_folds(blocks_resp, K_OUTER, seed=SEED), blocks_resp, block_resp),
        "random": (np.random.default_rng(SEED).permutation(np.arange(len(y)) % K_OUTER), blocks_res, None),
    }
    TABLES.mkdir(parents=True, exist_ok=True)
    corr = corr_resp.rename(columns={"autocorrelation": "response"}).assign(
        linear_residual=corr_res["autocorrelation"].to_numpy())
    corr.to_csv(TABLES / "response_correlogram.csv", index=False)
    print(corr.round(3).to_string(index=False))
    print(f"\nresidual-based block {block_res / 1000:.0f} km ({len(np.unique(blocks_res))} blocks); "
          f"response-based block {block_resp / 1000:.0f} km ({len(np.unique(blocks_resp))} blocks)")

    rows, oof_cols, tuning, fold_rows = [], {}, [], []
    for scheme, (folds, blocks, block_m) in schemes.items():
        spatial = block_m is not None
        fold_rows += [{"scheme": scheme, "fold": int(f), "test_cells": int(len(te)),
                       "test_blocks": int(len(np.unique(blocks[te]))) if spatial else None}
                      for f, _, te in sc.iter_folds(folds)]
        desc = {"block_km": block_m / 1000 if spatial else None,
                "n_blocks": int(len(np.unique(blocks))) if spatial else None, "k_folds": K_OUTER}
        # inner RF tuning is blocked in every scheme (residual blocks for the random scheme)
        for name, make, rows_needed in (("null", NullModel, False), ("linear", linear_model, False),
                                        ("rf", lambda: TunedRF(blocks), True)):
            oof, fitted = cross_validate(make, X, y, folds, needs_rows=rows_needed)
            m = metrics(y, oof, folds)
            rows.append({"scheme": scheme, "model": name, **desc, **m})
            oof_cols[f"{scheme}_{name}"] = oof
            if name == "rf":
                tuning += [{"scheme": scheme, "fold": int(f), **mdl.params_} for f, mdl in fitted]
            print(f"   {scheme:16s} {name:6s} rho {m['spearman_rho']:.3f}  RMSE {m['rmse']:.2f}  R2 {m['r2']:.3f}",
                  flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(TABLES / "cv_results.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(TABLES / "cv_folds.csv", index=False)
    pd.DataFrame(tuning).to_csv(TABLES / "rf_tuning.csv", index=False)
    oof_df = pd.DataFrame({"cell_id": train.cell_id, "observed": y, **oof_cols,
                           **{f"fold_{s}": v[0] for s, v in schemes.items()}})
    oof_df.to_parquet(PROCESSED / "oof_predictions.parquet", index=False)
    with pd.option_context("display.width", 200):
        print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
