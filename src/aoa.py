"""Part 7 machinery: area of applicability (Meyer & Pebesma 2021), in Python.

A model can only be trusted where the new data resemble the training data in
the predictors that matter TO THE MODEL. So:

  1. Standardise predictors with the training mean and SD, then weight each by
     its permutation importance (negative importance -> 0). A variable the model
     ignores contributes no distance, however strange its values.
  2. Dissimilarity index DI of a new point = distance to the nearest training
     point / mean pairwise distance among training points.
  3. Training DI is cross-validated: each training point's distance to the
     nearest training point OUTSIDE ITS OWN FOLD (spatial folds go in here).
  4. Threshold = upper whisker of the training DI distribution (Tukey hinges,
     1.5 x IQR), as R's boxplot.stats -- CAST's default.
  5. Area of applicability: DI <= threshold.

Not validated against R's CAST (R is not installed here); behaviour is pinned by
construction tests in tests/test_aoa.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist


# --- importance -------------------------------------------------------------
def permutation_importance(predict, X, y, n_repeats: int = 10, seed: int = 0) -> np.ndarray:
    """Increase in RMSE when each column is permuted, averaged over repeats.
    `predict` is any callable X -> yhat, so the fitted model never needs to be
    a particular library's object."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    rng = np.random.default_rng(seed)
    rmse = lambda yhat: float(np.sqrt(np.mean((y - yhat) ** 2)))
    base = rmse(predict(X))
    imp = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        scores = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            scores.append(rmse(predict(Xp)) - base)
        imp[j] = np.mean(scores)
    return imp


def importance_weights(importance) -> np.ndarray:
    w = np.clip(np.asarray(importance, float), 0, None)
    if not np.any(w > 0):
        raise ValueError("all importances are <= 0; the AOA would be undefined")
    return w


# --- scaling and distances --------------------------------------------------
def weighted_scale(X_train, X_new, weights):
    X_train, X_new = np.asarray(X_train, float), np.asarray(X_new, float)
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)          # a constant predictor carries no information
    w = np.asarray(weights, float)
    return (X_train - mu) / sd * w, (X_new - mu) / sd * w


def mean_pairwise_distance(Z: np.ndarray, max_points: int = 8000, seed: int = 0) -> float:
    if len(Z) > max_points:                  # pdist is O(n^2) memory; a random subset estimates the mean
        Z = Z[np.random.default_rng(seed).choice(len(Z), max_points, replace=False)]
    return float(pdist(Z).mean())


def training_di(Z_train: np.ndarray, folds=None, dbar: float | None = None) -> np.ndarray:
    """Cross-validated DI of each training point: distance to the nearest
    training point in a different fold (folds=None: every other point)."""
    dbar = dbar or mean_pairwise_distance(Z_train)
    n = len(Z_train)
    if folds is None:
        folds = np.arange(n)
    folds = np.asarray(folds)
    d = np.full(n, np.inf)
    for f in np.unique(folds):
        inside, outside = folds == f, folds != f
        if outside.any():
            d[inside] = cKDTree(Z_train[outside]).query(Z_train[inside], k=1)[0]
    return d / dbar


def fivenum(x) -> np.ndarray:
    """R's fivenum (Tukey hinges), which boxplot.stats uses."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    n4 = np.floor((n + 3) / 2) / 2
    d = np.array([1, n4, (n + 1) / 2, n + 1 - n4, n]) - 1
    return 0.5 * (x[np.floor(d).astype(int)] + x[np.ceil(d).astype(int)])


def di_threshold(train_di) -> float:
    """Upper whisker: the largest training DI not beyond Q3 + 1.5 IQR."""
    x = np.asarray(train_di, float)
    x = x[np.isfinite(x)]
    _, q1, _, q3, _ = fivenum(x)
    return float(x[x <= q3 + 1.5 * (q3 - q1)].max())


@dataclass
class AOAResult:
    di: np.ndarray
    aoa: np.ndarray
    threshold: float
    train_di: np.ndarray
    dbar: float


def area_of_applicability(X_train, X_new, weights, folds=None) -> AOAResult:
    Z_train, Z_new = weighted_scale(X_train, X_new, weights)
    dbar = mean_pairwise_distance(Z_train)
    di = cKDTree(Z_train).query(Z_new, k=1)[0] / dbar
    tdi = training_di(Z_train, folds, dbar)
    thr = di_threshold(tdi)
    return AOAResult(di=di, aoa=di <= thr, threshold=thr, train_di=tdi, dbar=dbar)
