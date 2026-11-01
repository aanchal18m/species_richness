"""Tests for src/aoa.py.

Run: python tests/test_aoa.py   (or pytest)

No model is fitted: the "model" is the analytic function f(x) = 3*x1 + x2, whose
permutation importances are known exactly, and x3 is irrelevant by construction.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import aoa  # noqa: E402


def f(X):
    return 3 * X[:, 0] + X[:, 1]


def _train(n=400, seed=0):
    return np.random.default_rng(seed).normal(0, 1, (n, 3))


def test_permutation_importance_known_answer():
    """y = b*x with x ~ N(0,1): permuting x_j gives RMSE |b_j| * sqrt(2)."""
    X = _train(20000)
    imp = aoa.permutation_importance(f, X, f(X), n_repeats=3)
    assert np.allclose(imp, [3 * math.sqrt(2), math.sqrt(2), 0.0], atol=0.05), imp
    assert imp[2] == 0.0                                  # exactly: f never reads x3


def test_identical_point_is_inside():
    X = _train()
    w = aoa.importance_weights(aoa.permutation_importance(f, X, f(X)))
    res = aoa.area_of_applicability(X, X[:25], w)
    assert np.all(res.di == 0) and res.aoa.all()


def test_near_cluster_inside_distant_cluster_outside():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (400, 3))
    near = rng.normal(0, 0.3, (50, 3))
    far = rng.normal(0, 0.3, (50, 3)) + np.array([12.0, 12.0, 0.0])
    w = aoa.importance_weights(aoa.permutation_importance(f, X, f(X)))
    res = aoa.area_of_applicability(X, np.vstack([near, far]), w)
    assert res.aoa[:50].all(), res.di[:50].max()
    assert not res.aoa[50:].any(), res.di[50:].min()


def test_irrelevant_predictor_cannot_push_points_outside():
    """x3 has zero importance; wild x3 values at prediction time must not matter.
    The unweighted control (all weights 1) must fail on the same points."""
    X = _train()
    new = X[:40].copy()
    new[:, 2] = 1e6
    w = aoa.importance_weights(aoa.permutation_importance(f, X, f(X)))
    assert w[2] == 0
    weighted = aoa.area_of_applicability(X, new, w)
    assert weighted.aoa.all(), weighted.di.max()
    unweighted = aoa.area_of_applicability(X, new, np.ones(3))
    assert not unweighted.aoa.any(), "control: unweighted AOA should reject every wild point"


def test_threshold_matches_r_boxplot_stats():
    """R: boxplot.stats(c(1:10, 100))$stats[5] == 10 (hinges 3.5 and 8.5, fence 16).
    R: fivenum(c(1:10, 100)) == c(1, 3.5, 6, 8.5, 100)."""
    x = np.r_[np.arange(1, 11), 100.0]
    assert np.allclose(aoa.fivenum(x), [1, 3.5, 6, 8.5, 100])
    # R: fivenum(1:10) == c(1, 3, 5.5, 8, 10); numpy's default quartiles give 3.25 / 7.75
    assert np.allclose(aoa.fivenum(np.arange(1, 11.0)), [1, 3, 5.5, 8, 10])
    assert aoa.di_threshold(x) == 10
    assert aoa.di_threshold(np.arange(1, 11.0)) == 10   # no outlier: the maximum


def test_training_di_respects_folds():
    """Two points 0.1 apart in the same fold must each find their nearest point
    in another fold, so their CV training DI is larger than under LOO."""
    Z = np.array([[0.0, 0], [0.1, 0], [5, 0], [5, 5], [0, 5]])
    folds = np.array([0, 0, 1, 2, 3])
    dbar = aoa.mean_pairwise_distance(Z)
    loo = aoa.training_di(Z, None, dbar)
    cv = aoa.training_di(Z, folds, dbar)
    assert np.isclose(loo[0] * dbar, 0.1) and np.isclose(cv[0] * dbar, 5.0)
    assert np.allclose(loo[2:], cv[2:])


def test_constant_predictor_does_not_divide_by_zero():
    X = _train()
    X[:, 2] = 7.0
    res = aoa.area_of_applicability(X, X[:5], np.ones(3))
    assert np.isfinite(res.di).all() and res.aoa.all()


if __name__ == "__main__":
    tests = [(n, fn) for n, fn in list(globals().items()) if n.startswith("test_")]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{passed + failed} checks passed")
    sys.exit(1 if failed else 0)
