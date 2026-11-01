"""Tests for src/model_table.py and src/fit_models.py.

Run: python tests/test_models.py   (or pytest)
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import fit_models as fm  # noqa: E402
import model_table as mt  # noqa: E402
import spatial_cv as sc  # noqa: E402


# --- model table -------------------------------------------------------------
def test_train_eligible_rule():
    df = pd.DataFrame({"coverage_full": [0.90, 0.95, 0.89, 0.95, 0.95, np.nan],
                       "T": [20, 100, 100, 19, 100, np.nan],
                       "andaman_nicobar": [False, False, False, False, True, False]})
    assert mt.train_eligible(df).tolist() == [True, True, False, False, False, False]


def test_vif_known_answers():
    """X3 = X1 + X2 + e, sd(e) = 0.1: VIF3 = (2 + s^2)/s^2 = 201 and
    VIF1 = VIF2 = (1 + s^2)/s^2 = 101; an independent X4 has VIF 1."""
    rng = np.random.default_rng(0)
    n, s = 40000, 0.1
    x1, x2, x4 = rng.normal(size=(3, n))
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x1 + x2 + rng.normal(0, s, n), "x4": x4})
    v = mt.vif(X)
    assert abs(v["x3"] / 201 - 1) < 0.05 and abs(v["x1"] / 101 - 1) < 0.05, v.round(1).to_dict()
    assert abs(v["x4"] - 1) < 0.01
    exact = X.assign(x5=X.x1 - X.x4)
    assert np.isinf(mt.vif(exact)["x5"])


def test_vif_screen_drops_the_collinear_column_and_stops():
    rng = np.random.default_rng(1)
    x1, x2, x4 = rng.normal(size=(3, 5000))
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x1 + x2 + rng.normal(0, 0.1, 5000), "x4": x4})
    kept, log = mt.vif_screen(X, threshold=10)
    assert [r["dropped"] for r in log] == ["x3"] and kept == ["x1", "x2", "x4"]
    assert (mt.vif(X[kept]) <= 10).all()


def test_bio12_dropped_in_place_of_bio16():
    """bio16 carries the largest VIF (it is explained by bio12 and a third
    variable), yet the rule keeps bio16 and drops the clipped bio12."""
    rng = np.random.default_rng(2)
    bio12, z, other = rng.normal(size=(3, 5000))
    X = pd.DataFrame({"bio12": bio12, "bio16": bio12 + 0.6 * z + rng.normal(0, 0.05, 5000),
                      "z": z, "other": other})
    assert mt.vif(X).idxmax() == "bio16"
    kept, log = mt.vif_screen(X, threshold=10)
    assert "bio16" in kept and "bio12" not in kept, (kept, log)
    assert log[0]["largest"] == "bio16" and log[0]["dropped"] == "bio12"


def test_elev_mean_protected_next_largest_dropped():
    """elev_mean carries the largest VIF; the rule keeps it and drops the
    next-largest instead."""
    rng = np.random.default_rng(5)
    t1, t2, other = rng.normal(size=(3, 5000))
    X = pd.DataFrame({"elev_mean": t1 + 0.5 * t2 + rng.normal(0, 0.05, 5000),
                      "t1": t1, "t2": t2, "other": other})
    assert mt.vif(X).idxmax() == "elev_mean"
    kept, log = mt.vif_screen(X, threshold=10)
    assert "elev_mean" in kept and log[0]["largest"] == "elev_mean" and log[0]["dropped"] != "elev_mean", log
    assert (mt.vif(X[kept]) <= 10).all()


def test_elev_mean_alone_above_threshold_stops_the_screen():
    """If elev_mean is the only predictor above the threshold, dropping the
    next-largest would remove a predictor that is not collinear: stop, keep
    everything, and let the final VIF of elev_mean (> 10) be reported."""
    rng = np.random.default_rng(6)
    xs = rng.normal(size=(8, 5000))
    X = pd.DataFrame({f"x{i}": xs[i] for i in range(8)})
    X["elev_mean"] = xs.sum(axis=0) + rng.normal(0, 0.5, 5000)   # VIF ~33; each x ~5
    v = mt.vif(X)
    assert v["elev_mean"] > 10 and v.drop("elev_mean").max() < 10
    kept, log = mt.vif_screen(X, threshold=10)
    assert kept == list(X.columns) and log == []


def test_landcover_reference_breaks_closure_in_practice():
    """The principle behind a substantial reference class: dropping it leaves a
    remainder that varies (VIFs ~1); dropping a near-empty class leaves the rest
    summing to ~1, far past the screening threshold, so the screen would then
    remove a major class."""
    assert mt.LANDCOVER_REFERENCE == "lc_crop" and mt.LANDCOVER_REFERENCE in mt.LANDCOVER
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.dirichlet([20, 10, 5, 0.05], size=3000), columns=["crop", "tree", "grass", "snow"])
    assert np.isinf(mt.vif(X)).all()
    assert mt.vif(X.drop(columns="crop")).max() < 2
    assert mt.vif(X.drop(columns="snow")).max() > mt.VIF_MAX


# --- cross-validation --------------------------------------------------------
def _spatial_data(seed=0, n=1200):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 1_000_000, (n, 2))
    x1, x2, x3 = rng.normal(size=(3, n))
    y = 50 + 10 * x1 - 5 * x2 + rng.normal(0, 1, n)
    return xy, np.column_stack([x1, x2, x3]), y


class Spy:
    """Wraps a model; records the training rows it is fitted on."""
    seen = []

    def __init__(self, inner):
        self.inner = inner

    def fit(self, X, y, *rows):
        Spy.seen.append(set(map(tuple, np.round(X, 9))))
        self.inner.fit(X, y)
        return self

    def predict(self, X):
        return self.inner.predict(X)


def test_no_test_row_is_ever_used_for_fitting():
    xy, X, y = _spatial_data()
    folds = sc.blocked_folds(sc.block_ids(xy, 100_000), 5)
    Spy.seen = []
    fm.cross_validate(lambda: Spy(fm.linear_model()), X, y, folds)
    for (f, tr, te), seen in zip(sc.iter_folds(folds), Spy.seen):
        assert not (set(map(tuple, np.round(X[te], 9))) & seen), f"fold {f} trained on its test rows"


def test_tuned_rf_tunes_on_training_rows_only():
    xy, X, y = _spatial_data(n=500)
    blocks = sc.block_ids(xy, 150_000)
    folds = sc.blocked_folds(blocks, 4)
    fitted_on = []

    def tiny_rf(**params):
        class R:
            def fit(self, Xf, yf):
                fitted_on.append(set(map(tuple, np.round(Xf, 9))))
                self.m = fm.rf_model(**params).set_params(n_estimators=10).fit(Xf, yf)
                return self

            def predict(self, Xp):
                return self.m.predict(Xp)
        return R()

    grid = {"max_features": [0.5], "min_samples_leaf": [1, 5]}
    for f, tr, te in sc.iter_folds(folds):
        fitted_on.clear()
        fm.TunedRF(blocks, k_inner=3, grid=grid, make=tiny_rf).fit(X[tr], y[tr], tr)
        test_rows = set(map(tuple, np.round(X[te], 9)))
        assert len(fitted_on) == 2 * 3 + 1                  # grid x inner folds + final refit
        assert all(not (s & test_rows) for s in fitted_on), f"fold {f}: tuning saw test rows"


def test_null_and_linear_known_behaviour():
    xy, X, y = _spatial_data()
    folds = sc.blocked_folds(sc.block_ids(xy, 100_000), 5)
    oof_null, fitted = fm.cross_validate(fm.NullModel, X, y, folds)
    for (f, tr, te), (_, m) in zip(sc.iter_folds(folds), fitted):
        assert np.allclose(oof_null[te], y[tr].mean())
    m_null = fm.metrics(y, oof_null, folds)
    assert np.isnan(m_null["spearman_rho_fold_mean"]) and m_null["r2"] <= 0
    oof_lin, _ = fm.cross_validate(fm.linear_model, X, y, folds)
    m_lin = fm.metrics(y, oof_lin, folds)
    assert m_lin["spearman_rho"] > 0.98 and m_lin["r2"] > 0.98, m_lin


def test_residual_blocks_ignore_autocorrelation_the_environment_explains():
    """Richness autocorrelated only because the environment is: the response
    correlogram demands big blocks, the residuals of a model that captures the
    environment are uncorrelated, so the residual rule gives the minimum."""
    rng = np.random.default_rng(7)
    xy = rng.uniform(0, 1_000_000, (1500, 2))
    env = np.column_stack([np.sin(xy[:, 0] / 160_000), np.cos(xy[:, 1] / 200_000)])
    y = 100 + 30 * env[:, 0] + 20 * env[:, 1] + rng.normal(0, 2, 1500)
    assert fm.block_size_from(fm.correlogram(xy, y)) >= 150_000
    block, corr = fm.residual_block_size(env, y, xy)
    assert block == 50_000, corr
    assert corr["autocorrelation"].abs().max() < 0.1


def test_correlogram_and_block_size():
    rng = np.random.default_rng(4)
    xy = rng.uniform(0, 1_000_000, (1500, 2))
    noise = rng.normal(size=1500)
    c = fm.correlogram(xy, noise)
    assert c["autocorrelation"].abs().max() < 0.1, c
    # smooth field: a broad sine wave across 1000 km -- correlated at short range
    smooth = np.sin(xy[:, 0] / 160_000) + 0.1 * rng.normal(size=1500)
    cs = fm.correlogram(xy, smooth)
    assert cs.loc[cs.to_km == 25, "autocorrelation"].iloc[0] > 0.8
    assert fm.block_size_from(cs) >= 100_000
    assert fm.block_size_from(c) == 50_000          # first band (0-10 km) already < 0.1 -> rounded up


if __name__ == "__main__":
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_")]
    passed = failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"PASS  {name}")
        except unittest.SkipTest as e:
            skipped += 1
            print(f"SKIP  {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{passed + failed} checks passed" + (f", {skipped} skipped" if skipped else ""))
    sys.exit(1 if failed else 0)
