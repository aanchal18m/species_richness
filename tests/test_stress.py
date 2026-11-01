"""Tests for Part 6 (src/stress_test.py, fit_models.hotspot_recall) and the
Gate 2 freeze (model_table.freeze / verify_frozen).

Run: python tests/test_stress.py   (or pytest)
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import fit_models as fm  # noqa: E402
import model_table as mt  # noqa: E402
import spatial_cv as sc  # noqa: E402
import stress_test as st  # noqa: E402


# --- C2 ----------------------------------------------------------------------
def test_hotspot_recall_known_answers():
    obs = np.arange(1, 101, dtype=float)                  # top quintile = 81..100
    assert fm.hotspot_recall(obs, obs) == 1.0
    assert fm.hotspot_recall(obs, -obs) == 0.0
    # hand case, n = 10, k = 2: true top {9, 8}; predicted top {9, 0} -> 1 of 2
    o = np.arange(10, dtype=float)
    p = o.copy()
    p[8], p[0] = -1, 100
    assert fm.hotspot_recall(o, p) == 0.5
    # a monotone transform of the truth is still perfect: recall is rank-based
    assert fm.hotspot_recall(obs, np.log(obs) * 3 + 7) == 1.0


def test_hotspot_recall_random_ranking_is_about_one_fifth():
    rng = np.random.default_rng(0)
    obs = rng.normal(size=1000)
    r = [fm.hotspot_recall(obs, rng.normal(size=1000)) for _ in range(400)]
    assert abs(np.mean(r) - 0.2) < 0.01, np.mean(r)


# --- downsampling --------------------------------------------------------------
class Spy:
    calls = []

    def fit(self, X, y, rows):
        Spy.calls.append(np.asarray(rows))
        self.m = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.m)


def _setup(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 1_000_000, (n, 2))
    blocks = sc.block_ids(xy, 100_000)
    folds = sc.blocked_folds(blocks, 5)
    return rng.normal(size=(n, 3)), rng.normal(size=n), folds, blocks


def test_downsampling_thins_training_side_only():
    X, y, folds, blocks = _setup()
    Spy.calls = []
    oof, used = st.downsampled_cv(X, y, folds, blocks, 0.25, 25, 0, make=Spy)
    assert not np.isnan(oof).any()                         # every held-out cell is still predicted
    for (f, tr, te), rows in zip(sc.iter_folds(folds), Spy.calls):
        assert len(rows) == round(0.25 * len(tr))
        assert set(rows) <= set(tr) and not set(rows) & set(te)
        assert np.array_equal(rows, used[int(f)])


def test_downsampling_is_reproducible_and_replicates_differ():
    X, y, folds, blocks = _setup()
    _, a = st.downsampled_cv(X, y, folds, blocks, 0.5, 50, 1, make=Spy)
    _, b = st.downsampled_cv(X, y, folds, blocks, 0.5, 50, 1, make=Spy)
    _, c = st.downsampled_cv(X, y, folds, blocks, 0.5, 50, 2, make=Spy)
    assert all(np.array_equal(a[f], b[f]) for f in a)
    assert any(not np.array_equal(a[f], c[f]) for f in a)


# --- Gate 2 freeze -------------------------------------------------------------
def test_freeze_detects_any_change():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        table, preds, frozen = d / "model_table.parquet", d / "predictors.json", d / "FROZEN.json"
        pd.DataFrame({"a": [1, 2]}).to_parquet(table)
        preds.write_text('{"selected": ["a"]}')
        mt.freeze("test", frozen, files=(table, preds))
        mt.verify_frozen(frozen, d)                        # unchanged: passes
        preds.write_text('{"selected": ["a", "b"]}')
        try:
            mt.verify_frozen(frozen, d)
        except RuntimeError as e:
            assert "predictors.json" in str(e)
        else:
            raise AssertionError("a changed predictor list passed verification")


def test_model_table_refuses_to_rebuild_when_frozen():
    with tempfile.TemporaryDirectory() as d:
        original = mt.FROZEN
        mt.FROZEN = Path(d) / "FROZEN.json"
        mt.FROZEN.write_text("{}")
        try:
            mt.main()
        except SystemExit as e:
            assert "frozen" in str(e)
        else:
            raise AssertionError("frozen model table was rebuilt")
        finally:
            mt.FROZEN = original


if __name__ == "__main__":
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_")]
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
