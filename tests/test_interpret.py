"""Tests for src/interpret.py helpers.

Run: python tests/test_interpret.py   (or pytest)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import interpret as ip  # noqa: E402


def test_additivity_check_rejects_values_that_do_not_sum():
    sv = np.array([[1.0, 2.0], [0.5, -0.5]])
    ip.check_additivity(sv, 10.0, np.array([13.0, 10.0]))
    try:
        ip.check_additivity(sv, 10.0, np.array([13.0, 10.5]))
    except AssertionError:
        pass
    else:
        raise AssertionError("non-additive SHAP accepted")


def test_importance_comparison_shares_and_ranks():
    S = pd.DataFrame({"a": [3, -3], "b": [1, -1], "c": [0.0, 0.0]})
    perm = pd.Series({"a": 1.0, "b": 4.0, "c": -0.2})          # negative -> 0
    cmp = ip.importance_comparison(S, perm)
    assert np.isclose(cmp.shap_share.sum(), 1) and np.isclose(cmp.perm_share.sum(), 1)
    assert cmp.loc["a", "shap_rank"] == 1 and cmp.loc["b", "perm_rank"] == 1
    assert cmp.loc["c", "perm_share"] == 0
    assert cmp.loc["a", "rank_shift"] == 1                       # 2nd by permutation, 1st by SHAP


def test_interior_peak_detects_a_hump_not_a_trend():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, 4000)
    hump = ip.binned_curve(x, -(x - 5) ** 2 + rng.normal(0, 0.3, 4000))
    p = ip.interior_peak(hump, min_drop=3)
    assert p["interior_peak"] and 4 < p["peak_x"] < 6, p
    trend = ip.interior_peak(ip.binned_curve(x, 2 * x + rng.normal(0, 0.3, 4000)), min_drop=3)
    assert not trend["interior_peak"], trend
    edge = ip.interior_peak(ip.binned_curve(x, -(x - 9.6) ** 2), min_drop=3)   # peak in the last bin
    assert not edge["interior_peak"], edge


def test_binned_curve_handles_tied_values():
    x = np.r_[np.zeros(800), np.linspace(0.1, 1, 200)]          # 80% zeros, like rare land cover
    c = ip.binned_curve(x, x * 2)
    assert c["n"].sum() == 1000 and len(c) < 10                  # tied edges merge bins


def test_plot_curve_reaches_the_sparse_tail():
    """A decline confined to the top 3% of x must show in the drawn line."""
    x = np.r_[np.linspace(0, 1000, 970), np.linspace(2000, 5000, 30)]
    s = np.where(x > 1500, -6.0, 2.0)
    c = ip.plot_curve(x, s)
    assert c.x_median.max() > 3000 and c.shap_median.iloc[-1] == -6.0
    assert ip.binned_curve(x, s).shap_median.iloc[-1] > -6.0     # the decile view blurs it


def test_block_bootstrap_interval():
    rng = np.random.default_rng(1)
    a = rng.normal(size=2000)
    blocks = np.repeat(np.arange(200), 10)
    rho, lo, hi = ip.block_bootstrap_spearman(a, rng.normal(size=2000), blocks, n_boot=300)
    assert lo < 0 < hi, (rho, lo, hi)
    rho, lo, hi = ip.block_bootstrap_spearman(a, 3 * a, blocks, n_boot=300)
    assert rho == 1 and lo > 0.99


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
