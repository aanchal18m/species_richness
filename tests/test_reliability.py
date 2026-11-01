"""Tests for src/reliability.py.

Run: python tests/test_reliability.py   (or pytest)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import reliability as rl  # noqa: E402


def test_spearman_brown_and_ceiling_known_answers():
    assert np.isclose(rl.spearman_brown(0.5), 2 / 3)
    assert rl.spearman_brown(1.0) == 1.0 and rl.spearman_brown(0.0) == 0.0
    assert np.isclose(rl.c1_ceiling(0.81), 0.9)
    assert rl.c1_ceiling(-0.1) == 0.0


def test_split_units_is_a_partition_and_reproducible():
    units = np.arange(41)
    a, b = rl.split_units(units, np.random.default_rng([1, 0]))
    assert len(a) == 20 and len(b) == 21
    assert set(a) | set(b) == set(units) and not set(a) & set(b)
    a2, _ = rl.split_units(units, np.random.default_rng([1, 0]))
    assert np.array_equal(a, a2)


def test_simulated_recall_limits_and_monotone():
    n = 1000
    assert rl.simulated_recall(1.0, n, draws=20) > 0.99
    assert abs(rl.simulated_recall(0.0, n, draws=200) - 0.2) < 0.01
    assert rl.simulated_recall(0.5, n, draws=100) < rl.simulated_recall(0.8, n, draws=100)


def test_reliability_recovers_known_signal_to_noise():
    """Synthetic cells: each half's measurement = truth + independent noise of
    equal variance -> r_half = var_true / (var_true + var_noise) = 0.5, and
    Spearman-Brown then gives the full-depth value 2/3."""
    from scipy.stats import pearsonr
    rng = np.random.default_rng(0)
    truth = rng.normal(size=20000)
    a, b = truth + rng.normal(size=20000), truth + rng.normal(size=20000)
    r_half = pearsonr(a, b)[0]
    assert abs(r_half - 0.5) < 0.02
    full = truth + rng.normal(size=20000) / np.sqrt(2)          # full depth: half the noise variance
    assert abs(rl.spearman_brown(r_half) - pearsonr(full, truth + rng.normal(size=20000) / np.sqrt(2))[0]) < 0.02


def test_half_richness_on_identical_halves_agrees():
    pairs = pd.DataFrame({"unit": np.repeat(np.arange(20), 3), "species_id": np.tile([1, 2, 3], 20)})
    r1 = rl.half_richness(pairs, np.arange(10))
    r2 = rl.half_richness(pairs, np.arange(10, 20))
    assert r1 == r2


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
