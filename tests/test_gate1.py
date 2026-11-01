"""Tests for src/gate1.py.

Run: python tests/test_gate1.py   (or pytest)
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import gate1  # noqa: E402
from richness import coverage_at  # noqa: E402


def test_andaman_nicobar_flag_known_places():
    places = {"Port Blair": (92.73, 11.62, True), "Great Nicobar": (93.85, 7.0, True),
              "Chennai": (80.27, 13.08, False), "Kolkata": (88.36, 22.57, False),
              "Aizawl (Mizoram, east of 92E)": (92.72, 23.73, False),
              "Agartala": (91.28, 23.83, False)}
    got = gate1.andaman_nicobar([v[0] for v in places.values()], [v[1] for v in places.values()])
    assert got.tolist() == [v[2] for v in places.values()], dict(zip(places, got))


def test_split_rule_needs_coverage_and_guard():
    r = pd.DataFrame({"coverage_full": [1.0, 0.95, 0.95, 0.85], "T": [2, 50, 9, 50]})
    assert gate1.well_sampled(r, 0.90, 10).tolist() == [False, True, False, False]


def _pairs(seed=3, T=60, S=80):
    rng = np.random.default_rng(seed)
    p = np.clip(rng.lognormal(-2.5, 1.0, S), 0, 0.9)
    hits = rng.random((T, S)) < p
    u, s = np.nonzero(hits)
    return pd.DataFrame({"unit": [f"u{i}" for i in u], "species_id": s})


def test_guard_truth_is_coverage_of_a_further_unit():
    """coverage_at(Y, T, k) (Chao's rarefied coverage, leave-one-out) is the
    expected share of a FURTHER unit's incidences that belong to species already
    found in k units. Monte Carlo: draw k + 1 units, score the (k+1)th against
    the first k; ratio of sums over draws."""
    pairs = _pairs()
    Y = pairs.groupby("species_id").size()
    units = pairs.unit.unique()
    by_unit = {u: set(g) for u, g in pairs.groupby("unit")["species_id"]}
    rng = np.random.default_rng(0)
    for k in (3, 10, 30):
        hit = tot = 0
        for _ in range(20000):
            draw = rng.choice(units, size=k + 1, replace=False)
            found = set().union(*(by_unit[u] for u in draw[:k]))
            extra = by_unit[draw[k]]
            hit += len(extra & found)
            tot += len(extra)
        exact = coverage_at(Y.to_numpy(float), len(units), k)
        assert abs(hit / tot - exact) < 0.006, (k, hit / tot, exact)
        got = gate1.subsample_coverage(pairs, k, reps=2, rng=rng)
        assert all(abs(t - exact) < 1e-12 for _, t in got)


def test_estimate_error_shrinks_with_units():
    """The guard's rationale: with few units the coverage estimate is unreliable;
    its error must shrink as units are added and be small by k = 30."""
    pairs = _pairs(T=200, S=150)
    rng = np.random.default_rng(1)
    mae = {k: np.mean([abs(e - t) for e, t in gate1.subsample_coverage(pairs, k, 300, rng)])
           for k in (2, 10, 30)}
    assert mae[2] > mae[10] > mae[30] and mae[30] < 0.02, mae


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
