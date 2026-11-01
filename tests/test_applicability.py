"""Tests for src/applicability.py helpers.

Run: python tests/test_applicability.py   (or pytest)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import applicability as ap  # noqa: E402


def _cells():
    return pd.DataFrame({
        "has_records":    [False, True, True, True, True],
        "train_eligible": [False, False, True, False, False],
        "andaman_nicobar": [False, False, False, True, True],
        "coverage_full":  [np.nan, 0.5, 0.95, 0.95, 0.5],
        "T":              [np.nan, 3, 50, 40, 40],
        "aoa":            [True, False, True, False, True],
    })


def test_record_status_labels():
    s = ap.record_status(_cells()).tolist()
    assert s == ["no records", "records, poorly sampled", "training-eligible",
                 "Andaman & Nicobar (excluded)", "records, poorly sampled"], s


def test_c3_counts_only_non_eligible_cells():
    t = _cells()
    # non-eligible rows 0, 1, 3, 4 -> inside: True, False, False, True -> 2/4
    assert ap.c3_fraction(t["aoa"], t["train_eligible"]) == 0.5
    t.loc[2, "aoa"] = False                       # an eligible cell must not change C3
    assert ap.c3_fraction(t["aoa"], t["train_eligible"]) == 0.5


def test_breakdown_shares():
    t = _cells().assign(di=[0.1, 0.9, 0.0, 1.5, 0.2], grp=["a", "a", "b", "b", "b"])
    b = ap.breakdown(t, "grp").set_index("group")
    assert b.loc["a", "cells"] == 2 and b.loc["a", "share_inside"] == 0.5
    assert b.loc["b", "inside_aoa"] == 2 and np.isclose(b.loc["b", "median_di"], 0.2)


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
