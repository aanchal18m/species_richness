"""Tests for the N-sensitivity step: the parameterised assembly must reproduce
the frozen N = 3 table exactly, and other N must change only the response side.

Run: python tests/test_n_sensitivity.py   (or pytest)
"""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import model_table as mt  # noqa: E402


def _need():
    if not mt.TABLE.exists():
        raise unittest.SkipTest("frozen model table not built")


def test_default_assembly_reproduces_the_frozen_table():
    _need()
    pd.testing.assert_frame_equal(mt.assemble(), pd.read_parquet(mt.TABLE), check_dtype=False)


def test_other_n_changes_only_the_response_side():
    _need()
    a, b = mt.assemble(3), mt.assemble(10)
    preds = [c for c in a.columns if c.startswith(("bio", "elev", "lc_", "slope"))]
    pd.testing.assert_frame_equal(a[["cell_id"] + preds], b[["cell_id"] + preds])
    assert (a.richness_c090.fillna(-1) != b.richness_c090.fillna(-1)).any()
    assert mt.train_eligible(b).equals(b.train_eligible)


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
