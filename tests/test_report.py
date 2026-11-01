"""Tests for src/findings.py: the findings document must match the tables.

Run: python tests/test_report.py   (or pytest)
"""
import json
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import findings as fd  # noqa: E402


def test_md_table_known_output():
    df = pd.DataFrame({"a": ["x", "y"], "n": [1200, 3], "v": [0.12345, float("nan")]})
    assert fd.md(df) == "| a | n | v |\n|---|---|---|\n| x | 1,200 | 0.123 |\n| y | 3 |  |"


def test_claude_sections_are_verbatim():
    text = fd.claude_sections(["Traps already discovered"])
    assert text.startswith("## Traps already discovered")
    assert text in (fd.ROOT / "CLAUDE.md").read_text()
    assert "## Parts" not in text


def _doc():
    if not (fd.T / "cv_results.csv").exists():
        raise unittest.SkipTest("outputs not built")
    return fd.build()


def test_criteria_rows_match_source_tables():
    doc = _doc()
    cv = pd.read_csv(fd.T / "cv_results.csv", keep_default_na=False, na_values=[""])
    rho = cv.query("scheme == 'spatial_residual' and model == 'rf'").spearman_rho.iloc[0]
    c2 = pd.read_csv(fd.T / "c2_recall.csv").query("scheme == 'spatial_residual' and model == 'rf'").c2_recall.iloc[0]
    c3 = json.loads((fd.T / "aoa_report.json").read_text())["c3"]
    crit = doc.split("## 1. Pre-registered criteria")[1].split("## 2.")[0]
    assert f"| {rho:.3f} (100 km" in crit and f"| {c2:.3f} (" in crit and f"| {c3:.3f} (" in crit


def test_no_missing_values_or_lost_model_names():
    doc = _doc()
    body = doc.split("## Appendix")[0]
    assert "nan" not in body.lower().replace("nancowry", ""), "a NaN leaked into the findings"
    assert body.count("| null |") == 3, "null-model rows lost (pandas reads 'null' as NaN)"


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
