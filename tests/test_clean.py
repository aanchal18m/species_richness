"""Tests for src/clean.py.

Run: python tests/test_clean.py   (or pytest)

The audit table is a deliverable: each row must report what its step removed.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clean import clean_occurrences  # noqa: E402

BASE = dict(decimalLatitude=20.0, decimalLongitude=78.0, coordinateUncertaintyInMeters=None,
            basisOfRecord="HUMAN_OBSERVATION", occurrenceStatus="PRESENT", year=2016,
            taxonRank="SPECIES", speciesKey=100, eventDate="2016-01-10")


def _records():
    originals = [{"speciesKey": 100 + i, "eventDate": f"2016-01-{10 + i:02d}"} for i in range(5)]
    # 7 true duplicates: the SAME observer (obsr1) repeating a record, spread over
    # the originals; columns outside the rule (here occurrenceID) may differ.
    dups = [{**originals[i], "occurrenceID": f"dup{k}"} for k, i in enumerate((0, 0, 0, 1, 2, 2, 4))]
    # different observers at the same point-day (an eBird hotspot): independent
    # observations, all must survive
    cross = [{**originals[0], "recordedBy": "obsr2"}, {**originals[0], "recordedBy": "obsr3"},
             {**originals[3], "recordedBy": "obsr2"}]
    # near-duplicates: each differs from originals[0] in exactly one rule column
    near = [{**originals[0], "eventDate": "2016-01-30"},
            {**originals[0], "speciesKey": 999},
            {**originals[0], "decimalLatitude": 20.001},
            {**originals[0], "decimalLongitude": 78.001}]
    rows = [{**BASE, "recordedBy": "obsr1", "occurrenceID": "o", **r} for r in originals + dups + cross + near]
    return pd.DataFrame(rows), len(originals) + len(cross) + len(near), len(dups)


def test_duplicate_audit_reports_the_real_count():
    df, n_unique, n_dups = _records()
    clean, audit = clean_occurrences(df)
    row = audit.set_index("filter").loc["exact duplicate records"]
    assert len(clean) == n_unique, len(clean)
    assert row["dropped"] == n_dups, f"audit says {row['dropped']} dropped, {n_dups} were planted"
    assert row["kept"] == n_unique


def test_cross_observer_records_at_one_point_day_survive():
    df, *_ = _records()
    clean, _ = clean_occurrences(df)
    first = clean[(clean.speciesKey == 100) & (clean.eventDate == "2016-01-10")
                  & (clean.decimalLatitude == 20.0) & (clean.decimalLongitude == 78.0)]
    assert sorted(first.recordedBy) == ["obsr1", "obsr2", "obsr3"], first.recordedBy.tolist()


def test_audit_rows_chain():
    """Every row's kept = previous kept - dropped, so the table adds up."""
    df, *_ = _records()
    df = pd.concat([df, pd.DataFrame([{**BASE, "occurrenceStatus": "ABSENT", "speciesKey": 555},
                                      {**BASE, "taxonRank": "GENUS", "speciesKey": None}])],
                   ignore_index=True)
    _, audit = clean_occurrences(df)
    kept, dropped = audit["kept"].tolist(), audit["dropped"].tolist()
    for i in range(1, len(audit)):
        assert kept[i] == kept[i - 1] - dropped[i], audit.iloc[i].to_dict()
    assert kept[-1] == kept[0] - sum(dropped)


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
