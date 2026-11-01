"""Tests for src/build_richness.py.

Run: python tests/test_build_richness.py   (or pytest)

Known answers by construction: each artefact planted in a synthetic chunk must
be removed by exactly its own filter; the N filter is checked against
hand-computed Y and T; and a simulated community with known detection
probabilities checks coverage end to end, records -> units -> estimate.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import build_richness as br  # noqa: E402
from incidence import build_incidence  # noqa: E402


# --- winters and dates --------------------------------------------------------
def test_winter_assignment_known_answers():
    cases = {  # (year, month): kept?
        (2015, 1): False, (2015, 2): False,      # orphaned tail of winter 2014/15
        (2015, 11): True, (2016, 2): True,       # first whole winter 2015/16
        (2023, 12): True, (2024, 1): True, (2024, 2): True,   # last whole winter 2023/24
        (2024, 11): False, (2024, 12): False,    # orphaned head of winter 2024/25
    }
    df = pd.DataFrame([(y, m) for y, m in cases], columns=["year", "month"])
    got = br.whole_winters(df).tolist()
    assert got == list(cases.values()), dict(zip(cases, got))
    assert br.winter_end_year([2019], [12]).iloc[0] == 2020
    assert br.winter_end_year([2020], [1]).iloc[0] == 2020


def test_single_day_dates():
    s = pd.Series(["2016-01-10", "2016-01-10T07:30:00", "2016-01", "2016",
                   "2016-01-10/2016-01-12", "2016-01-10/2016-01-10", None, ""])
    assert br.single_day(s).tolist() == [True, True, False, False, False, True, False, False]


# --- one synthetic chunk, one artefact per filter -------------------------------
def _toy_grid():
    import geopandas as gpd
    from shapely.geometry import box
    from grid import INDIA_AEA, build_grid
    land = gpd.GeoDataFrame(geometry=[box(0, 0, 30_000, 30_000)], crs=INDIA_AEA)
    grid, meta = build_grid(land, cell_km=10, min_land_fraction=0.5)
    return grid, meta


def _lonlat(meta, x, y):
    import pyproj
    t = pyproj.Transformer.from_crs(meta["crs"], "EPSG:4326", always_xy=True)
    return t.transform(x, y)


def _raw_chunk(meta):
    lon, lat = _lonlat(meta, 5_000, 5_000)        # centre of cell (0,0)
    olon, olat = _lonlat(meta, 95_000, 95_000)    # outside the toy grid
    base = dict(gbifID="0", datasetKey="ds", occurrenceID="o", eventID=None, recordedBy="obsr1",
                eventDate="2016-01-10", year="2016", month="1", day="10",
                decimalLatitude=str(lat), decimalLongitude=str(lon),
                coordinateUncertaintyInMeters=None, speciesKey="100", species="Sp",
                taxonRank="SPECIES", basisOfRecord="HUMAN_OBSERVATION", occurrenceStatus="PRESENT")
    plant = {
        "good_a": {},
        "good_b": {"speciesKey": "101", "eventDate": "2016-01-10T06:00:00"},
        "good_nov": {"speciesKey": "102", "year": "2015", "month": "11", "eventDate": "2015-11-20"},
        "dup_of_a": {},                                                     # exact duplicate
        "absent": {"speciesKey": "103", "occurrenceStatus": "ABSENT"},
        "coarse": {"speciesKey": "104", "coordinateUncertaintyInMeters": "20000"},
        "genus": {"speciesKey": None, "taxonRank": "GENUS"},
        "orphan_jan15": {"speciesKey": "105", "year": "2015", "month": "1", "eventDate": "2015-01-05"},
        "orphan_dec24": {"speciesKey": "106", "year": "2024", "month": "12", "eventDate": "2024-12-05"},
        "month_only": {"speciesKey": "107", "eventDate": "2016-01"},
        "outside": {"speciesKey": "108", "decimalLatitude": str(olat), "decimalLongitude": str(olon)},
    }
    rows = [{**base, **v, "gbifID": k} for k, v in plant.items()]
    return pd.DataFrame(rows).astype(object).where(lambda d: d.notna(), None)


def test_each_planted_artefact_removed_by_its_own_filter():
    grid, meta = _toy_grid()
    kept, rows = br.clean_chunk(_raw_chunk(meta), meta, set(grid.cell_id))
    audit = dict(rows)
    expect_one = {
        "georeferencing finer than 10 km": "coarse",
        "not flagged absent": "absent",
        "resolved to species rank": "genus",
        "whole winters": "orphans",
        "eventDate resolves to a single day": "month_only",
        "falls in a grid cell": "outside",
        "exact duplicate records": "dup_of_a",
    }
    for prefix in expect_one:
        name = next(k for k in audit if k.startswith(prefix))
        want = 2 if prefix == "whole winters" else 1
        assert audit[name] == want, (name, audit[name])
    # the duplicate count comes straight from clean.py's audit (tests/test_clean.py)
    assert sorted(kept["species_id"].tolist()) == [100, 101, 102]
    assert (kept["cell_id"] == "c0000_0000").all()
    assert sum(audit.values()) == 11 - 3


def test_clean_chunk_handles_simple_csv_without_eventid():
    grid, meta = _toy_grid()
    raw = _raw_chunk(meta).drop(columns=["eventID"])
    kept, _ = br.clean_chunk(raw, meta, set(grid.cell_id))
    assert kept["eventID"].isna().all() and sorted(kept["species_id"]) == [100, 101, 102]


def test_alphanumeric_species_key_fails_loudly():
    """An SQL-layer key like '59PK8' must stop the pipeline, not become NaN and
    vanish through clean.py's species-rank filter."""
    grid, meta = _toy_grid()
    raw = _raw_chunk(meta)
    raw.loc[0, "speciesKey"] = "59PK8"
    try:
        br.clean_chunk(raw, meta, set(grid.cell_id))
    except ValueError as e:
        assert "speciesKey" in str(e) and "59PK8" in str(e)
    else:
        raise AssertionError("alphanumeric speciesKey passed silently")
    raw.loc[0, "speciesKey"] = None                # genuinely missing stays allowed
    br.clean_chunk(raw, meta, set(grid.cell_id))


# --- minimum species per unit: hand-computed -----------------------------------
def _units_frame():
    # units are (cell, point, day): (P1, Jan 1): {1,2,3}; (P2, Jan 1): {1}; (P1, Jan 2): {2,3}
    rec = [("obs1", 20.0, "2016-01-01", s) for s in (1, 2, 3)] + [("obs2", 20.01, "2016-01-01", 1)] + \
          [("obs1", 20.0, "2016-01-02", s) for s in (2, 3)]
    return pd.DataFrame({"cell_id": "cA", "eventID": None, "recordedBy": [r[0] for r in rec],
                         "decimalLatitude": [r[1] for r in rec], "decimalLongitude": 78.0,
                         "eventDate": [r[2] for r in rec], "species_id": [r[3] for r in rec]})


def test_shared_checklist_is_one_unit():
    """Three observers sharing an outing (same point, same day) are one unit:
    each species gets Y = 1, not 3."""
    d = pd.DataFrame({"cell_id": "cA", "eventID": None, "recordedBy": ["a"] * 2 + ["b"] * 2 + ["c"] * 2,
                      "decimalLatitude": 20.0, "decimalLongitude": 78.0, "eventDate": "2016-01-01",
                      "species_id": [1, 2] * 3})
    inc, _ = build_incidence(d, route="gbif")
    assert int(inc["T"].iloc[0]) == 1 and inc["Y"].tolist() == [1, 1], inc


def test_min_species_filter_hand_computed():
    d = _units_frame()
    expected = {1: (3, {1: 2, 2: 2, 3: 2}), 2: (2, {1: 1, 2: 2, 3: 2}), 3: (1, {1: 1, 2: 1, 3: 1}),
                4: (0, {})}
    for n, (T, Y) in expected.items():
        sub, stats = br.min_species_filter(d, n)
        inc, _ = build_incidence(sub, route="gbif")
        got_Y = dict(zip(inc.species_id, inc.Y))
        got_T = int(inc["T"].iloc[0]) if len(inc) else 0
        assert (got_T, got_Y) == (T, Y), (n, got_T, got_Y)
        assert stats["units_in"] == 3


# --- partitioning is exact ------------------------------------------------------
def _random_records(seed=1, n=20000, cells=40):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "cell_id": [f"c{c:04d}_0000" for c in rng.integers(0, cells, n)],
        "species_id": rng.integers(1, 60, n).astype("int32"),
        "eventID": None,
        "recordedBy": [f"obsr{o}" for o in rng.integers(0, 30, n)],
        "eventDate": [f"2016-01-{dd:02d}" for dd in rng.integers(1, 29, n)],
        "datasetKey": "ds",
    })
    d["decimalLatitude"] = 20 + d["cell_id"].str[1:5].astype(int) * 0.1 + rng.integers(0, 3, n) * 1e-4
    d["decimalLongitude"] = 78.0 + rng.integers(0, 3, n) * 1e-4   # small grid: many exact duplicates
    return d


def test_cross_chunk_dedup_uses_cleans_rule():
    """The cross-chunk pass must match clean.py: different observers at one
    point-day are kept, the same observer's repeat is dropped."""
    from clean import DEDUP_KEY
    d = pd.DataFrame({"species_id": [1, 1, 1], "recordedBy": ["a", "b", "a"],
                      "decimalLatitude": 20.0, "decimalLongitude": 78.0, "eventDate": "2016-01-01",
                      "cell_id": "cA", "eventID": None})
    assert br.DEDUP_SUBSET == DEDUP_KEY
    assert br.dedup(d)["recordedBy"].tolist() == ["a", "b"]


def test_partitioned_processing_equals_global():
    d = _random_records()
    glob = br.dedup(d)
    whole = pd.concat([build_incidence(br.min_species_filter(glob, n)[0], route="gbif")[0].assign(N=n)
                       for n in br.N_VALUES])
    parts = []
    for p in np.unique(br.part_of(d.cell_id.to_numpy())):
        sub = br.dedup(d[br.part_of(d.cell_id.to_numpy()) == p])
        parts += [build_incidence(br.min_species_filter(sub, n)[0], route="gbif")[0].assign(N=n)
                  for n in br.N_VALUES]
    key = ["N", "cell_id", "species_id"]
    a = whole.sort_values(key).reset_index(drop=True)
    b = pd.concat(parts).sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
    assert len(d) > len(glob), "fixture should contain duplicates"


# --- coverage end to end against a known truth ---------------------------------
def _simulate_cell(T, seed, n_species=150):
    rng = np.random.default_rng(seed)
    p = np.clip(rng.lognormal(-3.0, 1.2, n_species), 0, 0.9)  # few common, many rare species
    hits = rng.random((T, n_species)) < p
    rec = [(u, s) for u in range(T) for s in np.flatnonzero(hits[u])]
    d = pd.DataFrame({"cell_id": "cA", "eventID": None, "recordedBy": [f"obs{u}" for u, _ in rec],
                      "decimalLatitude": [20 + u * 1e-4 for u, _ in rec], "decimalLongitude": 78.0,
                      "eventDate": "2016-01-15", "species_id": [s for _, s in rec]})
    true_cov = 1 - np.sum(p * (1 - p) ** T) / p.sum()   # expected incidence coverage
    return d, true_cov


def test_coverage_estimate_tracks_known_truth_at_realistic_effort():
    for T in (30, 100):
        est, truth = [], []
        for seed in range(30):
            d, c = _simulate_cell(T, seed)
            inc, _ = build_incidence(d, route="gbif")
            r = br.cell_richness("cA", inc.Y.to_numpy(float), int(inc["T"].iloc[0]))
            est.append(r["coverage_full"])
            truth.append(c)
        bias = np.mean(est) - np.mean(truth)
        assert abs(bias) < 0.02, (T, np.mean(est), np.mean(truth))


def test_tiny_samples_report_spurious_full_coverage():
    """The trap the minimum-record guard exists for: 2 records of one common
    species in 2 units -> no singletons -> coverage 1.0."""
    d = pd.DataFrame({"cell_id": "cA", "eventID": None, "recordedBy": ["o1", "o2"],
                      "decimalLatitude": [20.0, 20.01], "decimalLongitude": 78.0,
                      "eventDate": ["2016-01-01", "2016-01-01"], "species_id": [7, 7]})
    inc, _ = build_incidence(d, route="gbif")
    r = br.cell_richness("cA", inc.Y.to_numpy(float), int(inc["T"].iloc[0]))
    assert r["coverage_full"] == 1.0 and r["T"] == 2 and r["S_obs"] == 1


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
