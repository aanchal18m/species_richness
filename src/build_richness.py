"""Part 4: per-cell incidence, coverage and coverage-standardised richness.

GBIF download -> clean (clean.py) -> whole winters -> cells (incidence.assign_cells)
-> sampling units and incidence (incidence.build_incidence, route="gbif")
-> minimum-species-per-unit filter at N = 1, 3, 5, 10
-> coverage and rarefied richness (richness.py) per cell and N.

The validated modules are called, never modified. Two passes keep memory
bounded: pass 1 streams the download and writes cleaned records partitioned by
cell; pass 2 works one partition at a time. Everything after cleaning groups
within a cell (dedup subset, units, incidence), so per-partition results are
identical to a global pass -- tests/test_build_richness.py checks this.

Stages
  clean      -> data/interim/occ_clean/part-XX.parquet, outputs/tables/cleaning_audit.csv
  incidence  -> data/interim/incidence.parquet (cell_id, species_id, Y, T, N)
  richness   -> data/interim/richness.parquet  (one row per cell x N)
  all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean import DEDUP_KEY, clean_occurrences  # noqa: E402
from incidence import assign_cells, build_incidence, define_sampling_units  # noqa: E402
from richness import chao2, coverage_standardised_richness, incidence_stats, observed_coverage  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
GBIF_DIR = ROOT / "data" / "raw" / "gbif"
OCC_DIR = INTERIM / "occ_clean"
INCIDENCE = INTERIM / "incidence.parquet"
RICHNESS = INTERIM / "richness.parquet"
AUDIT = ROOT / "outputs" / "tables" / "cleaning_audit.csv"
CLEAN_REPORT = ROOT / "outputs" / "tables" / "clean_report.json"

N_VALUES = (1, 3, 5, 10)
TARGET_COVERAGES = (0.80, 0.90, 0.95)
FIRST_WINTER_END, LAST_WINTER_END = 2016, 2024   # 2015/16 .. 2023/24
N_PARTS = 16
# Columns clean.py, the winter/date filters, cell assignment and the unit
# definition read. Everything else in SIMPLE_CSV is never loaded.
READ_COLS = ["datasetKey", "eventID", "recordedBy", "eventDate", "year", "month",
             "decimalLatitude", "decimalLongitude", "coordinateUncertaintyInMeters",
             "basisOfRecord", "occurrenceStatus", "taxonRank", "speciesKey"]
KEEP_COLS = ["cell_id", "species_id", "eventID", "recordedBy", "eventDate",
             "decimalLatitude", "decimalLongitude", "datasetKey"]
def _schema():
    import pyarrow as pa
    return pa.schema([("cell_id", pa.string()), ("species_id", pa.int32()), ("eventID", pa.string()),
                      ("recordedBy", pa.string()), ("eventDate", pa.string()),
                      ("decimalLatitude", pa.float64()), ("decimalLongitude", pa.float64()),
                      ("datasetKey", pa.string())])


DEDUP_SUBSET = DEDUP_KEY   # clean.py's definition, applied again across chunk boundaries

# Filters GBIF applied server-side in the predicate download. They appear in the audit with 0 dropped here
# because the download already excludes those records; `upstream_estimate` gives
# what the public search API counted for each before the request (2026-10-05).
SERVER_SIDE = [
    ("class Aves, country IN (GBIF predicate)", None),
    ("year 2015-2024, months Nov-Feb (GBIF predicate)", None),
    ("has coordinate, no geospatial issue (GBIF predicate)", None),
    ("basisOfRecord not FOSSIL/LIVING_SPECIMEN (GBIF predicate)", 0),
    ("not within 2 km of a country/province centroid (GBIF predicate)", 1558),
    ("no TAXON_MATCH_FUZZY issue (GBIF predicate)", 13684),
]
NUMERIC = ["decimalLatitude", "decimalLongitude", "coordinateUncertaintyInMeters",
           "year", "month", "speciesKey"]


# --- record-level steps -----------------------------------------------------
def to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """read_chunks yields strings; clean.py compares numbers.

    Any non-empty value that is not a number is an error, never a silent NaN:
    GBIF's SQL layer uses alphanumeric taxon keys ('V2', '59PK8'), and coercing
    one to NaN would let clean.py's species-rank filter drop it unseen (trap 15)."""
    d = df.copy()
    for c in NUMERIC:
        if c in d:
            raw = d[c]
            num = pd.to_numeric(raw, errors="coerce")
            bad = num.isna() & raw.notna() & (raw.astype("string").str.strip() != "")
            if bad.any():
                raise ValueError(f"{c}: {int(bad.sum())} non-numeric values, e.g. "
                                 f"{raw[bad].head(3).tolist()} -- alphanumeric GBIF keys? (trap 15)")
            d[c] = num
    return d


def winter_end_year(year, month):
    """The calendar year in which a record's winter ends: Nov-Dec belong to the
    winter ending the following year."""
    year = pd.to_numeric(pd.Series(year), errors="coerce")
    month = pd.to_numeric(pd.Series(month), errors="coerce")
    return year.where(month < 11, year + 1)


def whole_winters(df: pd.DataFrame) -> pd.Series:
    w = winter_end_year(df["year"].to_numpy(), df["month"].to_numpy())
    return pd.Series(w.between(FIRST_WINTER_END, LAST_WINTER_END).to_numpy(), index=df.index)


def single_day(event_date: pd.Series) -> pd.Series:
    """True where eventDate names one day. incidence.py keys units on
    eventDate[:10]; a month-only date would make an observer-month unit."""
    ed = event_date.astype("string").fillna("")
    parts = ed.str.split("/", n=1, expand=True).reindex(columns=[0, 1])
    start = parts[0].fillna("").str[:10]
    stop = parts[1].fillna("").str[:10]
    return start.str.fullmatch(r"\d{4}-\d{2}-\d{2}").fillna(False) & ((stop == "") | (stop == start))


def part_of(cell_ids) -> np.ndarray:
    return np.array([zlib.crc32(c.encode()) % N_PARTS for c in cell_ids], dtype=np.int16)


def clean_chunk(raw: pd.DataFrame, meta: dict, valid_cells: set) -> tuple[pd.DataFrame, list]:
    """One chunk through clean.py and the Part 4 record filters. Returns the kept
    records and audit rows (filter, dropped)."""
    d, audit = clean_occurrences(to_numeric(raw))
    if "eventID" not in d:     # SIMPLE_CSV has none; units then come from observer + date + cell
        d["eventID"] = None
    rows = [(r["filter"], r["dropped"]) for r in audit.to_dict("records") if r["filter"] != "input"]
    # clean.py skips its centroid step silently when no reference list is given;
    # keep a row so the audit shows it was not run (GBIF's 2 km filter ran server-side).
    rows.insert(-1, ("not within a known centroid list (clean.py; no list supplied, not run locally)", 0))

    keep = whole_winters(d)
    rows.append((f"whole winters {FIRST_WINTER_END - 1}/{str(FIRST_WINTER_END)[2:]}"
                 f"-{LAST_WINTER_END - 1}/{str(LAST_WINTER_END)[2:]}", int((~keep).sum())))
    d = d[keep]
    keep = single_day(d["eventDate"])
    rows.append(("eventDate resolves to a single day", int((~keep).sum())))
    d = d[keep]
    d = d.assign(cell_id=assign_cells(d["decimalLongitude"].to_numpy(), d["decimalLatitude"].to_numpy(),
                                      meta, valid_cells=valid_cells))
    keep = d["cell_id"].notna()
    rows.append(("falls in a grid cell (land fraction >= 0.5)", int((~keep).sum())))
    d = d[keep]
    return d[KEEP_COLS].reset_index(drop=True), rows


# --- unit-level steps -------------------------------------------------------
def dedup(d: pd.DataFrame) -> pd.DataFrame:
    """clean.py's duplicate rule, applied again across chunk boundaries."""
    return d.drop_duplicates(subset=DEDUP_SUBSET)


def min_species_filter(d: pd.DataFrame, n: int) -> tuple[pd.DataFrame, dict]:
    """Keep records whose sampling unit lists at least n distinct species.

    GBIF carries no 'all species reported' flag, so completeness is approximated
    by list length; n is a sensitivity parameter (fixed decision 7)."""
    units = define_sampling_units(d, route="gbif")
    per_unit = d.assign(_u=units).dropna(subset=["_u"]).groupby("_u")["species_id"].nunique()
    good = per_unit.index[per_unit >= n]
    keep = units.isin(good).fillna(False).to_numpy()
    return d[keep], {"units_in": int(len(per_unit)), "units_kept": int(len(good)),
                     "records_kept": int(keep.sum())}


def incidence_for_part(path: Path) -> tuple[pd.DataFrame, dict]:
    d = pd.read_parquet(path)
    n_in = len(d)
    d = dedup(d)
    out, info = [], {"records_in": n_in, "cross_chunk_duplicates": n_in - len(d)}
    records = d.groupby("cell_id").size().rename("n_records")
    for n in N_VALUES:
        sub, stats = min_species_filter(d, n)
        inc, rep = build_incidence(sub, route="gbif")
        recs_n = sub.groupby("cell_id").size().rename("n_records_N")
        out.append(inc.assign(N=n).merge(recs_n, on="cell_id", how="left"))
        info[f"N{n}"] = {**stats, **rep}
    inc = pd.concat(out, ignore_index=True).merge(records, on="cell_id", how="left")
    return inc, info


# --- richness ---------------------------------------------------------------
def cell_richness(cell_id: str, Y: np.ndarray, T: int) -> dict:
    st = incidence_stats(Y, T)
    row = {"cell_id": cell_id, **st, "coverage_full": observed_coverage(Y, T) if st["S_obs"] else 0.0,
           "chao2": chao2(Y, T)}
    for c in TARGET_COVERAGES:
        r = coverage_standardised_richness(Y, T, target_coverage=c)
        key = f"{int(round(c * 100)):03d}"
        row[f"richness_c{key}"] = r["richness"]
        row[f"t_star_c{key}"] = r["t_star"]
    return row


def _richness_group(args):
    n, frame = args
    rows = []
    for cid, g in frame.groupby("cell_id", sort=False):
        row = cell_richness(cid, g["Y"].to_numpy(float), int(g["T"].iloc[0]))
        row.update(N=n, n_records=int(g["n_records"].iloc[0]), n_records_N=int(g["n_records_N"].iloc[0]))
        rows.append(row)
    return rows


# --- stages -----------------------------------------------------------------
def stage_clean(zip_path: Path | None = None, chunksize: int = 250_000):
    import geopandas as gpd
    from build_grid import load_meta
    from gbif_download import KEY_FILE, read_chunks

    zip_path = zip_path or GBIF_DIR / f"{KEY_FILE.read_text().strip()}.zip"
    meta = load_meta()
    valid = set(gpd.read_file(INTERIM / "grid.gpkg", layer="grid", columns=["cell_id"],
                              ignore_geometry=True)["cell_id"])
    OCC_DIR.mkdir(parents=True, exist_ok=True)
    for old in OCC_DIR.glob("*.parquet"):
        old.unlink()
    writers, totals, n_in = {}, {}, 0
    import pyarrow as pa
    import pyarrow.parquet as pq
    try:
        for k, raw in enumerate(read_chunks(zip_path, chunksize=chunksize, usecols=READ_COLS)):
            n_in += len(raw)
            d, rows = clean_chunk(raw, meta, valid)
            for name, dropped in rows:
                totals[name] = totals.get(name, 0) + dropped
            d["species_id"] = d["species_id"].astype("int32")
            parts = part_of(d["cell_id"].to_numpy())
            for p in np.unique(parts):
                t = pa.Table.from_pandas(d[parts == p], schema=_schema(), preserve_index=False)
                if p not in writers:
                    writers[p] = pq.ParquetWriter(OCC_DIR / f"part-{p:02d}.parquet", _schema(),
                                                  compression="zstd")
                writers[p].write_table(t)
            print(f"   chunk {k + 1}: {n_in:,} records read", flush=True)
    finally:
        for w in writers.values():
            w.close()
    audit = [("input (download rows)", 0)] + [(name, 0) for name, _ in SERVER_SIDE] + list(totals.items())
    df = pd.DataFrame(audit, columns=["filter", "dropped"])
    df["stage"] = ["input"] + ["server (GBIF predicate download)"] * len(SERVER_SIDE) + ["local"] * len(totals)
    df["upstream_estimate"] = [None] + [u for _, u in SERVER_SIDE] + [None] * len(totals)
    df["kept"] = n_in - df["dropped"].cumsum()
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(AUDIT, index=False)
    print(df.to_string(index=False))


def stage_incidence():
    parts = sorted(OCC_DIR.glob("part-*.parquet"))
    incs, info = [], {}
    for p in parts:
        inc, i = incidence_for_part(p)
        incs.append(inc)
        info[p.stem] = i
        print(f"   {p.name}: {i['records_in']:,} records, {inc.cell_id.nunique():,} cells", flush=True)
    inc = pd.concat(incs, ignore_index=True)
    inc.to_parquet(INCIDENCE, index=False)
    summary = {"cross_chunk_duplicates": sum(v["cross_chunk_duplicates"] for v in info.values())}
    for n in N_VALUES:
        keys = ["units_in", "units_kept", "records_kept", "dropped_no_sampling_unit", "cells_with_data"]
        summary[f"N{n}"] = {k: int(sum(v[f"N{n}"][k] for v in info.values())) for k in keys}
    prev = json.loads(CLEAN_REPORT.read_text()) if CLEAN_REPORT.exists() else {}
    CLEAN_REPORT.write_text(json.dumps({**prev, "incidence": summary}, indent=2))
    print(json.dumps(summary, indent=2))


def stage_richness(workers: int = 10):
    inc = pd.read_parquet(INCIDENCE)
    cells = inc["cell_id"].unique()
    batches = np.array_split(cells, workers * 4)
    jobs = [(n, inc[(inc.N == n) & inc.cell_id.isin(b)]) for n in N_VALUES for b in batches]
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(_richness_group, jobs)):
            rows.extend(r)
            if (k + 1) % 20 == 0:
                print(f"   richness batches {k + 1}/{len(jobs)}", flush=True)
    out = pd.DataFrame(rows).sort_values(["N", "cell_id"]).reset_index(drop=True)
    out.to_parquet(RICHNESS, index=False)
    print(f"   {RICHNESS.relative_to(ROOT)}: {len(out):,} rows "
          f"({out.groupby('N').size().to_dict()} cells per N)")


def main(argv=None):
    stages = {"clean": stage_clean, "incidence": stage_incidence, "richness": stage_richness}
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=[*stages, "all"])
    args = p.parse_args(argv)
    for name in (stages if args.stage == "all" else [args.stage]):
        t0 = time.time()
        print(f"[{name}]", flush=True)
        stages[name]()
        print(f"[{name}] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
