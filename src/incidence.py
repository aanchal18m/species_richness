"""From cleaned occurrence records to per-cell incidence vectors.

This module is the join between the grid and the data. It answers two questions
that look like plumbing but are actually the most consequential modelling
choices in the data pipeline.


DECISION 1 — WHAT COUNTS AS A SAMPLING UNIT?
--------------------------------------------
`richness.py` works on INCIDENCE data: for each species it needs Y_i, the number
of sampling units in which that species was detected. Everything downstream --
coverage, rarefaction, the well/poorly sampled split -- is computed from those
counts. So the definition of "sampling unit" silently determines the response
variable of the entire project.

A sampling unit must be an *occasion of looking*. Not a record, and not an
observer: one birdwatcher walking one route one morning and writing down what
they saw is a unit. If they go back the next day, that is a second unit, and a
species appearing on both has Y_i = 2.

This matters because the Chao estimators read the signal of incompleteness from
singletons and doubletons -- species seen in exactly one or exactly two units.
If units are defined too finely, almost every species becomes a singleton and
the estimator reports enormous undetected richness. Too coarsely, and Q1 and Q2
collapse towards zero and coverage looks falsely perfect.

Two routes, two definitions:

  eBird route -- a unit is a CHECKLIST, identified by its sampling event ID.
      This is the right answer. eBird checklists are explicitly bounded
      occasions of looking, with a start time, a duration and a complete-
      species flag. Nothing needs to be inferred.

  GBIF route -- GBIF records are not organised into visits. eventID is used
      where present, but eBird records on GBIF carry none (CLAUDE.md trap 6),
      so in practice every unit is a pseudo-unit: (cell, point, day) -- all
      records at the same exact coordinates on the same day in the same 10 km
      cell, POOLING OBSERVERS (rule C, decided at Gate 1 review, 2026-10-06).

      Why pool observers: eBird shares one checklist among everyone in a group
      and gives every visitor to a hotspot the hotspot's coordinates. Keyed on
      the observer, a group of k people becomes k units with identical lists,
      so every species looks k times commoner and coverage is overstated --
      most of all in busy, heavily watched cells, which become the training
      data. Pooling merges shared checklists correctly; it also merges
      independent visits to one hotspot on one day, so it errs coarse. That
      UNDERstates coverage slightly, which is the recoverable direction. The
      observer is not needed, so a record with no recordedBy still has a unit.
      This is an approximation and must be reported as one.

A species detected several times within one unit counts ONCE. Incidence, not
abundance.


DECISION 2 — HOW ARE POINTS ASSIGNED TO CELLS?
----------------------------------------------
The obvious approach is a spatial join of points against cell polygons. For 61
million Indian bird records that is slow and needless, because our grid is not
an arbitrary set of polygons -- it is a regular lattice with a known origin and
a fixed step. The cell containing a point is therefore pure arithmetic:

    i = floor((x - x0) / step)
    j = floor((y - y0) / step)

This is O(n), needs no spatial index, and vectorises. `test_incidence.py` checks
it agrees exactly with a geopandas spatial join, so the fast path is verified
against the reference rather than merely assumed to match.

The points must be projected into the grid's equal-area CRS first. Doing the
arithmetic on raw longitude and latitude would reintroduce exactly the
area distortion the projection was chosen to remove.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyproj

from grid import cell_id


# ----------------------------------------------------------------------------
# Assigning records to cells
# ----------------------------------------------------------------------------

def assign_cells(lon, lat, meta: dict, valid_cells: set[str] | None = None):
    """Return the cell_id for each (lon, lat), or None where it falls outside.

    `meta` is the dict returned by grid.build_grid. `valid_cells` restricts the
    result to cells that survived the land-fraction filter; points landing in a
    lattice square that is mostly sea, or outside India, come back as None so
    the caller can count them rather than lose them silently.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    transformer = pyproj.Transformer.from_crs("EPSG:4326", meta["crs"], always_xy=True)
    x, y = transformer.transform(lon, lat)

    step, x0, y0 = meta["step_m"], meta["x0"], meta["y0"]
    i = np.floor((x - x0) / step).astype(np.int64)
    j = np.floor((y - y0) / step).astype(np.int64)

    finite = np.isfinite(x) & np.isfinite(y)
    out = np.full(lon.shape, None, dtype=object)
    idx = np.flatnonzero(finite)
    ids = np.array([cell_id(a, b) for a, b in zip(i[idx], j[idx])], dtype=object)
    if valid_cells is not None:
        keep = np.array([c in valid_cells for c in ids], dtype=bool)
        ids = np.where(keep, ids, None)
    out[idx] = ids
    return out


# ----------------------------------------------------------------------------
# Defining sampling units
# ----------------------------------------------------------------------------

def define_sampling_units(df: pd.DataFrame, route: str = "gbif") -> pd.Series:
    """Label each record with the sampling unit it belongs to.

    route="ebird"  use the checklist identifier directly.
    route="gbif"   use eventID where present; otherwise fall back to the
                   (cell, point, day) pseudo-unit described above.

    Records that cannot be assigned a unit -- no event and no date or
    coordinates -- return NA and are dropped by `build_incidence`, which
    reports how many were lost. An undated record cannot contribute to an
    effort-aware richness estimate, because we do not know whether it
    represents a separate occasion of looking or a duplicate of one already
    counted. If the fallback is needed and the coordinate columns are absent
    altogether, this raises rather than silently returning no units.
    """
    if route == "ebird":
        if "samplingEventIdentifier" not in df.columns:
            raise KeyError("eBird route requires a samplingEventIdentifier column")
        return df["samplingEventIdentifier"].astype("string")

    if route != "gbif":
        raise ValueError("route must be 'ebird' or 'gbif'")

    unit = pd.Series(pd.NA, index=df.index, dtype="string")

    if "eventID" in df.columns:
        ev = df["eventID"].astype("string")
        unit = unit.mask(ev.notna() & (ev.str.len() > 0), "EV:" + ev)

    need = unit.isna()
    if need.any():
        missing = [c for c in ("decimalLatitude", "decimalLongitude") if c not in df.columns]
        if missing:
            raise KeyError(f"GBIF pseudo-units are (cell, point, day); missing columns {missing}")
        # exact coordinates: shared checklists and hotspot visits carry identical ones
        lat = df["decimalLatitude"].astype("string")
        lon = df["decimalLongitude"].astype("string")
        day = df.get("eventDate", pd.Series(pd.NA, index=df.index)).astype("string").str[:10]
        cel = df.get("cell_id", pd.Series(pd.NA, index=df.index)).astype("string")
        ok = need & lat.notna() & lon.notna() & day.notna() & cel.notna()
        unit = unit.mask(ok, "PS:" + cel + "|" + lat + "," + lon + "|" + day)

    return unit


# ----------------------------------------------------------------------------
# Building the incidence table
# ----------------------------------------------------------------------------

def build_incidence(df: pd.DataFrame, route: str = "gbif") -> tuple[pd.DataFrame, dict]:
    """Collapse records into per-cell, per-species incidence counts.

    Input must already carry `cell_id` (from assign_cells) and `species_id`
    (from clean.clean_occurrences).

    Returns
    -------
    incidence : one row per (cell_id, species_id) with Y = number of distinct
        sampling units in that cell containing that species.
    report : counts of what was dropped and why, for the audit trail.
    """
    report = {"records_in": len(df)}

    d = df[df["cell_id"].notna()].copy()
    report["dropped_outside_grid"] = report["records_in"] - len(d)

    d["unit_id"] = define_sampling_units(d, route=route)
    before = len(d)
    d = d[d["unit_id"].notna()].copy()
    report["dropped_no_sampling_unit"] = before - len(d)

    # One species in one unit counts once, however many times it was recorded.
    d = d.drop_duplicates(subset=["cell_id", "unit_id", "species_id"])
    report["unique_species_unit_pairs"] = len(d)

    incidence = (
        d.groupby(["cell_id", "species_id"], observed=True)
         .size().rename("Y").reset_index()
    )
    units = (
        d.groupby("cell_id", observed=True)["unit_id"]
         .nunique().rename("T").reset_index()
    )

    report["cells_with_data"] = len(units)
    report["mean_units_per_cell"] = float(units["T"].mean()) if len(units) else 0.0
    return incidence.merge(units, on="cell_id", how="left"), report


def incidence_vectors(incidence: pd.DataFrame):
    """Yield (cell_id, Y_array, T) ready to hand to richness.py."""
    for cid, grp in incidence.groupby("cell_id", observed=True):
        yield cid, grp["Y"].to_numpy(dtype=float), int(grp["T"].iloc[0])
