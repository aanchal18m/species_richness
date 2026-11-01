"""Occurrence record cleaning.

WHY RECORDS MUST BE FILTERED BEFORE THEY ARE COUNTED
-----------------------------------------------------
GBIF aggregates records from thousands of institutions with incompatible
conventions. A raw count of distinct species per cell will include artefacts
that have nothing to do with where birds are:

  * Country and province CENTROIDS. A record whose location was recorded only
    as "India" is often given the coordinates of the country's centroid. These
    pile hundreds of species onto one cell in central India that no one ever
    visited. This is the single most damaging artefact for a richness model,
    because it creates a false richness peak with no environmental cause.
  * Institution coordinates. Specimens are sometimes georeferenced to the
    museum or herbarium holding them, not the collection site.
  * Zero coordinates. (0, 0) is in the Atlantic; it means a missing value that
    was silently filled.
  * Coarse georeferencing. A record accurate to +/- 50 km cannot be assigned to
    a 10 km cell. Keeping it smears species across cells.
  * Non-wild records. Zoo and captive birds, and fossils, are not part of the
    local community.
  * Taxonomic noise. Subspecies, hybrids and unresolved names inflate counts if
    not collapsed to accepted species.

Each filter below removes a specific, named failure mode. We record how many
records each one drops, because that table belongs in the report -- it is the
evidence that the response variable was built carefully.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Known institution / centroid artefact locations should be loaded from a
# reference list (the CoordinateCleaner package ships one). Kept as a parameter
# so the pipeline does not hide a hard-coded assumption.
DEFAULT_MAX_UNCERTAINTY_M = 10_000      # must be finer than the 10 km cell
DEFAULT_MIN_YEAR = 2015                 # matches the project's study period
# What makes two records duplicates: the SAME observer, species, place and time.
# One definition, imported wherever duplicates are removed, so copies cannot drift.
DEDUP_KEY = ["species_id", "recordedBy", "decimalLatitude", "decimalLongitude", "eventDate"]
ACCEPTABLE_BASIS = {
    "HUMAN_OBSERVATION",
    "MACHINE_OBSERVATION",
    "PRESERVED_SPECIMEN",
    "OBSERVATION",
}


def clean_occurrences(
    df: pd.DataFrame,
    max_uncertainty_m: int = DEFAULT_MAX_UNCERTAINTY_M,
    min_year: int = DEFAULT_MIN_YEAR,
    centroid_buffer_m: float = 5000.0,
    known_centroids: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the filters in order, returning (clean_df, audit_table).

    The audit table is a deliverable, not a debug aid: it tells a reader exactly
    how many records each decision removed.
    """
    audit = []
    n = len(df)

    def step(name, mask, frame):
        nonlocal audit
        kept = int(mask.sum())
        audit.append({"filter": name, "dropped": int(len(frame) - kept), "kept": kept})
        return frame[mask].copy()

    d = df.copy()
    audit.append({"filter": "input", "dropped": 0, "kept": n})

    d = step("has coordinates", d[["decimalLatitude", "decimalLongitude"]].notna().all(axis=1), d)
    d = step("coordinates not (0,0)",
             ~((d.decimalLatitude.abs() < 1e-6) & (d.decimalLongitude.abs() < 1e-6)), d)
    d = step("coordinates in valid range",
             d.decimalLatitude.between(-90, 90) & d.decimalLongitude.between(-180, 180), d)
    d = step(f"georeferencing finer than {max_uncertainty_m/1000:.0f} km",
             d.coordinateUncertaintyInMeters.isna() |
             (d.coordinateUncertaintyInMeters <= max_uncertainty_m), d)
    d = step("acceptable basisOfRecord", d.basisOfRecord.isin(ACCEPTABLE_BASIS), d)
    d = step("not flagged absent", d.get("occurrenceStatus", pd.Series("PRESENT", index=d.index)) != "ABSENT", d)
    d = step(f"year >= {min_year}", d.year.notna() & (d.year >= min_year), d)
    d = step("resolved to species rank",
             d.taxonRank.isin(["SPECIES", "SUBSPECIES"]) & d.speciesKey.notna(), d)

    if known_centroids is not None and len(known_centroids):
        from shapely.geometry import Point
        import geopandas as gpd
        pts = gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(
            d.decimalLongitude, d.decimalLatitude), crs="EPSG:4326").to_crs(3857)
        cps = gpd.GeoDataFrame(known_centroids, geometry=gpd.points_from_xy(
            known_centroids.lon, known_centroids.lat), crs="EPSG:4326").to_crs(3857)
        near = pts.geometry.apply(
            lambda p: bool((cps.distance(p) < centroid_buffer_m).any()))
        d = step(f"not within {centroid_buffer_m/1000:.0f} km of a known centroid", ~near, d)

    # Collapse subspecies onto the accepted species so richness counts species,
    # not names. GBIF's speciesKey already does this resolution for us.
    d["species_id"] = d.speciesKey.astype("int64")
    # A duplicate is the SAME observer recording the same species at the same
    # place and time. Without recordedBy, different observers at one eBird
    # hotspot (identical coordinates) on one day collapse into one record: on
    # the GBIF download that deleted 9.8M of 23.6M records, 92% of them
    # cross-observer -- independent observations, not duplicates.
    before = len(d)
    d = d.drop_duplicates(subset=DEDUP_KEY)
    audit.append({"filter": "exact duplicate records", "dropped": before - len(d), "kept": len(d)})

    return d.reset_index(drop=True), pd.DataFrame(audit)
