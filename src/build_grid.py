"""Part 3: build the 10 km equal-area grid from the India boundary.

Writes
  data/interim/grid.gpkg        cell polygons (INDIA_AEA), cell_id, land_fraction,
                                lon/lat of centroid, area_km2
  data/interim/grid_meta.json   the lattice metadata returned by grid.build_grid:
                                CRS, origin (x0, y0), step. incidence.assign_cells
                                needs these, not the polygons, to place points by
                                arithmetic.

Run: python src/build_grid.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid import build_grid  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BOUNDARY = ROOT / "data" / "raw" / "india_boundary.gpkg"
GRID = ROOT / "data" / "interim" / "grid.gpkg"
META = ROOT / "data" / "interim" / "grid_meta.json"
CELL_KM = 10.0
MIN_LAND_FRACTION = 0.5


def load_meta(path: Path = META) -> dict:
    return json.loads(Path(path).read_text())


def omitted_land(grid: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, float]:
    """Boundary polygons (islands) with no kept cell, and the total land area
    outside kept cells. Islands smaller than about half a cell can never pass
    MIN_LAND_FRACTION, so whole archipelagos drop out; this makes that visible."""
    from shapely.ops import unary_union

    b = boundary.to_crs(grid.crs).union_all()
    cells = unary_union(grid.geometry.values)
    parts = gpd.GeoDataFrame(geometry=list(getattr(b, "geoms", [b])), crs=grid.crs)
    parts["area_km2"] = parts.area / 1e6
    parts["covered_frac"] = [p.intersection(cells).area / p.area for p in parts.geometry]
    c = parts.geometry.centroid.to_crs("EPSG:4326")
    parts["lon"], parts["lat"] = c.x, c.y
    lost = parts[parts.covered_frac < 0.01].sort_values("area_km2", ascending=False)
    return lost, b.difference(cells).area / 1e6


def main():
    boundary = gpd.read_file(BOUNDARY, layer="india")
    grid, meta = build_grid(boundary, cell_km=CELL_KM, min_land_fraction=MIN_LAND_FRACTION)
    if grid["cell_id"].duplicated().any():
        raise ValueError("duplicate cell_id in grid")
    GRID.parent.mkdir(parents=True, exist_ok=True)
    if GRID.exists():
        GRID.unlink()
    grid.to_file(GRID, layer="grid", driver="GPKG")
    META.write_text(json.dumps({**meta, "n_cells": len(grid), "boundary": BOUNDARY.name}, indent=2))
    print(f"{len(grid):,} cells, {grid.area_km2.sum():,.0f} km2 of cell area; "
          f"land_fraction < 1 in {(grid.land_fraction < 1 - 1e-9).sum():,} cells; meta -> {META.name}")


if __name__ == "__main__":
    main()
