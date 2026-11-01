"""Equal-area analysis grid.

WHY A PROJECTION DECISION COMES FIRST
-------------------------------------
Species richness is an area-dependent quantity. The species-area relationship
is one of the oldest regularities in ecology: bigger patches hold more species,
roughly as a power law. So if our grid cells differ in true ground area, part of
the richness variation we model is just cell size, and we would be fitting the
species-area curve while calling it climate.

A grid defined in degrees has exactly this problem. A cell one degree wide
covers about 111 km at the equator but only 111*cos(latitude) km east-west
further north. India runs from roughly 6 N to 37 N, so a degree-based cell in
Kashmir is materially smaller than one in Kerala.

That matters more here than in most projects. We are comparing richness BETWEEN
biogeographic zones, and those zones are stacked along the latitude India spans
-- the Himalaya at the top, the Western Ghats and Deccan at the bottom. A
latitude-driven area gradient would therefore line up almost perfectly with the
zone contrast we are trying to measure. It would not be noise; it would be a
confound pointing in the same direction as the signal.

The fix is to build the grid in an equal-area projection, where every cell has
the same ground area by construction.

CHOICE OF PROJECTION
--------------------
Albers Equal Area Conic, with standard parallels placed inside India's latitude
range. Conic projections suit mid-latitude regions wider than they are tall, and
placing the two standard parallels at roughly one-sixth and five-sixths of the
range minimises distortion across the interior.

  +proj=aea +lat_1=12.5 +lat_2=32.5 +lat_0=20 +lon_0=80 +datum=WGS84 +units=m

Equal-area projections distort shape, which is a price worth paying: we care
that cells are comparable, not that coastlines look right.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from shapely.geometry import box

INDIA_AEA = (
    "+proj=aea +lat_1=12.5 +lat_2=32.5 +lat_0=20 +lon_0=80 "
    "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)


def build_grid(boundary: gpd.GeoDataFrame, cell_km: float = 10.0,
               min_land_fraction: float = 0.5) -> tuple[gpd.GeoDataFrame, dict]:
    """Tile `boundary` with square equal-area cells.

    Parameters
    ----------
    boundary : polygons of the study area, any CRS
    cell_km : cell edge length in kilometres
    min_land_fraction : drop cells whose overlap with the boundary is below
        this. Coastal and border cells are partly sea or partly outside India;
        keeping a cell that is 5% land means dividing its species count by an
        area that is mostly not there. 0.5 is a common, defensible cut -- record
        it, because it is a judgement call that changes the sample.

    Returns a GeoDataFrame in INDIA_AEA with cell_id, land_fraction, and the
    cell centroid in lon/lat (needed later for spatial cross-validation blocks).
    """
    b = boundary.to_crs(INDIA_AEA)
    unioned = b.union_all()
    minx, miny, maxx, maxy = unioned.bounds
    step = cell_km * 1000.0

    # Anchor the lattice to a round number so the grid is reproducible and does
    # not shift when the boundary file changes by a metre.
    x0 = np.floor(minx / step) * step
    y0 = np.floor(miny / step) * step
    xs = np.arange(x0, maxx + step, step)
    ys = np.arange(y0, maxy + step, step)

    cells, ids = [], []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            cells.append(box(x, y, x + step, y + step))
            ids.append(cell_id(i, j))

    g = gpd.GeoDataFrame({"cell_id": ids}, geometry=cells, crs=INDIA_AEA)

    # Spatial index makes the intersection test tractable at national extent.
    g = g[g.intersects(unioned)].copy()
    inter = g.geometry.intersection(unioned)
    g["land_fraction"] = inter.area / g.geometry.area
    g = g[g["land_fraction"] >= min_land_fraction].copy()

    cent = g.geometry.centroid.to_crs("EPSG:4326")
    g["lon"] = cent.x.values
    g["lat"] = cent.y.values
    g["area_km2"] = g.geometry.area / 1e6

    # The lattice origin and step are returned alongside the cells. They are
    # what let `incidence.assign_cells` locate a point by arithmetic instead of
    # a spatial join -- see that module for why it matters at 60M records.
    meta = {"crs": INDIA_AEA, "step_m": float(step),
            "x0": float(x0), "y0": float(y0), "cell_km": float(cell_km),
            "min_land_fraction": float(min_land_fraction)}
    return g.reset_index(drop=True), meta


def cell_id(i: int, j: int) -> str:
    """Canonical cell identifier from lattice indices. One definition, used
    by both the grid builder and the point assigner, so they cannot drift."""
    return f"c{int(i):04d}_{int(j):04d}"
