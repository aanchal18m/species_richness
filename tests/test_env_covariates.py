"""Tests for src/env_covariates.py and src/build_grid.py.

Run: python tests/test_env_covariates.py   (or pytest)

Synthetic tests check hand-computed answers. Real-data tests (skipped until the
Part 3 outputs exist) check known places, where a CRS or axis-order mix-up that
is invisible in summary statistics becomes obvious.
"""
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import env_covariates as ec  # noqa: E402

import geopandas as gpd  # noqa: E402
import rasterio  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from shapely.geometry import box  # noqa: E402


def _write(path, arr, left=0.0, top=4.0, res=1.0, nodata=None, scale=None, offset=None):
    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[1], height=arr.shape[0],
                       count=1, dtype=arr.dtype, crs="EPSG:4326",
                       transform=from_origin(left, top, res, res), nodata=nodata) as d:
        d.write(arr, 1)
        if scale is not None:
            d.scales, d.offsets = (scale,), (offset,)
    return path


def _cells(**boxes):
    return gpd.GeoDataFrame({"cell_id": list(boxes)}, geometry=list(boxes.values()), crs="EPSG:4326")


# --- continuous aggregation: hand-computed ----------------------------------
#   values (row 0 = north)     cell layout (x right, y up; pixels are 1x1)
#    1  2  3  4                aligned : x 0-2, y 2-4  -> pixels 1,2,5,6
#    5  6  7  8                half    : x 0.5-2.5, y 2-4 -> edge columns half covered
#    9 10  X 12                nodata  : x 1-3, y 0-2  -> 10, X, 14, 15
#   13 14 15 16                outside : x 3-5, y 2-4  -> half the cell is off the raster
GRID4 = np.arange(1, 17, dtype="float32").reshape(4, 4)
GRID4[2, 2] = -9999


def test_continuous_aggregation_hand_computed():
    with tempfile.TemporaryDirectory() as d:
        p = _write(Path(d) / "r.tif", GRID4.copy(), nodata=-9999)
        cells = _cells(aligned=box(0, 2, 2, 4), half=box(0.5, 2, 2.5, 4),
                       nodata=box(1, 0, 3, 2), outside=box(3, 2, 5, 4))
        z = ec.zonal_continuous(p, cells).set_index("cell_id")

    # aligned: plain mean/population SD of 1,2,5,6
    assert np.isclose(z.loc["aligned", "mean"], 3.5)
    assert np.isclose(z.loc["aligned", "sd"], math.sqrt(4.25))
    assert z.loc["aligned", "valid_frac"] == 1.0
    # half: weights 0.5,1,0.5 on columns -> sum 16 over weight 4
    assert np.isclose(z.loc["half", "mean"], 16 / 4)
    var = (0.5 * 9 + 4 + 0.5 * 1 + 0.5 * 1 + 4 + 0.5 * 9) / 4
    assert np.isclose(z.loc["half", "sd"], math.sqrt(var))
    assert z.loc["half", "min"] == 1 and z.loc["half", "max"] == 7  # partial pixels count for range
    # nodata: 10, 14, 15 only; 3 of 4 pixels valid
    assert np.isclose(z.loc["nodata", "mean"], 13.0)
    assert np.isclose(z.loc["nodata", "valid_frac"], 0.75)
    # outside: 4, 8 at full weight; half the polygon has no raster -> valid_frac 0.5
    assert np.isclose(z.loc["outside", "mean"], 6.0)
    assert np.isclose(z.loc["outside", "valid_frac"], 0.5)


def test_scale_and_offset_applied_exactly_once():
    """CHELSA stores bio1 as K*10 with scale 0.1, offset -273.15, in the file's
    metadata. Applying it zero or two times gives 3000 or -270.465, not 26.85."""
    raw = np.array([[3000, 3010], [2990, 3000]], dtype="uint16")
    with tempfile.TemporaryDirectory() as d:
        p = _write(Path(d) / "t.tif", raw, top=2.0, scale=0.1, offset=-273.15)
        z = ec.zonal_continuous(p, _cells(c=box(0, 0, 2, 2))).iloc[0]
    assert np.isclose(z["mean"], 26.85, atol=1e-4), z["mean"]
    assert np.isclose(z["sd"], 0.1 * np.std([3000, 3010, 2990, 3000]))
    assert np.isclose(z["min"], 299.0 - 273.15) and np.isclose(z["max"], 301.0 - 273.15)


# --- land cover --------------------------------------------------------------
#   classes (0 = nodata)       cell x 0-3, y 0-2 ... see expected below
LC = np.array([[10, 10, 40, 40],
               [10, 30, 40, 40],
               [80, 80,  0, 40],
               [80, 80, 40, 40]], dtype="uint8")


def _lc_expected():
    # cell x 0.5-2.5, y 1-4: column weights 0.5,1,0.5; row 2 (y 1-2) fully in, row 3 out
    cov = {10: 0.5 + 1 + 0.5, 30: 1, 40: 0.5 + 0.5, 80: 0.5 + 1}
    # nodata pixel at (row 2, col 2) has weight 0.5 -> excluded
    valid = sum(cov.values())
    total = 2 * 3
    frac = {k: v / valid for k, v in cov.items()}
    shannon = -sum(p * math.log(p) for p in frac.values())
    return frac, shannon, valid / total


def test_landcover_fractions_hand_computed_with_partial_pixels_and_nodata():
    with tempfile.TemporaryDirectory() as d:
        p = _write(Path(d) / "lc.tif", LC, nodata=0)
        cells = _cells(c=box(0.5, 1, 2.5, 4))
        cov = ec.zonal_class_coverage(p, cells)
        total = pd.Series(ec.total_coverage(cells, (1, 1)), index=cells.cell_id)
        out = ec.class_fractions(cov, total).set_index("cell_id").loc["c"]
    frac, shannon, vfrac = _lc_expected()
    for k, name in ec.WC_CLASSES.items():
        assert np.isclose(out[f"lc_{name}"], frac.get(k, 0.0)), (name, out[f"lc_{name}"])
    assert np.isclose(out["lc_shannon"], shannon)
    assert np.isclose(out["lc_valid_frac"], vfrac)
    assert np.isclose(sum(out[f"lc_{n}"] for n in ec.WC_CLASSES.values()), 1.0)


def test_landcover_tiles_combine_to_single_raster_answer():
    """A cell straddling two tiles must get the same fractions as if the tiles
    were one raster -- coverage is summed, not fractions averaged."""
    with tempfile.TemporaryDirectory() as d:
        whole = _write(Path(d) / "whole.tif", LC, nodata=0)
        west = _write(Path(d) / "w.tif", LC[:, :2].copy(), left=0, nodata=0)
        east = _write(Path(d) / "e.tif", LC[:, 2:].copy(), left=2, nodata=0)
        cells = _cells(c=box(0.5, 0.5, 3.5, 3.5))
        total = pd.Series(ec.total_coverage(cells, (1, 1)), index=cells.cell_id)
        one = ec.class_fractions(ec.zonal_class_coverage(whole, cells), total)
        two = ec.class_fractions(pd.concat([ec.zonal_class_coverage(west, cells),
                                            ec.zonal_class_coverage(east, cells)]), total)
    pd.testing.assert_frame_equal(one, two, check_exact=False, rtol=1e-9)


def test_unknown_class_value_is_an_error_not_dropped():
    cov = pd.DataFrame({"cell_id": ["c", "c"], "value": [10, 77], "coverage": [1.0, 1.0]})
    try:
        ec.class_fractions(cov, pd.Series([2.0], index=["c"]))
    except ValueError as e:
        assert "77" in str(e)
    else:
        raise AssertionError("class 77 silently accepted")


# --- terrain -----------------------------------------------------------------
def _plane(lat0, kx, ky, n=7, res=1 / 1200):
    """Elevation on a lon/lat grid whose metric gradient is exactly (kx, ky)."""
    lats = lat0 + (n // 2 - np.arange(n)) * res          # north to south
    lons = (np.arange(n) - n // 2) * res
    dx, dy = ec.metric_spacing(lats, res, res)
    x = lons[None, :] / res * dx[:, None]                # metres east, row-wise
    # metres north: integrate the meridional spacing (exact for a smooth plane)
    y = np.cumsum(np.r_[0, -dy[:-1] / 2 - dy[1:] / 2])[:, None] * np.ones((1, n))
    return kx * x + ky * y, lats


def test_slope_matches_known_planes_at_two_latitudes():
    for lat0 in (10.0, 30.0, 60.0):
        for kx, ky in ((0.1, 0.0), (0.0, 0.1), (0.3, -0.4)):
            z, lats = _plane(lat0, kx, ky)
            s = ec.slope_deg(z, lats[1:-1], 1 / 1200, 1 / 1200)
            expected = math.degrees(math.atan(math.hypot(kx, ky)))
            assert np.allclose(s, expected, atol=0.01), (lat0, kx, ky, s.mean(), expected)


def test_degree_spacing_would_be_detectably_wrong():
    """Guard that the plane test can fail: east-west spacing taken as the
    north-south spacing (ignoring cos(lat)) misstates slope at 60N."""
    z, lats = _plane(60.0, 0.1, 0.0)
    _, dy = ec.metric_spacing(lats[1:-1], 1 / 1200, 1 / 1200)
    naive = np.degrees(np.arctan(np.abs((z[1:-1, 2:] - z[1:-1, :-2]) / (2 * dy[:, None]))))
    assert abs(naive.mean() - math.degrees(math.atan(0.1))) > 2.0


def test_vrt_gap_reads_as_nodata_not_zero():
    with tempfile.TemporaryDirectory() as d:
        a = _write(Path(d) / "a.tif", np.full((2, 2), 5.0, "float32"), left=0, top=2)
        b = _write(Path(d) / "b.tif", np.full((2, 2), 7.0, "float32"), left=4, top=2)
        vrt = ec.build_vrt([a, b], Path(d) / "m.vrt")
        with rasterio.open(vrt) as s:
            row = s.read(1)[0]
    assert list(row[:2]) == [5, 5] and list(row[4:]) == [7, 7]
    assert np.isnan(row[2:4]).all(), row


# --- geometry ----------------------------------------------------------------
def test_extraction_cells_clip_to_land_and_land_in_target_crs():
    from grid import INDIA_AEA, build_grid
    land = gpd.GeoDataFrame(geometry=[box(0, 0, 25_000, 20_000)], crs=INDIA_AEA)
    grid, _ = build_grid(land, cell_km=10, min_land_fraction=0.5)
    cells = ec.extraction_cells(grid, land, INDIA_AEA)
    clipped = cells.set_index("cell_id").geometry.area
    expected = grid.set_index("cell_id").eval("land_fraction * area_km2 * 1e6")
    assert np.allclose(clipped.loc[expected.index], expected, rtol=1e-9)
    geo = ec.extraction_cells(grid, land, "EPSG:4326")
    assert geo.crs.to_epsg() == 4326
    # A clipped coastal cell need not contain its full-square centroid (here it
    # lies exactly on the coastline), so test with a point inside each clipped
    # polygon, carried to lon/lat independently of the polygon reprojection.
    inside = cells.geometry.representative_point().to_crs("EPSG:4326")
    assert geo.contains(inside.set_axis(geo.index)).all()


# --- real data ---------------------------------------------------------------
KNOWN = {  # lon, lat, elevation m, tolerance m
    "Leh": (77.5771, 34.1526, 3500, 250),
    "Chennai": (80.2707, 13.0827, 6, 30),
    "Darjeeling": (88.2627, 27.0360, 2040, 250),
}


def _need(*paths):
    for p in paths:
        if not Path(p).exists():
            raise unittest.SkipTest(f"{Path(p).name} not built yet")


def test_grid_meta_locates_every_cell_by_arithmetic():
    _need(ec.GRID, ec.INTERIM / "grid_meta.json")
    from build_grid import load_meta
    from incidence import assign_cells
    grid = gpd.read_file(ec.GRID, layer="grid")
    got = assign_cells(grid.lon.to_numpy(), grid.lat.to_numpy(), load_meta(),
                       valid_cells=set(grid.cell_id))
    assert (got == grid.cell_id.to_numpy()).all()


def test_known_location_point_elevations():
    _need(ec.DEM_VRT)
    with rasterio.open(ec.DEM_VRT) as s:
        vals = {k: float(v[0]) for k, v in zip(KNOWN, s.sample([(x, y) for x, y, *_ in KNOWN.values()]))}
    bad = {k: round(vals[k]) for k, (_, _, z, tol) in KNOWN.items() if abs(vals[k] - z) > tol}
    assert not bad, f"DEM at known places off: {bad} (expected {[(k, v[2]) for k, v in KNOWN.items()]})"


def _env_table():
    """Full table if built, else the climate/terrain half (land cover pending)."""
    path = ec.ENV if ec.ENV.exists() else ec.ENV_CT
    _need(path)
    return path


def test_known_location_cell_elevations():
    _need(ec.GRID)
    from build_grid import load_meta
    from incidence import assign_cells
    env = pd.read_parquet(_env_table()).set_index("cell_id")
    ids = assign_cells([v[0] for v in KNOWN.values()], [v[1] for v in KNOWN.values()],
                       load_meta(), valid_cells=set(env.index))
    rows = {k: env.loc[c] for k, c in zip(KNOWN, ids)}
    for k, (_, _, z, tol) in KNOWN.items():
        r = rows[k]
        assert r.elev_min - tol <= z <= r.elev_max + tol, (k, r.elev_min, r.elev_max)
    assert rows["Leh"].elev_mean > 3000
    assert rows["Chennai"].elev_mean < 50
    assert 1000 < rows["Darjeeling"].elev_mean < 2600


def test_unwrap_recovers_wrapped_quarterly_precip():
    """Synthetic: true wettest-quarter totals above the uint16 ceiling wrap; the
    rule must restore exactly those and leave genuine small values alone."""
    true16 = np.array([70000, 64000, 66000, 3000])
    true18 = np.array([70000, 500, 400, 3000])          # pixel 0: warmest quarter = wettest
    b13 = np.array([26000, 25000, 26500, 1200])
    wrap = lambda a: a % 65536
    u16, u18, w16, w18 = ec.unwrap_quarterly_precip(b13, wrap(true16), wrap(true18))
    assert (u16 == true16).all(), u16
    assert (u18 == true18).all(), u18
    assert w16.tolist() == [True, False, True, False]


def test_chelsa_precip_identities_hold_after_fix():
    _need(*(ec.chelsa_path(i) for i in (12, 13, 19)), ec.chelsa_source(16), ec.chelsa_source(18))
    import geopandas as gpd
    from rasterio.features import geometry_mask
    from scipy.ndimage import maximum_filter
    r = {}
    for i in (12, 13, 19, 16, 18):
        with rasterio.open(ec.chelsa_source(i)) as s:
            r[i] = s.read(1).astype("int64")
            tr, sh = s.transform, s.shape
    land = ~geometry_mask([gpd.read_file(ec.BOUNDARY).geometry.iloc[0]], out_shape=sh, transform=tr)
    for name, ok in (("bio16 >= bio13", r[16] >= r[13]), ("bio16 >= bio19", r[16] >= r[19]),
                     ("bio16 >= bio18", r[16] >= r[18])):
        assert ok[land].all(), f"{name} fails in {(~ok & land).sum()} pixels"
    for i in (16, 18):
        nb = maximum_filter(r[i], size=3)
        sig = land & (nb > 60000) & (r[i] < 0.3 * nb)
        assert not sig.any(), f"bio{i}: {sig.sum()} pixels still look wrapped"


def _chelsa_land_sample(n=20000, seed=0):
    import geopandas as gpd
    from rasterio.features import geometry_mask
    out = {}
    for i in (2, 3, 4, 7, 10, 11):
        with rasterio.open(ec.chelsa_path(i)) as s:
            a = s.read(1).astype("float64") * s.scales[0] + s.offsets[0]
            tr, sh = s.transform, s.shape
        out[i] = a * ec.CHELSA_UNIT_FIX.get(i, 1.0)
    land = ~geometry_mask([gpd.read_file(ec.BOUNDARY).geometry.iloc[0]], out_shape=sh, transform=tr)
    idx = np.random.default_rng(seed).choice(np.flatnonzero(land), n, replace=False)
    return {i: a.ravel()[idx] for i, a in out.items()}


def test_chelsa_unit_corrections_match_physical_identities():
    """If CHELSA republishes bio3/bio4 with different scaling, this fails."""
    _need(*(ec.chelsa_path(i) for i in (2, 3, 4, 7, 10, 11)))
    b = _chelsa_land_sample()
    iso = 100 * b[2] / b[7]
    # bio2 and bio7 are stored to 0.1 degC, so the reference itself carries rounding
    # error, large where bio7 is small. Bound it rather than use a flat tolerance.
    bound = 100 * (0.05 / b[7] + b[2] * 0.05 / b[7] ** 2) + 0.05
    assert (np.abs(b[3] - iso) <= bound).all(), np.percentile(b[3] / iso, [1, 50, 99])
    assert 15 < np.median(b[3]) < 80                                  # percent, not a ratio
    r = (b[10] - b[11]) / b[4]                                       # sinusoid: 2.55
    assert 2.3 < np.median(r) < 2.8, np.median(r)
    assert 0.2 < np.median(b[4]) < 15                                # degC, not degC x 100


def test_every_grid_cell_appears_exactly_once():
    _need(ec.GRID)
    env = pd.read_parquet(_env_table(), columns=["cell_id"])
    grid = gpd.read_file(ec.GRID, layer="grid", columns=["cell_id"], ignore_geometry=True)
    assert len(env) == len(grid) and not env.cell_id.duplicated().any()
    assert set(env.cell_id) == set(grid.cell_id)


def test_landcover_fractions_sum_to_one_where_fully_covered():
    _need(ec.ENV)
    env = pd.read_parquet(ec.ENV)
    full = env[env.lc_valid_frac >= 0.999]
    assert len(full) > 0.9 * len(env), f"only {len(full)} of {len(env)} cells fully covered"
    s = full[[f"lc_{n}" for n in ec.WC_CLASSES.values()]].sum(axis=1)
    assert np.allclose(s, 1.0, atol=1e-6), s.describe()


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
