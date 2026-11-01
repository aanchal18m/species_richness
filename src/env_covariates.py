"""Part 3: environmental covariates per grid cell -> data/interim/env.parquet

Sources (all EPSG:4326)
  CHELSA V2.1 bioclim 1981-2010, 30 arc-sec. India rows read by HTTP range
      request from the global files and stored as exact pixel subsets.
  Copernicus DEM GLO-90 (3 arc-sec COG tiles). Tiles touching India downloaded.
  ESA WorldCover 2021 v200, 10 m. Streamed one tile at a time and aggregated;
      nothing at 10 m is kept. The built-in overviews are NOT used: they are
      majority-resampled, which understates fragmented classes.

Extraction rules
  * The grid is reprojected into each raster's CRS (edges densified first);
    rasters are never resampled.
  * Area-weighted: exactextract weights each pixel by the fraction of it that
    lies inside the cell polygon.
  * Each cell is clipped to the India boundary before extraction. CHELSA has
    modelled values over the sea and the DEM has ~0 m there, so a whole-square
    extraction would silently blend ocean into coastal cells. The sea share is
    grid.land_fraction; per-source valid_frac reports raster gaps on land.
  * Slope is computed on the native lon/lat DEM with the true metric spacing of
    each row (WGS84 radii of curvature): the local projected slope, without
    resampling elevation.
  * Pixel areas vary by ~0.1% across a 10 km cell at 30N; ignored.

Stages
  fetch-chelsa | fetch-dem | slope | worldcover | assemble | report | all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("CPL_VSIL_CURL_CHUNK_SIZE", str(4 << 20))
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "3")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
GRID = INTERIM / "grid.gpkg"
BOUNDARY = RAW / "india_boundary.gpkg"
CHELSA_DIR = RAW / "chelsa"
CHELSA_FIXED = INTERIM / "chelsa_fixed"
DEM_DIR = RAW / "copdem90"
DEM_VRT = INTERIM / "copdem90.vrt"
SLOPE_DIR = INTERIM / "copdem90_slope"
SLOPE_VRT = INTERIM / "copdem90_slope.vrt"
WC_TMP = INTERIM / "worldcover_tiles"
WC_COV = INTERIM / "worldcover_cell_coverage.parquet"
ENV = INTERIM / "env.parquet"
ENV_CT = INTERIM / "env_climate_terrain.parquet"  # climate + terrain only; land cover joins in `assemble`
SUMMARY = ROOT / "outputs" / "tables" / "env_summary.csv"
REPORT = ROOT / "outputs" / "tables" / "env_report.json"

CHELSA_URL = ("https://os.zhdk.cloud.switch.ch/chelsav2/GLOBAL/climatologies/"
              "1981-2010/bio/CHELSA_bio{i}_1981-2010_V.2.1.tif")
DEM_BASE = "https://copernicus-dem-90m.s3.amazonaws.com"
WC_URL = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
          "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")
WC_CLASSES = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare",
              70: "snow", 80: "water", 90: "wetland", 95: "mangrove", 100: "moss"}
# Unit corrections applied AFTER the file's own scale/offset, established against
# physical identities on the India subsets (see tests/test_env_covariates.py):
#   bio3  isothermality: stored float32 already /10 yet tagged scale 0.1, so the
#         scaled value is a 0-1 ratio. raw / (100*bio2/bio7) = 0.1000 (IQR
#         0.0997-0.1002). x100 -> percent.
#   bio4  temperature seasonality: SD of monthly means x 100 (ANUCLIM convention).
#         (bio10-bio11)/(bio4/100) has median 2.58 vs 2.55 for a sinusoidal cycle.
#         x0.01 -> degC, so SHAP curves read in real units.
CHELSA_UNIT_FIX = {3: 100.0, 4: 0.01}
CHELSA_UNITS = {1: "degC", 2: "degC", 3: "%", 4: "degC (SD)", 5: "degC", 6: "degC", 7: "degC",
                8: "degC", 9: "degC", 10: "degC", 11: "degC", 12: "mm/yr", 13: "mm/month",
                14: "mm/month", 15: "% (CV)", 16: "mm/quarter", 17: "mm/quarter",
                18: "mm/quarter", 19: "mm/quarter"}
# CHELSA stores precipitation as uint16 (x0.1 mm), and two layers overflow in
# India's wettest places in DIFFERENT ways (Western Ghats, Meghalaya):
#   bio12 annual        CLIPS at 65535 (6553.5 mm): 13,909 land pixels. The true value
#                       is lost; bio12_saturated_frac marks the cells where bio12 is a
#                       lower bound.
#   bio16 wettest qtr   WRAPS modulo 65536: Mawsynram reads 34.7 mm beside 6,552 mm,
#                       while its wettest month alone is 2,663 mm. Recoverable exactly:
#                       a quarter cannot hold less rain than its wettest month.
#   bio18 warmest qtr   wraps where it is the same quarter as bio16.
UINT16_WRAP = 65536
WC_RES = (1 / 12000, 1 / 12000)  # 3-degree tiles of 36000 px; checked per tile when streamed
MARGIN_DEG = 0.1

# Physical plausibility over India, in final units. Wide on purpose: the point is
# to catch a factor-of-10 scaling slip or a sentinel leaking in, not to police
# the climatology.
PLAUSIBLE = {
    "bio01": (-30, 32), "bio02": (3, 25), "bio03": (10, 90), "bio04": (0.2, 15),
    "bio05": (-10, 50), "bio06": (-45, 26), "bio07": (5, 50), "bio08": (-25, 36),
    "bio09": (-35, 34), "bio10": (-15, 38), "bio11": (-40, 30), "bio12": (20, 13000),
    "bio13": (5, 4500), "bio14": (0, 300), "bio15": (5, 250), "bio16": (10, 10000),
    "bio17": (0, 900), "bio18": (0, 8000), "bio19": (0, 3000),
    "elev_mean": (-20, 7500), "elev_min": (-60, 7500), "elev_max": (-20, 8700),
    "elev_range": (0, 6000), "elev_sd": (0, 1500), "slope_mean": (0, 60),
    "lc_shannon": (0, float(np.log(len(WC_CLASSES)))),
}


# --- geometry ---------------------------------------------------------------
def extraction_cells(grid, boundary, dst_crs, densify_m: float = 500.0):
    """Cell polygons clipped to land, edges densified, in the raster's CRS."""
    import geopandas as gpd
    import shapely

    land = boundary.to_crs(grid.crs).union_all()
    geoms = grid.geometry.to_numpy().copy()
    edge = grid["land_fraction"].to_numpy() < 1 - 1e-12
    geoms[edge] = shapely.intersection(geoms[edge], land)
    geoms = shapely.segmentize(geoms, densify_m)
    return gpd.GeoDataFrame({"cell_id": grid["cell_id"].to_numpy()}, geometry=geoms,
                            crs=grid.crs).to_crs(dst_crs)


def total_coverage(cells, res) -> np.ndarray:
    """Polygon area in pixel units, computed in the raster's own (planar) CRS --
    the denominator exactextract's coverage fractions are measured against."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # geographic-CRS area warning: intended here
        return cells.geometry.area.to_numpy() / (abs(res[0]) * abs(res[1]))


# --- zonal statistics -------------------------------------------------------
def zonal_continuous(raster, cells) -> pd.DataFrame:
    """Coverage-weighted mean, population SD, min, max and valid fraction per cell.

    Values come back in physical units: exactextract applies the raster's own
    GeoTIFF scale/offset (CHELSA bio1: raw K*10 -> scale 0.1, offset -273.15).
    Do not apply them again -- tests/test_env_covariates.py caught exactly that."""
    import rasterio
    from exactextract import exact_extract

    with rasterio.open(raster) as src:
        res = src.res
    df = exact_extract(str(raster), cells[["cell_id", "geometry"]],
                       ["mean", "stdev", "min", "max", "count"],
                       include_cols=["cell_id"], output="pandas")
    df = cells[["cell_id"]].merge(df, on="cell_id", how="left", validate="one_to_one")
    valid = df["count"].fillna(0).to_numpy() / total_coverage(cells, res)
    if (valid > 1 + 1e-6).any():
        raise AssertionError("valid coverage exceeds polygon area; CRS mismatch?")
    has = df["count"].fillna(0).to_numpy() > 0
    return pd.DataFrame({
        "cell_id": df["cell_id"],
        "mean": np.where(has, df["mean"], np.nan),
        "sd": np.where(has, df["stdev"], np.nan),
        "min": np.where(has, df["min"], np.nan),
        "max": np.where(has, df["max"], np.nan),
        "valid_frac": np.minimum(valid, 1.0),
    })


def zonal_class_coverage(raster, cells) -> pd.DataFrame:
    """Per-cell coverage (in pixel units) of each class value. Additive, so
    results from adjacent tiles can be summed."""
    from exactextract import exact_extract

    df = exact_extract(str(raster), cells[["cell_id", "geometry"]], ["unique", "frac", "count"],
                       include_cols=["cell_id"], output="pandas")
    rows = []
    for cid, vals, fracs, n in zip(df["cell_id"], df["unique"], df["frac"], df["count"]):
        if not n or len(vals) == 0:
            continue
        for v, f in zip(vals, fracs):
            rows.append((cid, int(v), float(f) * float(n)))
    return pd.DataFrame(rows, columns=["cell_id", "value", "coverage"])


def class_fractions(coverage: pd.DataFrame, total_cov: pd.Series, classes=WC_CLASSES) -> pd.DataFrame:
    """Combine per-tile class coverage into fractions of valid area, Shannon
    diversity, and valid fraction. `total_cov` is indexed by cell_id."""
    unexpected = set(coverage["value"]) - set(classes)
    if unexpected:
        raise ValueError(f"class values not in legend: {sorted(unexpected)}")
    wide = coverage.pivot_table(index="cell_id", columns="value", values="coverage",
                                aggfunc="sum", fill_value=0.0)
    wide = wide.reindex(columns=list(classes), fill_value=0.0)
    valid = wide.sum(axis=1)
    frac = wide.div(valid, axis=0)
    p = frac.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        shannon = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    out = frac.rename(columns={k: f"lc_{v}" for k, v in classes.items()})
    out["lc_shannon"] = shannon
    out["lc_valid_frac"] = np.minimum((valid / total_cov.reindex(valid.index)).to_numpy(), 1.0)
    return out.rename_axis("cell_id").reset_index()


# --- terrain ----------------------------------------------------------------
A_WGS84 = 6378137.0
F_WGS84 = 1 / 298.257223563
E2_WGS84 = F_WGS84 * (2 - F_WGS84)


def metric_spacing(lat_deg, dlon_deg, dlat_deg):
    """Ground distance (m) of one pixel step east and north at each latitude."""
    phi = np.radians(np.asarray(lat_deg, dtype=float))
    w = 1 - E2_WGS84 * np.sin(phi) ** 2
    n = A_WGS84 / np.sqrt(w)                       # prime-vertical radius
    m = A_WGS84 * (1 - E2_WGS84) / w ** 1.5        # meridional radius
    return n * np.cos(phi) * np.radians(abs(dlon_deg)), m * np.radians(abs(dlat_deg))


def slope_deg(z_halo, lat_rows, dlon_deg, dlat_deg):
    """Horn (1981) slope in degrees. `z_halo` has a one-pixel border; rows run
    north to south; `lat_rows` are the centre latitudes of the inner rows.
    NaN in any of the 3x3 neighbours gives NaN."""
    z = np.asarray(z_halo, dtype="float64")
    a, b, c = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    d, f = z[1:-1, :-2], z[1:-1, 2:]
    g, h, i = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]
    dx, dy = metric_spacing(lat_rows, dlon_deg, dlat_deg)
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * dx[:, None])
    dzdy = ((a + 2 * b + c) - (g + 2 * h + i)) / (8 * dy[:, None])
    return np.degrees(np.arctan(np.hypot(dzdx, dzdy)))


def build_vrt(paths, out: Path, nodata="nan"):
    """Mosaic same-resolution, non-overlapping tiles into a VRT. Areas no tile
    covers read as nodata, never as 0."""
    import rasterio

    infos = []
    for p in paths:
        with rasterio.open(p) as s:
            infos.append((Path(p), s.transform, s.width, s.height, s.dtypes[0], s.crs))
    res = infos[0][1].a, infos[0][1].e
    if any(abs(t.a - res[0]) > 1e-12 or abs(t.e - res[1]) > 1e-12 for _, t, *_ in infos):
        raise ValueError("tiles differ in resolution")
    left = min(t.c for _, t, *_ in infos)
    top = max(t.f for _, t, *_ in infos)
    right = max(t.c + w * t.a for _, t, w, *_ in infos)
    bottom = min(t.f + h * t.e for _, t, _, h, *_ in infos)
    width, height = round((right - left) / res[0]), round((bottom - top) / res[1])
    dtype = {"float32": "Float32", "uint8": "Byte", "uint16": "UInt16"}[infos[0][4]]
    src_xml = []
    for p, t, w, h, *_ in infos:
        xo, yo = (t.c - left) / res[0], (t.f - top) / res[1]
        if abs(xo - round(xo)) > 1e-6 or abs(yo - round(yo)) > 1e-6:
            raise ValueError(f"{p.name} is not aligned to the mosaic pixel grid")
        rel = os.path.relpath(p, out.parent)
        src_xml.append(
            f'<SimpleSource><SourceFilename relativeToVRT="1">{rel}</SourceFilename>'
            f'<SourceBand>1</SourceBand><SrcRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>'
            f'<DstRect xOff="{round(xo)}" yOff="{round(yo)}" xSize="{w}" ySize="{h}"/></SimpleSource>')
    xml = (f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">'
           f'<SRS>{infos[0][5].to_wkt()}</SRS>'
           f'<GeoTransform>{left!r}, {res[0]!r}, 0, {top!r}, 0, {res[1]!r}</GeoTransform>'
           f'<VRTRasterBand dataType="{dtype}" band="1"><NoDataValue>{nodata}</NoDataValue>'
           + "".join(src_xml) + "</VRTRasterBand></VRTDataset>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xml)
    return out


# --- acquisition ------------------------------------------------------------
def _download(url: str, dest: Path, session=None, timeout=300) -> Path:
    import requests

    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    s = session or requests
    for attempt in range(5):
        try:
            with s.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                expected = int(r.headers.get("content-length", -1))
                with open(part, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            if expected >= 0 and part.stat().st_size != expected:
                raise IOError(f"{dest.name}: got {part.stat().st_size} of {expected} bytes")
            part.replace(dest)
            return dest
        except Exception:  # noqa: BLE001
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))


def india_bounds():
    import geopandas as gpd
    w, s, e, n = gpd.read_file(BOUNDARY, layer="india").total_bounds
    return w - MARGIN_DEG, s - MARGIN_DEG, e + MARGIN_DEG, n + MARGIN_DEG


def chelsa_path(i: int) -> Path:
    return CHELSA_DIR / f"CHELSA_bio{i}_1981-2010_V.2.1_india.tif"


def chelsa_source(i: int) -> Path:
    """The raster each bioclim variable is extracted from: the overflow-corrected
    copy for bio16/bio18, the untouched subset otherwise."""
    return CHELSA_FIXED / f"bio{i}_unwrapped.tif" if i in (16, 18) else chelsa_path(i)


def unwrap_quarterly_precip(b13, b16, b18):
    """Undo uint16 wraparound in bio16 and bio18 (raw integer arrays).

    bio16 < bio13 is impossible (the wettest quarter contains the wettest month),
    so it marks a wrapped value. bio18 is unwrapped where it equals the wrapped
    bio16, i.e. the warmest quarter is the wettest one."""
    b13, b16, b18 = (np.asarray(a, dtype="int64") for a in (b13, b16, b18))
    w16 = b16 < b13
    w18 = w16 & (b18 == b16)
    return b16 + UINT16_WRAP * w16, b18 + UINT16_WRAP * w18, w16, w18


def fix_chelsa():
    import rasterio

    def read(i):
        with rasterio.open(chelsa_path(i)) as s:
            return s.read(1), s.profile, s.scales[0], s.offsets[0]
    b13, prof, *_ = read(13)
    b16, _, sc16, of16 = read(16)
    b18, _, sc18, of18 = read(18)
    b12, *_ = read(12)
    u16, u18, w16, w18 = unwrap_quarterly_precip(b13, b16, b18)
    CHELSA_FIXED.mkdir(parents=True, exist_ok=True)
    for i, arr, sc, of in ((16, u16, sc16, of16), (18, u18, sc18, of18)):
        out = chelsa_source(i)
        with rasterio.open(out, "w", **{**prof, "dtype": "uint32", "predictor": 2}) as d:
            d.write(arr.astype("uint32"), 1)
            d.scales, d.offsets = (sc,), (of,)
            d.update_tags(derived_from=chelsa_path(i).name, fix="uint16 wraparound undone where bio16 < bio13")
    with rasterio.open(CHELSA_FIXED / "bio12_saturated.tif", "w",
                       **{**prof, "dtype": "uint8", "nodata": None}) as d:
        d.write((b12 == np.iinfo("uint16").max).astype("uint8"), 1)
    print(f"   unwrapped bio16 in {int(w16.sum())} pixels, bio18 in {int(w18.sum())}; "
          f"bio12 clipped in {int((b12 == 65535).sum())} pixels (whole subset, incl. sea)")


def fetch_chelsa_one(i: int, bounds) -> Path:
    """Exact pixel subset of the global file: same grid, same raw values,
    scale/offset carried in the GeoTIFF metadata."""
    import rasterio
    from rasterio.windows import from_bounds

    out = chelsa_path(i)
    if out.exists():
        return out
    url = CHELSA_URL.format(i=i)
    with rasterio.open("/vsicurl/" + url) as src:
        win = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
        data = src.read(1, window=win)
        prof = dict(driver="GTiff", width=data.shape[1], height=data.shape[0], count=1,
                    dtype=data.dtype, crs=src.crs, transform=src.window_transform(win),
                    compress="deflate", predictor=2, tiled=True, blockxsize=512, blockysize=512)
        scale, offset = src.scales[0], src.offsets[0]
    tmp = out.with_suffix(".tif.part")
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(data, 1)
        dst.scales, dst.offsets = (scale,), (offset,)
        dst.update_tags(source=url, subset_window=str(win))
    tmp.replace(out)
    return out


def fetch_chelsa():
    b = india_bounds()
    with ThreadPoolExecutor(4) as ex:
        for p in ex.map(lambda i: fetch_chelsa_one(i, b), range(1, 20)):
            print("  ", p.name, flush=True)


def dem_tiles_for_india():
    import geopandas as gpd
    import requests
    from shapely.geometry import box

    india = gpd.read_file(BOUNDARY, layer="india").geometry.iloc[0]
    names = requests.get(f"{DEM_BASE}/tileList.txt", timeout=60).text.split()

    def tbox(n):
        p = n.split("_")
        lat = float(p[4][1:]) * (1 if p[4][0] == "N" else -1)
        lon = float(p[6][1:]) * (1 if p[6][0] == "E" else -1)
        return box(lon, lat, lon + 1, lat + 1)
    return [n for n in names if tbox(n).intersects(india)]


def fetch_dem():
    import requests

    names = dem_tiles_for_india()
    s = requests.Session()
    done = 0
    with ThreadPoolExecutor(8) as ex:
        for _ in ex.map(lambda n: _download(f"{DEM_BASE}/{n}/{n}.tif", DEM_DIR / f"{n}.tif", s), names):
            done += 1
            if done % 50 == 0:
                print(f"   {done}/{len(names)} DEM tiles", flush=True)
    build_vrt(sorted(DEM_DIR.glob("*.tif")), DEM_VRT)
    print(f"   {len(names)} tiles; mosaic {DEM_VRT.name}")


def compute_slope():
    import rasterio
    from rasterio.windows import Window

    SLOPE_DIR.mkdir(parents=True, exist_ok=True)
    tiles = sorted(DEM_DIR.glob("*.tif"))
    with rasterio.open(DEM_VRT) as mosaic:
        for k, t in enumerate(tiles):
            out = SLOPE_DIR / t.name.replace("_DEM", "_SLOPE")
            if out.exists():
                continue
            with rasterio.open(t) as src:
                tr, w, h, prof = src.transform, src.width, src.height, src.profile
            col = round((tr.c - mosaic.transform.c) / mosaic.transform.a)
            row = round((tr.f - mosaic.transform.f) / mosaic.transform.e)
            z = mosaic.read(1, window=Window(col - 1, row - 1, w + 2, h + 2),
                            boundless=True, fill_value=np.nan).astype("float64")
            lat = tr.f + (np.arange(h) + 0.5) * tr.e
            s = slope_deg(z, lat, tr.a, tr.e).astype("float32")
            prof.update(dtype="float32", nodata=np.nan, compress="deflate", predictor=3,
                        tiled=True, blockxsize=512, blockysize=512)
            tmp = out.with_suffix(".tif.part")
            with rasterio.open(tmp, "w", **prof) as dst:
                dst.write(s, 1)
            tmp.replace(out)
            if (k + 1) % 50 == 0:
                print(f"   slope {k + 1}/{len(tiles)}", flush=True)
    build_vrt(sorted(SLOPE_DIR.glob("*.tif")), SLOPE_VRT)


def worldcover_tiles():
    import geopandas as gpd
    from shapely.geometry import box

    india = gpd.read_file(BOUNDARY, layer="india").geometry.iloc[0]
    return [f"N{lat:02d}E{lon:03d}" for lat in range(0, 39, 3) for lon in range(66, 99, 3)
            if box(lon, lat, lon + 3, lat + 3).intersects(india)]


def stream_worldcover(cells):
    """Download a tile, aggregate it, delete it. A per-tile checkpoint makes
    the stage restartable; one download runs ahead of processing."""
    import requests
    from shapely.geometry import box

    tiles = worldcover_tiles()
    WC_TMP.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    todo = [t for t in tiles if not (WC_TMP / f"{t}.parquet").exists()]
    lock = threading.Lock()
    with ThreadPoolExecutor(2) as ex:
        futures = {t: ex.submit(_download, WC_URL.format(tile=t), WC_TMP / f"{t}.tif", s) for t in todo[:2]}
        for k, t in enumerate(todo):
            path = futures.pop(t).result()
            import rasterio
            with rasterio.open(path) as src:
                if not np.allclose(np.abs(src.res), WC_RES, rtol=1e-9):
                    raise ValueError(f"{t}: resolution {src.res} != {WC_RES}")
            if k + 2 < len(todo):
                nxt = todo[k + 2]
                futures[nxt] = ex.submit(_download, WC_URL.format(tile=nxt), WC_TMP / f"{nxt}.tif", s)
            lat, lon = int(t[1:3]), int(t[4:7])
            sub = cells[cells.intersects(box(lon, lat, lon + 3, lat + 3))]
            cov = zonal_class_coverage(path, sub) if len(sub) else pd.DataFrame(
                columns=["cell_id", "value", "coverage"])
            cov.assign(tile=t).to_parquet(WC_TMP / f"{t}.parquet")
            with lock:
                path.unlink()
            print(f"   WorldCover {t}: {len(sub)} cells ({len(tiles) - len(todo) + k + 1}/{len(tiles)})",
                  flush=True)
    parts = [pd.read_parquet(WC_TMP / f"{t}.parquet") for t in tiles]
    pd.concat(parts, ignore_index=True).to_parquet(WC_COV)


# --- assembly ---------------------------------------------------------------
def load_cells(dst_crs="EPSG:4326"):
    import geopandas as gpd
    grid = gpd.read_file(GRID, layer="grid")
    boundary = gpd.read_file(BOUNDARY, layer="india")
    return grid, extraction_cells(grid, boundary, dst_crs)


def check_one_row_per_cell(env, grid):
    if len(env) != len(grid) or env["cell_id"].duplicated().any() or set(env.cell_id) != set(grid.cell_id):
        raise AssertionError("table does not have exactly one row per grid cell")


def climate_terrain(grid=None, cells=None):
    """CHELSA + DEM + slope per cell -> ENV_CT. Independent of land cover, so it
    can be built and checked while WorldCover is still streaming."""
    import rasterio

    if grid is None:
        grid, cells = load_cells()
    env = grid[["cell_id"]].copy()

    chelsa_valid = []
    for i in range(1, 20):
        p = chelsa_source(i)
        with rasterio.open(p) as src:
            scale, offset = src.scales[0], src.offsets[0]
        z = zonal_continuous(p, cells)
        env[f"bio{i:02d}"] = z["mean"].to_numpy() * CHELSA_UNIT_FIX.get(i, 1.0)
        chelsa_valid.append(z["valid_frac"].to_numpy())
        print(f"   bio{i:02d} (file scale {scale}, offset {offset}; "
              f"unit fix x{CHELSA_UNIT_FIX.get(i, 1.0)} -> {CHELSA_UNITS[i]})", flush=True)
    env["chelsa_valid_frac"] = np.min(chelsa_valid, axis=0)
    # share of the cell's land where bio12 sat at the uint16 ceiling (bio12 is then a lower bound)
    env["bio12_saturated_frac"] = zonal_continuous(CHELSA_FIXED / "bio12_saturated.tif", cells)["mean"].to_numpy()

    dem = zonal_continuous(DEM_VRT, cells)
    env["elev_mean"], env["elev_sd"] = dem["mean"].to_numpy(), dem["sd"].to_numpy()
    env["elev_min"], env["elev_max"] = dem["min"].to_numpy(), dem["max"].to_numpy()
    env["elev_range"] = env["elev_max"] - env["elev_min"]
    env["dem_valid_frac"] = dem["valid_frac"].to_numpy()
    slope = zonal_continuous(SLOPE_VRT, cells)
    env["slope_mean"] = slope["mean"].to_numpy()
    env["slope_valid_frac"] = slope["valid_frac"].to_numpy()
    print("   DEM and slope", flush=True)
    check_one_row_per_cell(env, grid)
    env.to_parquet(ENV_CT, index=False)
    print(f"   {ENV_CT.relative_to(ROOT)}: {len(env):,} rows x {env.shape[1]} columns")
    return env


def assemble():
    """Climate/terrain (reused if built) + land cover -> ENV, the Part 3 output."""
    if not WC_COV.exists():
        raise FileNotFoundError(f"{WC_COV.name} missing: finish the `worldcover` stage first")
    grid, cells = load_cells()
    env = pd.read_parquet(ENV_CT) if ENV_CT.exists() else climate_terrain(grid, cells)
    total = pd.Series(total_coverage(cells, WC_RES), index=cells["cell_id"].to_numpy())
    lc = class_fractions(pd.read_parquet(WC_COV), total)
    env = env.merge(lc, on="cell_id", how="left", validate="one_to_one")
    env["lc_valid_frac"] = env["lc_valid_frac"].fillna(0.0)
    check_one_row_per_cell(env, grid)
    env.to_parquet(ENV, index=False)
    write_provenance()
    print(f"   {ENV.relative_to(ROOT)}: {len(env):,} rows x {env.shape[1]} columns")
    return env


def write_provenance():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gbif_download import README, write_section

    n_dem = len(list(DEM_DIR.glob("*.tif")))
    write_section(README, "chelsa", f"""## CHELSA V2.1 bioclimatic variables (Part 3)

- Used for all 19 bioclim predictors (bio1-bio19), climatology 1981-2010, 30 arc-sec (~1 km)
- Files: `chelsa/CHELSA_bio{{1..19}}_1981-2010_V.2.1_india.tif` -- exact pixel subsets
  (India bounds + {MARGIN_DEG} deg) of the global files, read by HTTP range request; raw
  integer values unchanged, GeoTIFF scale/offset carried over (e.g. bio1: 0.1, -273.15)
- Source: <{CHELSA_URL.format(i='N')}>
- Note: CHELSA has modelled values over the sea and declares no nodata
- Corrections applied in `src/env_covariates.py` (raw files untouched):
  bio3 x100 (stored /10 yet tagged scale 0.1) and bio4 x0.01 (SD x100) to physical units;
  bio16/bio18 uint16 wraparound undone (`fix-chelsa` -> `data/interim/chelsa_fixed/`);
  bio12 clipping at 6553.5 mm cannot be undone and is flagged per cell as
  `bio12_saturated_frac`. See `outputs/limitations.md`.
- Citation: Karger, D.N. et al. (2017) Climatologies at high resolution for the earth's land
  surface areas. *Scientific Data* 4, 170122. Data: https://doi.org/10.16904/envidat.228
- Produced by: `python src/env_covariates.py fetch-chelsa`
""")
    write_section(README, "copdem", f"""## Copernicus DEM GLO-90 (Part 3)

- {n_dem} 1x1 degree COG tiles intersecting India, 3 arc-sec (~90 m), in `copdem90/`
- Source: <{DEM_BASE}/> (AWS Open Data registry)
- Note: sea is filled with ~0 m, not nodata
- Citation: Copernicus DEM GLO-90, (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space
  GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA.
  https://doi.org/10.5270/ESA-c5d3d65
- Produced by: `python src/env_covariates.py fetch-dem`
""")
    write_section(README, "worldcover", f"""## ESA WorldCover 2021 v200 (Part 3) -- streamed, not stored

- 10 m tiles ({len(worldcover_tiles())} 3x3 degree tiles intersecting India) downloaded one at a time,
  aggregated to per-cell class coverage, then deleted. Only the aggregate is kept:
  `data/interim/worldcover_cell_coverage.parquet`.
- The internal overviews were not used: they are majority-resampled (checked on tile
  N27E075: overview followed the block majority in 99% of disagreeing blocks), which
  understated grassland 0.43% -> 0.17% in a test window.
- Source: <{WC_URL.format(tile='<tile>')}>
- Citation: Zanaga, D. et al. (2022) ESA WorldCover 10 m 2021 v200. Zenodo.
  https://doi.org/10.5281/zenodo.7254221 (CC BY 4.0)
- Produced by: `python src/env_covariates.py worldcover`
""")


def summarise(env: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    preds = [c for c in env.columns if c != "cell_id" and not c.endswith("_valid_frac")
             and c != "bio12_saturated_frac"]
    q = env[preds].describe(percentiles=[0.01, 0.5, 0.99]).T
    q["n_null"] = env[preds].isna().sum()
    flags = []
    for col, (lo, hi) in PLAUSIBLE.items():
        if col in env:
            mn, mx = env[col].min(), env[col].max()
            if mn < lo or mx > hi:
                flags.append(f"{col}: observed [{mn:.4g}, {mx:.4g}] outside plausible [{lo}, {hi}]")
    lc_cols = [f"lc_{v}" for v in WC_CLASSES.values()]
    has_lc = all(c in env for c in lc_cols)
    vf = {c: env[c].describe(percentiles=[0.01, 0.05, 0.5]).round(4).to_dict()
          for c in env.columns if c.endswith("_valid_frac")}
    report = {
        "cells": int(len(env)),
        "cells_complete_predictors": int(env[preds].notna().all(axis=1).sum()),
        "cells_with_any_null": {c: int(n) for c, n in env[preds].isna().sum().items() if n},
        "valid_frac": vf,
        "valid_frac_below": {c: {t: int((env[c] < t).sum()) for t in (0.5, 0.9, 0.99)}
                             for c in vf},
        "lc_fraction_sum_max_abs_dev": (float((env[lc_cols].sum(axis=1).dropna() - 1).abs().max())
                                        if has_lc else "land cover pending"),
        "plausibility_flags": flags,
    }
    return q, report


def unit_of(col: str) -> str:
    if col.startswith("bio"):
        return CHELSA_UNITS[int(col[3:])]
    if col.startswith("elev"):
        return "m"
    return {"slope_mean": "degrees", "lc_shannon": "nats"}.get(col, "fraction")


def island_groups(lost: pd.DataFrame) -> dict:
    """Name the omitted boundary parts by region (bounding boxes are coarse but
    unambiguous for India's island groups)."""
    def region(lon, lat):
        if 71.5 <= lon <= 74.5 and 8 <= lat <= 12.5:
            return "Lakshadweep"
        if 92 <= lon <= 94.5 and 6.5 <= lat <= 10:
            return "Nicobar Islands"
        if 92 <= lon <= 94.5 and 10 < lat <= 14.5:
            return "Andaman Islands"
        if 88 <= lon <= 89.5 and 21 <= lat <= 22.5:
            return "Sundarbans delta"
        return "other"
    g = lost.assign(region=[region(x, y) for x, y in zip(lost.lon, lost.lat)])
    return {r: {"parts": int(len(d)), "area_km2": round(float(d.area_km2.sum()), 1)}
            for r, d in g.groupby("region")}


def report():
    import geopandas as gpd
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_grid import omitted_land

    path = ENV if ENV.exists() else ENV_CT
    env = pd.read_parquet(path)
    q, rep = summarise(env)
    rep = {"table": path.name, **rep}
    q.insert(0, "unit", [unit_of(c) for c in q.index])
    lost, outside_km2 = omitted_land(gpd.read_file(GRID, layer="grid"),
                                     gpd.read_file(BOUNDARY, layer="india"))
    rep["grid_omissions"] = {
        "boundary_parts_without_any_cell": int(len(lost)),
        "their_area_km2": round(float(lost.area_km2.sum()), 1),
        "by_region": island_groups(lost),
        "india_land_outside_kept_cells_km2": round(outside_km2),
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    q.to_csv(SUMMARY)
    REPORT.write_text(json.dumps(rep, indent=2, default=float))
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(q[["min", "1%", "50%", "99%", "max", "n_null"]].round(3))
    print(json.dumps(rep, indent=2, default=float))


def main(argv=None):
    stages = {
        "fetch-chelsa": fetch_chelsa,
        "fetch-dem": fetch_dem,
        "slope": compute_slope,
        "worldcover": lambda: stream_worldcover(load_cells()[1]),
        "fix-chelsa": fix_chelsa,
        "climate-terrain": climate_terrain,
        "assemble": assemble,
        "report": report,
    }
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
