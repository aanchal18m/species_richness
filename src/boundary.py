"""Part 2: India national boundary -> data/raw/india_boundary.gpkg

Source: Natural Earth 1:10m Admin-0 countries, India point-of-view edition
(ne_10m_admin_0_countries_ind). This edition draws India's boundary as the
Government of India claims it (all of Jammu & Kashmir and Ladakh, including
Aksai Chin, and Arunachal Pradesh). The default Natural Earth and GADM layers
draw de facto lines instead. Public domain, versioned, stable URL.

The output is a lossless subset of the raw zip (one feature selected, no
geometry edits), so it sits in data/raw beside its source.

Run: python src/boundary.py
"""
from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gbif_download import README, write_section  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NE_URL = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries_ind.zip"
RAW_ZIP = ROOT / "data" / "raw" / "naturalearth" / "ne_10m_admin_0_countries_ind.zip"
OUT = ROOT / "data" / "raw" / "india_boundary.gpkg"


def download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(part, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    zipfile.ZipFile(part).testzip()
    part.replace(dest)
    return dest


def extract_india(zip_path: Path):
    import geopandas as gpd

    world = gpd.read_file(f"zip://{zip_path}")
    india = world.loc[world["ADM0_A3"] == "IND", ["ADM0_A3", "NAME", "geometry"]].reset_index(drop=True)
    if len(india) != 1:
        raise ValueError(f"expected one India feature, found {len(india)}")
    if not india.geometry.is_valid.all():
        # Fail rather than repair silently: a repaired boundary is an edited raw file.
        raise ValueError("India geometry is invalid in the source file")
    return india.to_crs("EPSG:4326")


def geodesic_area_km2(geom) -> float:
    """Area on the WGS84 ellipsoid, independent of any projection choice."""
    from pyproj import Geod

    area, _ = Geod(ellps="WGS84").geometry_area_perimeter(geom)
    return abs(area) / 1e6


def ne_version(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.endswith("VERSION.txt"))
        return z.read(name).decode().strip()


def main():
    zip_path = download(NE_URL, RAW_ZIP)
    india = extract_india(zip_path)
    india["source"] = f"Natural Earth {ne_version(zip_path)} ne_10m_admin_0_countries_ind"
    if OUT.exists():
        OUT.unlink()
    india.to_file(OUT, layer="india", driver="GPKG")

    geom = india.geometry.iloc[0]
    area = geodesic_area_km2(geom)
    w, s, e, n = india.total_bounds
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    write_section(README, "boundary", f"""## India national boundary (Part 2)

- File: `india_boundary.gpkg` (layer `india`, EPSG:4326, one MultiPolygon, {len(geom.geoms)} parts)
- Source: Natural Earth 1:10m Admin-0, India point-of-view edition, version {ne_version(zip_path)}
  <{NE_URL}>; raw zip kept at `naturalearth/{zip_path.name}`, SHA-256 `{digest}`
- License: public domain (Natural Earth terms of use)
- Depiction: boundary as claimed by the Government of India, including all of Jammu &
  Kashmir, Ladakh (with Aksai Chin) and Arunachal Pradesh
- Ellipsoidal area: {area:,.0f} km² (official figure 3,287,263 km²)
- Bounds: {w:.3f}–{e:.3f} °E, {s:.3f}–{n:.3f} °N
- Produced by: `python src/boundary.py`

Citation: Natural Earth. Free vector and raster map data @ naturalearthdata.com.
""")
    print(f"{OUT.relative_to(ROOT)}: {area:,.0f} km², bounds {w:.2f}-{e:.2f}E {s:.2f}-{n:.2f}N")


if __name__ == "__main__":
    main()
