"""Tests for src/boundary.py.

Run: python tests/test_boundary.py   (or pytest)

Known-answer checks: the area routine against a closed-form ellipsoid result,
and the boundary against places whose side of the line is not in doubt, plus
places that only fall inside if the official (not de facto) boundary is used.
"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import boundary as b  # noqa: E402

INSIDE = {
    "Delhi": (77.21, 28.61), "Chennai": (80.27, 13.08), "Kanyakumari": (77.54, 8.09),
    "Leh": (77.58, 34.16), "Port Blair (Andamans)": (92.73, 11.62), "Shillong": (91.88, 25.58),
    # Only inside under the official depiction; a de facto boundary fails these.
    "Gilgit": (74.31, 35.92), "Muzaffarabad": (73.47, 34.37), "Aksai Chin": (79.20, 35.20),
    "Tawang": (91.87, 27.59),
}
OUTSIDE = {
    "Kathmandu": (85.32, 27.72), "Dhaka": (90.41, 23.81), "Colombo": (79.86, 6.93),
    "Lahore": (74.34, 31.55), "Thimphu": (89.64, 27.47), "Male": (73.51, 4.18),
    "Kashgar": (75.99, 39.47),
}
OFFICIAL_AREA_KM2 = 3_287_263


def _india():
    if not b.OUT.exists():
        raise unittest.SkipTest("run `python src/boundary.py` first")
    import geopandas as gpd
    return gpd.read_file(b.OUT, layer="india")


def test_geodesic_area_known_answer():
    """Octant bounded by the equator and two meridians: every edge is a geodesic,
    so the polygon is exactly 1/8 of the WGS84 surface, which has a closed form.
    (A lat/lon box is not a fair target: its top edge is a parallel, and the
    geodesic between its corners bows poleward.)"""
    from shapely.geometry import Polygon, box
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    e = math.sqrt(e2)
    surface = 2 * math.pi * a * a * (1 + (1 - e2) / e * math.atanh(e)) / 1e6
    got = b.geodesic_area_km2(Polygon([(0, 0), (90, 0), (0, 90)]))
    assert abs(got - surface / 8) / (surface / 8) < 1e-9, (got, surface / 8)
    # magnitude check on a 1-degree cell: catches m^2 vs km^2 and lon/lat swaps
    assert 12_000 < b.geodesic_area_km2(box(0, 0, 1, 1)) < 12_500


def test_boundary_is_one_valid_feature_in_wgs84():
    india = _india()
    assert len(india) == 1
    assert india.crs.to_epsg() == 4326
    assert india.geometry.is_valid.all()


def test_known_places_fall_on_the_right_side():
    from shapely.geometry import Point
    geom = _india().geometry.iloc[0]
    wrong_out = [k for k, xy in INSIDE.items() if not geom.contains(Point(xy))]
    wrong_in = [k for k, xy in OUTSIDE.items() if geom.contains(Point(xy))]
    assert not wrong_out and not wrong_in, f"should be inside: {wrong_out}; should be outside: {wrong_in}"


def test_area_matches_official_figure():
    area = b.geodesic_area_km2(_india().geometry.iloc[0])
    assert abs(area - OFFICIAL_AREA_KM2) / OFFICIAL_AREA_KM2 < 0.02, f"{area:,.0f} km2"


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
