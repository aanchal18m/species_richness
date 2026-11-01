"""Validation of cell assignment and incidence assembly."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

from grid import build_grid
from incidence import assign_cells, define_sampling_units, build_incidence
from richness import coverage_standardised_richness, incidence_stats

results = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    results.append(bool(cond)); return cond

# Geometry fixture: a small test area, not India.
area = gpd.GeoDataFrame(geometry=[box(76.0, 12.0, 79.0, 15.0)], crs="EPSG:4326")
grid, meta = build_grid(area, cell_km=10.0, min_land_fraction=0.5)
valid = set(grid.cell_id)
print(f"\nfixture grid: {len(grid):,} cells of {grid.area_km2.iloc[0]:.0f} km2 each")

# -- Test 1: the arithmetic fast path must match a real spatial join ---------
print("\n[1] Lattice arithmetic must agree exactly with a geopandas spatial join")
rng = np.random.default_rng(11)
n = 4000
lon = rng.uniform(76.0, 79.0, n)
lat = rng.uniform(12.0, 15.0, n)

fast = assign_cells(lon, lat, meta, valid_cells=valid)

pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs="EPSG:4326").to_crs(meta["crs"])
joined = gpd.sjoin(pts, grid[["cell_id", "geometry"]], how="left", predicate="within")
joined = joined[~joined.index.duplicated(keep="first")].sort_index()
slow = joined["cell_id"].to_numpy(dtype=object)

# Normalise "no cell" before comparing: the spatial join reports it as NaN,
# assign_cells reports it as None, and NaN != None would read as a mismatch
# when both sides actually agree the point lies outside the grid.
def norm(v):
    return None if (v is None or (isinstance(v, float) and np.isnan(v))) else v

agree = np.array([norm(a) == norm(b) for a, b in zip(fast, slow)])
n_outside = int(sum(norm(a) is None for a in fast))
check("every point assigned identically", bool(agree.all()),
      f"{agree.sum()}/{n} agree")
check("points were actually assigned, not all None", n - n_outside > n * 0.9,
      f"{n - n_outside}/{n} inside grid, {n_outside} outside")

# -- Test 2: sampling unit definitions ---------------------------------------
# GBIF pseudo-unit is (cell, point, day), pooling observers (rule C, Gate 1 review).
print("\n[2] Sampling units must follow the route's rule")
df = pd.DataFrame({
    "eventID":    ["E1", "E1", None, None, None, None, None],
    "recordedBy": ["asha", "asha", "asha", "biju", "asha", None, "biju"],
    "eventDate":  ["2021-01-05T07:00", "2021-01-05T07:30", "2021-01-06T06:00",
                   "2021-01-06T09:00", "2021-01-06T16:00", "2021-01-06", None],
    "decimalLatitude":  [20.0, 20.0, 20.0, 20.0, 20.05, 20.0, 20.0],
    "decimalLongitude": [78.0, 78.0, 78.0, 78.0, 78.05, 78.0, 78.0],
    "cell_id":    ["c0001_0001"] * 7,
    "samplingEventIdentifier": ["S1", "S1", "S2", "S3", "S4", "S5", "S6"],
})
u_gbif = define_sampling_units(df, route="gbif")
u_ebird = define_sampling_units(df, route="ebird")
check("eventID used where present", u_gbif.iloc[0] == u_gbif.iloc[1] == "EV:E1")
check("pseudo-unit pools different observers at one point-day-cell",
      u_gbif.iloc[2] == u_gbif.iloc[3], f"{u_gbif.iloc[2]} vs {u_gbif.iloc[3]}")
check("same observer at another point that day is a separate unit",
      u_gbif.iloc[4] != u_gbif.iloc[2], f"{u_gbif.iloc[4]}")
check("record with no observer still gets the point-day unit",
      u_gbif.iloc[5] == u_gbif.iloc[2], f"{u_gbif.iloc[5]}")
check("record with no date gets no unit", pd.isna(u_gbif.iloc[6]))
check("eBird route keeps checklists distinct",
      u_ebird.nunique() == 6, f"{u_ebird.nunique()} distinct units")
try:
    define_sampling_units(df.drop(columns=["decimalLatitude", "decimalLongitude"]), route="gbif")
    raised = False
except KeyError:
    raised = True
check("fallback without coordinate columns raises, not silently empty", raised)

# -- Test 3: incidence counts units, not records -----------------------------
print("\n[3] A species seen repeatedly within one unit must count once")
rep = pd.DataFrame({
    "cell_id":   ["cA"] * 6,
    "species_id": [1, 1, 1, 2, 1, 2],
    "eventID":    ["E1", "E1", "E1", "E1", "E2", "E2"],
    "recordedBy": ["x"] * 6,
    "eventDate":  ["2021-01-01"] * 6,
})
inc, rep_report = build_incidence(rep, route="gbif")
y1 = int(inc.loc[inc.species_id == 1, "Y"].iloc[0])
y2 = int(inc.loc[inc.species_id == 2, "Y"].iloc[0])
check("species 1 on both units -> Y=2", y1 == 2, f"Y={y1}")
check("species 2 on both units -> Y=2", y2 == 2, f"Y={y2}")
check("T counts distinct units, not records", int(inc["T"].iloc[0]) == 2,
      f"T={int(inc['T'].iloc[0])}")

# -- Test 4: end-to-end against a known answer -------------------------------
print("\n[4] INTEGRATION -- records in, correct richness estimate out")
print("    A synthetic cell with a known species pool is expanded into records,")
print("    pushed through the real pipeline, and the result compared against the")
print("    estimate computed directly from the known incidence matrix.")
S_true, T = 180, 60
r = np.random.default_rng(303)
p = r.beta(0.5, 2.5, size=S_true)
det = r.random((T, S_true)) < p                       # the ground-truth matrix

rows = []
for t in range(T):
    for s in np.flatnonzero(det[t]):
        # each detection duplicated at random, as real data contains repeats
        for _ in range(r.integers(1, 3)):
            rows.append({"cell_id": "cZ", "species_id": int(s),
                         "eventID": f"E{t}", "recordedBy": "obs",
                         "eventDate": "2021-01-01"})
recs = pd.DataFrame(rows)

inc, report = build_incidence(recs, route="gbif")
Y_pipeline = inc["Y"].to_numpy(dtype=float)
T_pipeline = int(inc["T"].iloc[0])

Y_direct = det.sum(axis=0); Y_direct = Y_direct[Y_direct > 0]

check("pipeline recovers T exactly", T_pipeline == T, f"{T_pipeline} vs {T}")
check("pipeline recovers the incidence vector exactly",
      np.array_equal(np.sort(Y_pipeline), np.sort(Y_direct)),
      f"{len(Y_pipeline)} species vs {len(Y_direct)}")

a = coverage_standardised_richness(Y_pipeline, T_pipeline, 0.90)
b = coverage_standardised_richness(Y_direct, T, 0.90)
check("richness estimate identical through the pipeline",
      abs(a["richness"] - b["richness"]) < 1e-9,
      f"{a['richness']:.4f} vs {b['richness']:.4f}")
print(f"        records fed in: {len(recs):,}   ->  S_obs {a['S_obs']}, "
      f"T {a['T']}, coverage {a['coverage_full']:.3f}")
print(f"        duplicate detections collapsed: "
      f"{len(recs) - report['unique_species_unit_pairs']:,}")

print("\n" + "=" * 72)
print(f"  {sum(results)}/{len(results)} checks passed")
print("=" * 72)
sys.exit(0 if all(results) else 1)
