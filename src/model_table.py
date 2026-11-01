"""Part 5, step 1-2: assemble the model table and screen predictors by VIF.

One row per grid cell -- training cells and prediction targets alike. Nothing
is dropped: `train_eligible` marks the cells a model may learn from.

  train_eligible = coverage_full >= 0.90 AND T >= 20 point-day units
                   AND NOT Andaman & Nicobar           (Gate 1 decision, N = 3)
  response       = richness_c090 (richness rarefied to 0.90 coverage)

VIF screening runs on the training-eligible cells, the data the models see.
Land-cover fractions sum to 1, so before screening one class is dropped as the
reference category: CROPLAND, fixed by decision rather than by the data. It is
India's matrix land use (median share 0.485 over all cells), so the other
classes read as "relative to cropland". Two data-driven rules were tried and
rejected: dropping the least-variance class (snow) left the rest summing to
almost exactly 1, so the screen then removed tree cover at VIF 6562; dropping
the class with the largest mean share in the training pool also removed tree
cover, because the pool is concentrated in the forested south-west.
Then the predictor with the largest VIF is dropped, one at a time, until every
VIF <= 10, with two protections (CLAUDE.md decisions):
  * elev_mean is never dropped: when it has the largest VIF the next-largest
    goes instead. Elevation is central to the ecological reading and would
    otherwise be represented only by elev_sd. If elev_mean is the only
    predictor above 10, the screen stops and its VIF is reported.
  * bio16 is never dropped while bio12 remains: bio12 is clipped at 6553.5 mm
    in the Western Ghats, so bio12 goes instead.

Writes
  data/processed/model_table.parquet
  data/processed/predictors.json   (selected predictors, screening log, counts)

Run: python src/model_table.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
TABLE = PROCESSED / "model_table.parquet"
PREDICTORS = PROCESSED / "predictors.json"
FROZEN = PROCESSED / "FROZEN.json"

N_CHOSEN = 3
COVERAGE_MIN = 0.90
UNITS_MIN = 20
RESPONSE = "richness_c090"
VIF_MAX = 10.0
PROTECTED = ("elev_mean",)

BIOCLIM = [f"bio{i:02d}" for i in range(1, 20)]
TERRAIN = ["elev_mean", "elev_sd", "elev_range", "slope_mean"]
LANDCOVER = ["lc_tree", "lc_shrub", "lc_grass", "lc_crop", "lc_built", "lc_bare", "lc_snow",
             "lc_water", "lc_wetland", "lc_mangrove", "lc_moss"]
CANDIDATES = BIOCLIM + TERRAIN + LANDCOVER + ["lc_shannon"]
LANDCOVER_REFERENCE = "lc_crop"


def train_eligible(df: pd.DataFrame) -> pd.Series:
    return ((df["coverage_full"] >= COVERAGE_MIN) & (df["T"] >= UNITS_MIN)
            & ~df["andaman_nicobar"].fillna(False).astype(bool)).fillna(False)


def vif(X: pd.DataFrame) -> pd.Series:
    """Variance inflation factor by definition: regress each column on all the
    others (least squares, with intercept) and return 1 / (1 - R^2). Exactly
    collinear columns get inf."""
    Z = X.to_numpy(float)
    Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
    out = {}
    for j, name in enumerate(X.columns):
        y, A = Z[:, j], np.delete(Z, j, axis=1)
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1 - np.sum((y - A @ beta) ** 2) / np.sum(y ** 2)
        out[name] = np.inf if r2 >= 1 - 1e-10 else 1 / (1 - r2)
    return pd.Series(out)


def vif_screen(X: pd.DataFrame, threshold: float = VIF_MAX,
               protected=PROTECTED) -> tuple[list[str], list[dict]]:
    kept = list(X.columns)
    log = []
    while True:
        v = vif(X[kept]).sort_values(ascending=False)
        largest = v.index[0]
        candidates = [c for c in v.index if c not in protected]
        if not candidates or v[candidates[0]] <= threshold:
            return kept, log
        drop, reason = candidates[0], "largest VIF"
        if drop != largest:
            reason = f"{largest} (protected) had the largest VIF; next-largest dropped"
        if drop == "bio16" and "bio12" in kept:
            drop, reason = "bio12", "bio16 had the largest VIF; bio12 dropped instead (clipped at 6553.5 mm)"
        log.append({"dropped": drop, "vif": float(v[drop]), "largest": largest,
                    "largest_vif": float(v[largest]), "reason": reason})
        kept.remove(drop)


def assemble(n: int = N_CHOSEN) -> pd.DataFrame:
    import geopandas as gpd

    g = gpd.read_file(INTERIM / "grid.gpkg", layer="grid")
    cent = g.geometry.centroid
    grid = pd.DataFrame({"cell_id": g.cell_id, "lon": g.lon, "lat": g.lat, "x": cent.x, "y": cent.y,
                         "land_fraction": g.land_fraction})
    env = pd.read_parquet(INTERIM / "env.parquet")
    flags = pd.read_parquet(INTERIM / "cell_flags.parquet")
    r = pd.read_parquet(INTERIM / "richness.parquet")
    r = r[r.N == n][["cell_id", "T", "S_obs", "coverage_full", "chao2", RESPONSE, "n_records_N"]]
    t = (grid.merge(env, on="cell_id", how="left", validate="one_to_one")
             .merge(flags[["cell_id", "andaman_nicobar"]], on="cell_id", how="left", validate="one_to_one")
             .merge(r, on="cell_id", how="left", validate="one_to_one"))
    t["has_records"] = t["T"].notna()
    t["train_eligible"] = train_eligible(t)
    if t.loc[t.train_eligible, RESPONSE].isna().any():
        raise AssertionError("a training-eligible cell has no response value")
    return t


def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(note: str, frozen: Path = FROZEN, files=(TABLE, PREDICTORS)) -> dict:
    """Gate 2: record the hashes of the model table and predictor list."""
    from datetime import datetime
    rec = {"frozen": datetime.now().isoformat(timespec="seconds"), "note": note,
           "files": {Path(f).name: _sha256(f) for f in files}}
    Path(frozen).write_text(json.dumps(rec, indent=2))
    return rec


def verify_frozen(frozen: Path = FROZEN, directory: Path = PROCESSED) -> None:
    """Raise if a frozen file has changed. Every downstream script calls this."""
    rec = json.loads(Path(frozen).read_text())
    for name, digest in rec["files"].items():
        if _sha256(Path(directory) / name) != digest:
            raise RuntimeError(f"{name} differs from its Gate 2 hash; the model table is frozen")


def main():
    if FROZEN.exists():
        sys.exit(f"Model table is frozen (Gate 2, {FROZEN.name}); refusing to rebuild it.")
    t = assemble()
    train = t[t.train_eligible]
    no_var = [c for c in CANDIDATES if train[c].std() == 0]          # undefined in VIF, uninformative
    dropped_class = LANDCOVER_REFERENCE
    pool = [c for c in CANDIDATES if c != dropped_class and c not in no_var]
    kept, log = vif_screen(train[pool])
    log = [{"dropped": c, "reason": "no variance among training cells"} for c in no_var] + \
          [{"dropped": dropped_class, "reason": "land-cover closure: reference class (cropland, fixed)"}] + log
    PROCESSED.mkdir(parents=True, exist_ok=True)
    t.to_parquet(TABLE, index=False)
    r_12_16 = float(train[["bio12", "bio16"]].corr().iloc[0, 1])
    info = {
        "response": RESPONSE, "N": N_CHOSEN, "coverage_min": COVERAGE_MIN, "units_min": UNITS_MIN,
        "counts": {"cells": int(len(t)), "with_records": int(t.has_records.sum()),
                   "train_eligible": int(t.train_eligible.sum()),
                   "andaman_nicobar_excluded": int((t.andaman_nicobar & (t.coverage_full >= COVERAGE_MIN)
                                                    & (t["T"] >= UNITS_MIN)).sum()),
                   "prediction_targets": int((~t.train_eligible).sum())},
        "closure_dropped": dropped_class, "vif_max": VIF_MAX, "candidates": CANDIDATES,
        "selected": kept, "screening_log": log,
        "final_vif": {k: round(float(v), 3) for k, v in vif(train[kept]).items()},
        "corr_bio12_bio16_train": round(r_12_16, 4),
    }
    PREDICTORS.write_text(json.dumps(info, indent=2))
    print(json.dumps({k: info[k] for k in ("counts", "closure_dropped", "selected", "final_vif",
                                           "corr_bio12_bio16_train")}, indent=2))
    for row in log:
        v = f"VIF {row['vif']:>10.1f}" if "vif" in row else " " * 14
        print(f"   drop {row['dropped']:12s} {v}   ({row['reason']})")


if __name__ == "__main__":
    main()
