"""Part 10: split-half reliability of the response (design pre-specified in CLAUDE.md).

How much of the C1/C2 shortfall could be noise in the richness estimate itself?
Each training cell's point-day units are split at random into halves; each half
is rarefied to coverage 0.80; the agreement between halves across cells is the
reliability at half depth, projected to full depth by Spearman-Brown.

  C1 ceiling: sqrt(r_full) -- a perfect model of true richness correlates with a
              noisy measurement at most this well.
  C2 ceiling: top-quintile recall of a perfect model, simulated under a bivariate
              normal with correlation sqrt(r_full). Raw half-half agreement is
              reported too; it understates the ceiling (noise on both sides).

Caveat: halves share observers and sites, so the reliability is optimistic and
the ceilings are optimistic.

Writes outputs/tables/reliability_splits.csv, reliability_summary.json

Run: python src/reliability.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_richness as br  # noqa: E402
import fit_models as fm  # noqa: E402
from incidence import define_sampling_units  # noqa: E402
from richness import coverage_standardised_richness  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
N_CHOSEN = 3
SPLITS = 5
SEED = 20261006
HALF_TARGET = 0.80
SIM_DRAWS = 2000


def spearman_brown(r: float) -> float:
    return 2 * r / (1 + r)


def c1_ceiling(r_full: float) -> float:
    return float(np.sqrt(max(r_full, 0.0)))


def simulated_recall(rho: float, n: int, draws: int = SIM_DRAWS, seed: int = 0, top: float = 0.2) -> float:
    """Mean top-quintile recall between two variables with correlation rho,
    bivariate normal, n cells."""
    rng = np.random.default_rng(seed)
    cov = [[1, rho], [rho, 1]]
    return float(np.mean([fm.hotspot_recall(*rng.multivariate_normal([0, 0], cov, n).T, top=top)
                          for _ in range(draws)]))


def split_units(units: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """Random halves: first floor(T/2) of a permutation -> A, the rest -> B."""
    perm = rng.permutation(units)
    h = len(perm) // 2
    return perm[:h], perm[h:]


def half_richness(pairs: pd.DataFrame, units: np.ndarray, target: float = HALF_TARGET) -> float:
    sub = pairs[pairs["unit"].isin(set(units))]
    Y = sub.groupby("species_id").size().to_numpy(float)
    r = coverage_standardised_richness(Y, len(units), target_coverage=target)
    return r["richness"] if r["attainable"] else np.nan


def cell_pairs(cells: set) -> dict[str, pd.DataFrame]:
    """(unit, species) pairs per training cell at N = 3, as in Part 4."""
    out = {}
    for path in sorted(br.OCC_DIR.glob("part-*.parquet")):
        d = pd.read_parquet(path)
        d = d[d.cell_id.isin(cells)]
        if d.empty:
            continue
        d = br.dedup(d)
        d, _ = br.min_species_filter(d, N_CHOSEN)
        u = define_sampling_units(d, route="gbif")
        pairs = pd.DataFrame({"cell_id": d.cell_id.to_numpy(), "unit": u.to_numpy(),
                              "species_id": d.species_id.to_numpy()}).dropna().drop_duplicates()
        for cid, g in pairs.groupby("cell_id"):
            out[cid] = g[["unit", "species_id"]]
    return out


def _one_split(args):
    split, cell_ids, pairs = args
    rng = np.random.default_rng([SEED, split])
    rows = []
    for cid in cell_ids:
        p = pairs[cid]
        a, b = split_units(np.sort(p["unit"].unique()), rng)
        rows.append({"split": split, "cell_id": cid, "T_a": len(a), "T_b": len(b),
                     "rich_a": half_richness(p, a), "rich_b": half_richness(p, b)})
    return rows


def main():
    table = pd.read_parquet(PROCESSED / "model_table.parquet")
    train = table[table.train_eligible].reset_index(drop=True)
    cell_ids = sorted(train.cell_id)
    pairs = cell_pairs(set(cell_ids))
    missing = set(cell_ids) - set(pairs)
    if missing:
        raise AssertionError(f"{len(missing)} training cells have no units")
    T_check = pd.Series({c: pairs[c]["unit"].nunique() for c in cell_ids})
    if not (T_check.reindex(train.cell_id).to_numpy() == train["T"].to_numpy()).all():
        raise AssertionError("rebuilt unit counts differ from the model table's T")
    print(f"   units rebuilt for {len(cell_ids):,} cells (T matches the model table)", flush=True)

    # sequential: a process pool would copy the whole pairs dict into every worker
    rows = []
    for s in range(SPLITS):
        rows += _one_split((s, cell_ids, pairs))
        print(f"   split {s} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "reliability_splits.csv", index=False)

    summ = []
    for s, d in df.groupby("split"):
        kept = d.dropna(subset=["rich_a", "rich_b"])
        r_half = float(spearmanr(kept.rich_a, kept.rich_b)[0])
        r_full = spearman_brown(r_half)
        n = len(kept)
        summ.append({"split": s, "kept_cells": n, "r_half": r_half, "r_full": r_full,
                     "c1_ceiling": c1_ceiling(r_full),
                     "c2_half_agreement": fm.hotspot_recall(kept.rich_a, kept.rich_b),
                     "c2_sim_at_r_half": simulated_recall(r_half, n),
                     "c2_ceiling_sim": simulated_recall(c1_ceiling(r_full), n)})
    summ = pd.DataFrame(summ)
    agg = {c: {"mean": float(summ[c].mean()), "min": float(summ[c].min()), "max": float(summ[c].max())}
           for c in summ.columns if c != "split"}

    kept_any = df.dropna(subset=["rich_a", "rich_b"]).cell_id.unique()
    tr = train.set_index("cell_id")
    def describe(ids):
        x = tr.loc[ids]
        return {"cells": int(len(x)), "T_median": float(x["T"].median()),
                "richness_c090_median": float(x.richness_c090.median()),
                "coverage_median": float(x.coverage_full.median()),
                "share_south_of_15N": float((x.lat < 15).mean())}
    kept_all_splits = df.dropna(subset=["rich_a", "rich_b"]).groupby("cell_id").size()
    kept_all_splits = kept_all_splits[kept_all_splits == SPLITS].index
    report = {"per_split": summ.to_dict("records"), "summary": agg,
              "pool_full": describe(train.cell_id), "pool_kept_in_all_splits": describe(kept_all_splits),
              "pool_dropped_in_any_split": describe(sorted(set(train.cell_id) - set(kept_all_splits))),
              "cells_kept_in_any_split": int(len(kept_any))}
    (TABLES / "reliability_summary.json").write_text(json.dumps(report, indent=2))
    with pd.option_context("display.width", 200):
        print(summ.round(4).to_string(index=False))
    print(json.dumps({k: report[k] for k in ("summary", "pool_full", "pool_kept_in_all_splits",
                                              "pool_dropped_in_any_split")}, indent=2))


if __name__ == "__main__":
    main()
