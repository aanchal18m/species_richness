"""Part 4, Gate 1: where is the data, and where is it not?

Reads data/interim/richness.parquet (one row per cell x N) and produces the
numbers the threshold decision needs. It does not choose the threshold.

Stages
  flags    -> data/interim/cell_flags.parquet (andaman_nicobar, bio12_saturated_frac)
  tables   -> outputs/tables/gate1_*.csv, outputs/tables/gate1_report.json
  guard    -> outputs/tables/gate1_guard_simulation.csv
  figures  -> outputs/figures/gate1_coverage_ecdf.png, gate1_split_map.png
              (--threshold and --min-units set the split drawn on the map)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_richness as br  # noqa: E402
from incidence import define_sampling_units  # noqa: E402
from richness import coverage_at, observed_coverage  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
FLAGS = INTERIM / "cell_flags.parquet"
REPORT = TABLES / "gate1_report.json"

THRESHOLDS = (0.80, 0.90, 0.95)
GUARD_UNITS = (1, 2, 3, 5, 10, 15, 20, 30, 50)
GUARD_RECORDS = (2, 5, 10, 20, 50, 100)
SIM_K = (2, 3, 5, 8, 10, 15, 20, 30, 50)

# Reference palette (dataviz skill, light mode). Three state colours at most on a map.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
WELL, POOR, NODATA = "#2a78d6", "#eb6834", "#d9d8d3"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


# --- inputs -----------------------------------------------------------------
def andaman_nicobar(lon, lat) -> np.ndarray:
    """No mainland cell lies east of 92E and south of 14.5N."""
    return (np.asarray(lon) > 92) & (np.asarray(lat) < 14.5)


def stage_flags():
    import geopandas as gpd
    g = gpd.read_file(INTERIM / "grid.gpkg", layer="grid", ignore_geometry=True)
    env = pd.read_parquet(INTERIM / "env.parquet", columns=["cell_id", "bio12_saturated_frac"])
    f = g[["cell_id", "lon", "lat"]].merge(env, on="cell_id", how="left", validate="one_to_one")
    f["andaman_nicobar"] = andaman_nicobar(f.lon, f.lat)
    f[["cell_id", "andaman_nicobar", "bio12_saturated_frac"]].to_parquet(FLAGS, index=False)
    print(f"   {FLAGS.name}: {int(f.andaman_nicobar.sum())} Andaman & Nicobar cells, "
          f"{int((f.bio12_saturated_frac > 0).sum())} with bio12 saturation")


def load():
    import geopandas as gpd
    r = pd.read_parquet(br.RICHNESS)
    g = gpd.read_file(INTERIM / "grid.gpkg", layer="grid")
    flags = pd.read_parquet(FLAGS)
    return r, g, flags


def well_sampled(r: pd.DataFrame, threshold: float, min_units: int) -> pd.Series:
    """The split rule: coverage at or above the threshold AND enough sampling
    units for the coverage estimate to mean anything (the guard)."""
    return (r["coverage_full"] >= threshold) & (r["T"] >= min_units)


# --- tables -----------------------------------------------------------------
def stage_tables():
    r, g, flags = load()
    n_grid = len(g)
    rep = {"grid_cells": n_grid}

    q = r.groupby("N")["coverage_full"].describe(percentiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9])
    q.insert(0, "cells_with_data", r.groupby("N").size())
    q.insert(1, "cells_without_data", n_grid - q["cells_with_data"])
    q.round(4).to_csv(TABLES / "gate1_coverage_distribution.csv")
    rep["coverage_distribution"] = q.round(4).reset_index().to_dict("records")

    rows = []
    for n, d in r.groupby("N"):
        for c in THRESHOLDS:
            for m in GUARD_UNITS:
                w = well_sampled(d, c, m)
                rows.append({"N": n, "threshold": c, "min_units": m, "well_sampled": int(w.sum()),
                             "q1_zero_among_well": int((w & (d["Q1"] == 0)).sum())})
    clear = pd.DataFrame(rows)
    clear.to_csv(TABLES / "gate1_threshold_counts.csv", index=False)
    rep["cells_clearing"] = clear.pivot_table(index=["N", "min_units"], columns="threshold",
                                              values="well_sampled").reset_index().to_dict("records")

    guard = []
    for n, d in r.groupby("N"):
        for m in GUARD_UNITS:
            guard.append({"N": n, "guard": f"T < {m}", "cells_below": int((d["T"] < m).sum()),
                          "of_which_coverage_1": int(((d["T"] < m) & (d["coverage_full"] >= 1)).sum())})
        for m in GUARD_RECORDS:
            guard.append({"N": n, "guard": f"records < {m}", "cells_below": int((d["n_records_N"] < m).sum()),
                          "of_which_coverage_1": int(((d["n_records_N"] < m) & (d["coverage_full"] >= 1)).sum())})
    guard = pd.DataFrame(guard)
    guard.to_csv(TABLES / "gate1_guard_counts.csv", index=False)
    rep["below_guard"] = guard.to_dict("records")
    rep["coverage_exactly_1"] = {int(n): {"cells": int((d.coverage_full >= 1).sum()),
                                          "median_T": float(d.loc[d.coverage_full >= 1, "T"].median())}
                                 for n, d in r.groupby("N")}

    # how far the split moves with N, and overlap with the flagged sets
    r = r.merge(flags, on="cell_id", how="left")
    move, overlap = [], []
    for c in THRESHOLDS:
        for m in (5, 10, 20):
            sets = {n: set(d.cell_id[well_sampled(d, c, m)]) for n, d in r.groupby("N")}
            base = sets[1]
            for n, s in sets.items():
                move.append({"threshold": c, "min_units": m, "N": n, "well_sampled": len(s),
                             "jaccard_vs_N1": round(len(s & base) / len(s | base), 4) if s | base else 1.0,
                             "lost_vs_N1": len(base - s), "gained_vs_N1": len(s - base)})
            for n, d in r.groupby("N"):
                w = well_sampled(d, c, m)
                overlap.append({"threshold": c, "min_units": m, "N": n, "well_sampled": int(w.sum()),
                                "andaman_nicobar": int((w & d.andaman_nicobar).sum()),
                                "bio12_saturated_any": int((w & (d.bio12_saturated_frac > 0)).sum()),
                                "bio12_saturated_gt_half": int((w & (d.bio12_saturated_frac > 0.5)).sum())})
    pd.DataFrame(move).to_csv(TABLES / "gate1_split_movement.csv", index=False)
    pd.DataFrame(overlap).to_csv(TABLES / "gate1_flag_overlap.csv", index=False)
    rep["split_movement"] = move
    rep["flag_overlap"] = overlap
    rep["flag_totals"] = {"andaman_nicobar": int(flags.andaman_nicobar.sum()),
                          "andaman_nicobar_with_data": int(r[r.N == 1].andaman_nicobar.sum()),
                          "bio12_saturated_any": int((flags.bio12_saturated_frac > 0).sum()),
                          "bio12_saturated_with_data": int((r[r.N == 1].bio12_saturated_frac > 0).sum())}
    prev = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    REPORT.write_text(json.dumps({**prev, **rep}, indent=2, default=float))
    print(q.round(3).to_string())


# --- guard simulation -------------------------------------------------------
def unit_incidence(d: pd.DataFrame) -> pd.DataFrame:
    """(unit, species) pairs for one cell's records, deduplicated."""
    return pd.DataFrame({"unit": define_sampling_units(d, route="gbif"),
                         "species_id": d["species_id"].to_numpy()}).dropna().drop_duplicates()


def subsample_coverage(pairs: pd.DataFrame, k: int, reps: int, rng) -> list[tuple[float, float]]:
    """(estimated coverage of a k-unit subsample, expected coverage at k computed
    exactly from the full sample) for `reps` random subsamples. Each unit's
    species list is built once, so a draw only concatenates k small arrays."""
    codes, units = pd.factorize(pairs["unit"])
    T = len(units)
    sp = pairs["species_id"].to_numpy()
    order = np.argsort(codes, kind="stable")
    per_unit = np.split(sp[order], np.cumsum(np.bincount(codes, minlength=T))[:-1])
    Y_full = pairs.groupby("species_id").size().to_numpy(float)
    truth = coverage_at(Y_full, T, k)
    out = []
    for _ in range(reps):
        pick = rng.choice(T, size=k, replace=False)
        Y = np.unique(np.concatenate([per_unit[i] for i in pick]), return_counts=True)[1].astype(float)
        out.append((observed_coverage(Y, k), truth))
    return out


def stage_guard(n_cells: int = 120, reps: int = 25, seed: int = 0):
    """For each N: take richly sampled cells (T >= 200, coverage >= 0.97), draw
    k of their units, and compare the coverage estimate with what k units really
    achieve. The guard should sit where a cell that truly falls short is rarely
    reported as passing."""
    r = pd.read_parquet(br.RICHNESS)
    rng = np.random.default_rng(seed)
    rows = []
    for n in br.N_VALUES:
        rich = r[(r.N == n) & (r["T"] >= 200) & (r.coverage_full >= 0.97)]
        pick = rich.sample(min(n_cells, len(rich)), random_state=seed)["cell_id"]
        by_part = pd.Series(br.part_of(pick.to_numpy()), index=pick.to_numpy())
        for p, cells in by_part.groupby(by_part):
            d = br.dedup(pd.read_parquet(br.OCC_DIR / f"part-{p:02d}.parquet",
                                         filters=[("cell_id", "in", list(cells.index))]))
            d, _ = br.min_species_filter(d, n)
            for cid, dc in d.groupby("cell_id"):
                pairs = unit_incidence(dc)
                for k in SIM_K:
                    for est, truth in subsample_coverage(pairs, k, reps, rng):
                        rows.append({"N": n, "cell_id": cid, "k": k, "estimate": est, "truth": truth})
        print(f"   N={n}: {len(pick)} cells simulated", flush=True)
    sim = pd.DataFrame(rows)
    summ = []
    for (n, k), d in sim.groupby(["N", "k"]):
        err = d.estimate - d.truth
        row = {"N": n, "k": k, "median_bias": round(float(err.median()), 4),
               "mae": round(float(err.abs().mean()), 4),
               "p90_abs_error": round(float(err.abs().quantile(0.9)), 4),
               "share_estimate_1": round(float((d.estimate >= 1).mean()), 4)}
        for c in THRESHOLDS:
            # Conditioning on truth < c makes this rate rise with k (what remains short
            # sits just below c), so it measures borderline noise. The gross rate --
            # passing while truly more than 0.05 short -- is what a guard must stop.
            short = d[d.truth < c]
            gross = d[d.truth < c - 0.05]
            row[f"false_pass_{c:.2f}"] = round(float((short.estimate >= c).mean()), 4) if len(short) else None
            row[f"gross_false_pass_{c:.2f}"] = round(float((gross.estimate >= c).mean()), 4) if len(gross) else None
            row[f"n_gross_short_{c:.2f}"] = int(len(gross))
        summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(TABLES / "gate1_guard_simulation.csv", index=False)
    with pd.option_context("display.width", 200):
        print(summ.to_string(index=False))


# --- figures ----------------------------------------------------------------
def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)


def stage_figures(threshold: float, min_units: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    r, g, flags = load()
    FIGURES.mkdir(parents=True, exist_ok=True)

    # ECDF of coverage, one line per N, direct-labelled
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for col, (n, d) in zip(SERIES, r.groupby("N")):
        x = np.sort(d.coverage_full.to_numpy())
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", color=col, linewidth=2, label=f"N = {n}")
    for c in THRESHOLDS:
        ax.axvline(c, color=INK_2, linewidth=0.8, linestyle=(0, (2, 3)))
        ax.text(c, 1.02, f"{c:.2f}", color=INK_2, fontsize=8, ha="center", transform=ax.get_xaxis_transform())
    ax.set_xlim(0, 1.0005)
    ax.set_xlabel("Sample coverage (observed, full sample)", color=INK_2, fontsize=9)
    ax.set_ylabel("Share of cells with records", color=INK_2, fontsize=9)
    ax.set_title("Coverage across cells with records, by minimum species per unit N",
                 color=INK, fontsize=11, loc="left", pad=20)
    # curves never cross, so legend order = vertical order: identity is not colour alone
    leg = ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left",
                    title="curves, top to bottom", title_fontsize=9)
    leg.get_title().set_color(INK_2)
    fig.tight_layout()
    fig.savefig(FIGURES / "gate1_coverage_ecdf.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)

    # split map, small multiples over N
    an = g.merge(flags, on="cell_id")
    an = an[an.andaman_nicobar]
    fig, axes = plt.subplots(2, 2, figsize=(10, 11), facecolor=SURFACE)
    for ax, (n, d) in zip(axes.ravel(), r.groupby("N")):
        _style(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        m = g.merge(d, on="cell_id", how="left")
        w = well_sampled(m.fillna({"coverage_full": -1, "T": 0}), threshold, min_units)
        has = m["T"].notna()
        m.plot(ax=ax, color=np.where(w, WELL, np.where(has, POOR, NODATA)), linewidth=0)
        an.plot(ax=ax, facecolor="none", edgecolor=INK, hatch="////", linewidth=0.3)
        ax.set_xlabel("")
        ax.set_ylabel("")       # geopandas labels projected axes; meaningless without ticks
        ax.set_title(f"N = {n}:  {int(w.sum()):,} well-sampled  ·  {int((has & ~w).sum()):,} poorly sampled",
                     color=INK, fontsize=10, loc="left")
    fig.legend(handles=[Patch(color=WELL, label=f"Well sampled (coverage ≥ {threshold:.2f}, ≥ {min_units} units)"),
                        Patch(color=POOR, label="Poorly sampled (records, fails threshold or guard)"),
                        Patch(color=NODATA, label="No records"),
                        Patch(facecolor="none", edgecolor=INK, hatch="////",
                              label="Andaman & Nicobar: excluded from training")],
               loc="lower center", ncol=2, frameon=False, fontsize=9, labelcolor=INK_2)
    fig.suptitle(f"Gate 1 split at a candidate threshold (coverage {threshold:.2f}, ≥ {min_units} units), "
                 f"for each N", color=INK, fontsize=12, x=0.02,
                 ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(FIGURES / "gate1_split_map.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"   figures -> {FIGURES.relative_to(ROOT)}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["flags", "tables", "guard", "figures"])
    p.add_argument("--threshold", type=float, default=0.90)
    p.add_argument("--min-units", type=int, default=10)
    a = p.parse_args(argv)
    {"flags": stage_flags, "tables": stage_tables, "guard": stage_guard,
     "figures": lambda: stage_figures(a.threshold, a.min_units)}[a.stage]()


if __name__ == "__main__":
    main()
