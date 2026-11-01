"""Part 9: report figures. Reads finished outputs only; computes nothing new.

  fig1_predicted_richness.png   predictions inside the AOA; outside = hatched, uncoloured
  fig2_aoa.png                  training / inside / outside, + share inside by elevation band
  fig3_observed_vs_predicted.png  spatially blocked OOF (C1)
  fig4_hotspots.png             true vs predicted top quintile on training cells (C2)
  fig5_stress.png               rho and C2 recall against training fraction

Palette: dataviz reference palette, light mode. Maps carry at most three
categorical colours plus neutral grey.

Run: python src/figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_models as fm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL, NEUTRAL_LIGHT = "#d9d8d3", "#efeeea"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _map_ax(ax):
    ax.set_facecolor(SURFACE)
    ax.set_axis_off()
    ax.set_aspect("equal")


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=0)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def load():
    import geopandas as gpd
    g = gpd.read_file(ROOT / "data" / "interim" / "grid.gpkg", layer="grid")[["cell_id", "geometry"]]
    p = pd.read_parquet(PROCESSED / "predictions.parquet")
    return g.merge(p, on="cell_id", validate="one_to_one")


def fig1(gdf, path):
    """Quantile classes: predictions are concentrated in ~90-120 species, so a
    continuous ramp over the full range reads as nearly uniform."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch
    plt.rcParams["hatch.linewidth"] = 0.5
    inside, outside = gdf[gdf.aoa], gdf[~gdf.aoa]
    v = inside.pred_richness_c090
    breaks = np.unique(np.round(np.quantile(v, np.linspace(0, 1, len(SEQ_BLUE) + 1))))
    breaks[0], breaks[-1] = np.floor(v.min()), np.ceil(v.max())
    cmap = ListedColormap(SEQ_BLUE[:len(breaks) - 1])
    norm = BoundaryNorm(breaks, cmap.N)
    fig, ax = plt.subplots(figsize=(8.5, 9.5), facecolor=SURFACE)
    _map_ax(ax)
    inside.plot(ax=ax, column="pred_richness_c090", cmap=cmap, norm=norm, linewidth=0)
    outside.plot(ax=ax, facecolor=NEUTRAL_LIGHT, edgecolor="#a9a8a2", hatch="////", linewidth=0)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.035, pad=0.02, aspect=40,
                      ticks=breaks, spacing="uniform")
    cb.ax.set_xticklabels([f"{int(b)}" for b in breaks])
    cb.set_label("Predicted winter bird richness at 0.90 sample coverage (species); "
                 "classes hold equal numbers of cells", color=INK_2, fontsize=9)
    cb.ax.tick_params(colors=INK_2, labelsize=8.5)
    cb.outline.set_visible(False)
    ax.legend(handles=[Patch(facecolor=NEUTRAL_LIGHT, edgecolor="#a9a8a2", hatch="////",
                             label=f"Outside area of applicability: not predicted ({len(outside):,} cells)")],
              loc="lower left", frameon=False, fontsize=9, labelcolor=INK_2)
    ax.set_title(f"Predicted species richness, {len(inside):,} of {len(gdf):,} cells inside the area of applicability",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig2(gdf, path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    cat = np.where(gdf.train_eligible, "train", np.where(gdf.aoa, "inside", "outside"))
    colours = {"train": BLUE, "inside": AQUA, "outside": ORANGE}
    fig = plt.figure(figsize=(12, 8.6), facecolor=SURFACE)
    ax = fig.add_axes([0.0, 0.04, 0.62, 0.9])
    _map_ax(ax)
    gdf.plot(ax=ax, color=[colours[c] for c in cat], linewidth=0)
    n = pd.Series(cat).value_counts()
    ax.legend(handles=[Patch(color=BLUE, label=f"Training cell ({n['train']:,})"),
                       Patch(color=AQUA, label=f"Predicted, inside AOA ({n['inside']:,})"),
                       Patch(color=ORANGE, label=f"Outside AOA ({n['outside']:,})")],
              loc="lower left", frameon=False, fontsize=9, labelcolor=INK_2)
    ax.set_title("Area of applicability", color=INK, fontsize=11, loc="left")
    b = pd.read_csv(TABLES / "aoa_breakdown.csv")
    e = b[b.breakdown.str.startswith("elev_band")].copy()
    e["lo"] = e.group.str.extract(r"\[(-?\d+)").astype(float)
    e = e.sort_values("lo")
    e["label"] = [f"{max(int(l), 0):,}–{int(h):,} m" if h < 9000 else f"≥ {int(l):,} m"
                  for l, h in zip(e.lo, e.group.str.extract(r", (\d+)\)")[0].astype(float))]
    ax2 = fig.add_axes([0.70, 0.18, 0.27, 0.6])
    _style(ax2)
    ax2.grid(axis="y", visible=False)
    y = np.arange(len(e))
    ax2.barh(y, e.share_inside, color=AQUA, height=0.6)
    for yy, s, c in zip(y, e.share_inside, e.cells):
        ax2.text(s + 0.02, yy, f"{s:.1%}  (n={c:,})", va="center", fontsize=8, color=INK_2)
    ax2.set_yticks(y, e.label, fontsize=8.5, color=INK_2)
    ax2.set_xlim(0, 1.5)
    ax2.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax2.set_title("Poorly sampled cells inside the AOA,\nby mean elevation", color=INK, fontsize=9.5, loc="left")
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig3(path):
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    o = pd.read_parquet(PROCESSED / "oof_predictions.parquet")
    y, p = o.observed.to_numpy(), o.spatial_residual_rf.to_numpy()
    rho = spearmanr(y, p)[0]
    rmse = fm.rmse(y, p)
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    fig, ax = plt.subplots(figsize=(6.4, 6.2), facecolor=SURFACE)
    _style(ax)
    lim = [0, max(y.max(), p.max()) * 1.03]
    ax.plot(lim, lim, color=INK_2, linewidth=1, linestyle=(0, (2, 3)), label="1:1")
    ax.scatter(p, y, s=7, color=BLUE, alpha=0.35, linewidth=0)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Predicted richness, out-of-fold (species)", color=INK_2, fontsize=9)
    ax.set_ylabel("Observed richness at 0.90 coverage (species)", color=INK_2, fontsize=9)
    ax.text(0.03, 0.97, f"Spearman ρ = {rho:.3f}   (C1 bar 0.60)\nRMSE = {rmse:.1f} species   R² = {r2:.2f}\n"
                        f"n = {len(y):,} cells, 100 km blocked 5-fold CV",
            transform=ax.transAxes, va="top", fontsize=9, color=INK)
    ax.set_title("Observed vs predicted, spatially blocked cross-validation", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig4(gdf, path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    o = pd.read_parquet(PROCESSED / "oof_predictions.parquet")
    k = int(round(0.2 * len(o)))
    true_top = set(o.cell_id.iloc[np.argsort(-o.observed.to_numpy(), kind="stable")[:k]])
    pred_top = set(o.cell_id.iloc[np.argsort(-o.spatial_residual_rf.to_numpy(), kind="stable")[:k]])
    hit, miss, false = true_top & pred_top, true_top - pred_top, pred_top - true_top
    colours = {"hit": BLUE, "miss": ORANGE, "false": AQUA}
    fig, ax = plt.subplots(figsize=(8.5, 9.5), facecolor=SURFACE)
    _map_ax(ax)
    gdf.plot(ax=ax, color=NEUTRAL_LIGHT, linewidth=0)
    gdf[gdf.train_eligible].plot(ax=ax, color="#b9b8b2", linewidth=0)   # darker than background grey
    for name, cells in (("hit", hit), ("miss", miss), ("false", false)):
        gdf[gdf.cell_id.isin(cells)].plot(ax=ax, color=colours[name], linewidth=0)
    ax.legend(handles=[Patch(color=BLUE, label=f"Hit: true hotspot, predicted ({len(hit)})"),
                       Patch(color=ORANGE, label=f"Miss: true hotspot, not predicted ({len(miss)})"),
                       Patch(color=AQUA, label=f"False alarm: predicted, not a true hotspot ({len(false)})"),
                       Patch(color="#b9b8b2", label="Other training cells"),
                       Patch(color=NEUTRAL_LIGHT, label="Not a training cell")],
              loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK_2)
    ax.set_title(f"Top-quintile hotspots (k = {k} of {len(o):,} training cells): recall = {len(hit) / k:.3f} "
                 f"(C2 bar 0.70)", color=INK, fontsize=10.5, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return len(hit), len(miss), len(false), k


def fig5(path):
    import matplotlib.pyplot as plt
    s = pd.read_csv(TABLES / "stress_downsample_summary.csv").sort_values("level")
    x = s.level.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), facecolor=SURFACE)
    for ax, (m, lo, hi), bar, title in zip(
            axes, (("rho_mean", "rho_min", "rho_max"), ("c2_mean", "c2_min", "c2_max")), (0.60, 0.70),
            ("Spearman ρ (C1)", "Top-quintile hotspot recall (C2)")):
        _style(ax)
        ax.fill_between(x, s[lo], s[hi], color=BLUE, alpha=0.18, linewidth=0, label="range over 5 replicates")
        ax.plot(x, s[m], color=BLUE, linewidth=2, marker="o", markersize=7, label="mean")
        ax.axhline(bar, color=ORANGE, linewidth=1.2, linestyle=(0, (4, 3)))
        ax.text(11, bar + 0.006, f"pre-registered bar {bar:.2f}", color=ORANGE, fontsize=8.5)
        for xx, v in zip(x, s[m]):
            ax.annotate(f"{v:.3f}", (xx, v), xytext=(0, 9), textcoords="offset points", ha="center",
                        fontsize=8, color=INK_2)
        ax.set_xticks(x, [f"{int(v)}%" for v in x])
        ax.set_xlabel("Share of training cells kept (held-out folds unchanged)", color=INK_2, fontsize=8.5)
        ax.set_ylim(0.35, 0.75)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main(which=None):
    import matplotlib
    matplotlib.use("Agg")
    FIGURES.mkdir(parents=True, exist_ok=True)
    gdf = load()
    jobs = {"1": lambda: fig1(gdf, FIGURES / "fig1_predicted_richness.png"),
            "2": lambda: fig2(gdf, FIGURES / "fig2_aoa.png"),
            "3": lambda: fig3(FIGURES / "fig3_observed_vs_predicted.png"),
            "4": lambda: fig4(gdf, FIGURES / "fig4_hotspots.png"),
            "5": lambda: fig5(FIGURES / "fig5_stress.png")}
    for k in (which or jobs):
        out = jobs[k]()
        print(f"   fig{k} done" + (f": {out}" if out else ""))


if __name__ == "__main__":
    main(sys.argv[1:] or None)
