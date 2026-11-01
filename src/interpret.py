"""Part 8: SHAP interpretation of the final RF, and the effort-confound check.

  * Exact TreeSHAP (tree_path_dependent) on the 3,236 training cells only --
    where the model is supported. Additivity is asserted: SHAP values plus the
    expected value must reproduce every prediction.
  * SHAP importance (mean |SHAP|) beside Part 7's permutation importance, each
    as a share of its own total.
  * Dependence curves: binned medians of SHAP against predictor deciles, for
    the top 6 predictors and for elev_mean regardless of rank.
  * Effort check: Spearman(observed richness, T), Spearman(OUT-OF-FOLD residual,
    T) and Spearman(lc_built, T) among training cells, with 95% intervals from
    a spatial block bootstrap over the 100 km CV blocks. In-sample residuals of
    an RF grown to single-sample leaves are near zero, so OOF residuals are used.

Writes data/processed/shap_train.parquet; outputs/tables/importance_comparison.csv,
dependence_curves.csv, effort_check.json; outputs/figures/shap_*.png, effort_check.png

Run: python src/interpret.py
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_table import FROZEN, verify_frozen  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
MODEL = ROOT / "models" / "rf_final.joblib"
TOP_N = 6
ALWAYS = ("elev_mean",)
N_BINS = 10
WORKERS = 8
BOOT = 1000

# reference palette (dataviz skill, light mode)
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

UNITS = {"bio02": "°C", "bio03": "%", "bio04": "°C", "bio08": "°C", "bio09": "°C", "bio13": "mm/month",
         "bio14": "mm/month", "bio15": "% CV", "bio18": "mm/quarter", "bio19": "mm/quarter",
         "elev_mean": "m", "elev_sd": "m", "lc_shannon": "nats"}
LABELS = {"bio02": "Mean diurnal range", "bio03": "Isothermality", "bio08": "Mean temp, wettest quarter",
          "bio09": "Mean temp, driest quarter", "bio13": "Precip, wettest month", "bio14": "Precip, driest month",
          "bio15": "Precip seasonality", "bio18": "Precip, warmest quarter", "bio19": "Precip, coldest quarter",
          "elev_mean": "Mean elevation", "elev_sd": "Elevation SD", "lc_built": "Built-up fraction",
          "lc_water": "Open water fraction", "lc_tree": "Tree cover fraction", "lc_shrub": "Shrubland fraction",
          "lc_grass": "Grassland fraction", "lc_bare": "Bare/sparse fraction", "lc_wetland": "Wetland fraction",
          "lc_mangrove": "Mangrove fraction", "lc_snow": "Snow/ice fraction", "lc_moss": "Moss/lichen fraction",
          "lc_shannon": "Land-cover diversity"}


def label(p: str) -> str:
    u = UNITS.get(p, "fraction of cell")
    return f"{LABELS.get(p, p)} ({u})"


# --- SHAP -------------------------------------------------------------------
_EXPLAINER = None


def _init_worker(path):
    global _EXPLAINER
    import joblib
    import shap
    _EXPLAINER = shap.TreeExplainer(joblib.load(path)["model"])


def _explain(X):
    return _EXPLAINER.shap_values(X)


def shap_values(path: Path, X: np.ndarray, workers: int = WORKERS):
    import joblib
    import shap
    chunks = np.array_split(X, workers * 4)
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(str(path),)) as ex:
        sv = np.vstack(list(ex.map(_explain, chunks)))
    model = joblib.load(path)["model"]
    expected = float(np.ravel(shap.TreeExplainer(model).expected_value)[0])
    return sv, expected, model


def check_additivity(sv, expected, predictions, tol=1e-6) -> float:
    err = float(np.max(np.abs(sv.sum(axis=1) + expected - predictions)))
    if err > tol:
        raise AssertionError(f"SHAP values do not add up to the predictions (max error {err:.2e})")
    return err


def importance_comparison(shap_vals: pd.DataFrame, permutation: pd.Series) -> pd.DataFrame:
    """Both importances as shares of their own totals, with ranks."""
    s = shap_vals.abs().mean()
    p = permutation.clip(lower=0).reindex(s.index).fillna(0)
    out = pd.DataFrame({"shap_mean_abs": s, "shap_share": s / s.sum(),
                        "perm_rmse_increase": p, "perm_share": p / p.sum()})
    out["shap_rank"] = out["shap_share"].rank(ascending=False, method="min").astype(int)
    out["perm_rank"] = out["perm_share"].rank(ascending=False, method="min").astype(int)
    out["rank_shift"] = out["perm_rank"] - out["shap_rank"]
    return out.sort_values("shap_rank")


def binned_curve(x, s, n_bins: int = N_BINS) -> pd.DataFrame:
    """Median SHAP per quantile bin of the predictor (bins with tied edges merge)."""
    x, s = np.asarray(x, float), np.asarray(s, float)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    rows = [{"bin": b, "x_low": edges[b], "x_high": edges[b + 1], "x_median": float(np.median(x[idx == b])),
             "shap_median": float(np.median(s[idx == b])), "n": int((idx == b).sum())}
            for b in np.unique(idx)]
    return pd.DataFrame(rows)


PLOT_QUANTILES = np.r_[0, 0.01, np.arange(0.05, 0.951, 0.05), 0.99, 1]


def plot_curve(x, s) -> pd.DataFrame:
    """For DRAWING only: medians in 5% bins with separate 1% tails, so the line
    reaches the sparse ends of the range (decile bins hid the decline of the
    elevation effect above ~1,500 m). The analysis uses binned_curve."""
    x, s = np.asarray(x, float), np.asarray(s, float)
    edges = np.unique(np.quantile(x, PLOT_QUANTILES))
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    return pd.DataFrame([{"x_median": float(np.median(x[idx == b])), "shap_median": float(np.median(s[idx == b]))}
                         for b in np.unique(idx)])


def interior_peak(curve: pd.DataFrame, min_drop: float) -> dict:
    """Does the binned curve peak inside the range and fall by at least
    `min_drop` on BOTH sides? (A hump, not a monotone trend.)"""
    y = curve["shap_median"].to_numpy()
    k = int(np.argmax(y))
    left, right = y[k] - y[:k + 1].min(), y[k] - y[k:].min()
    return {"peak_bin": k, "peak_x": float(curve["x_median"].iloc[k]),
            "rise_from_low_end": float(left), "fall_to_high_end": float(right),
            "interior_peak": bool(0 < k < len(y) - 1 and left >= min_drop and right >= min_drop)}


# --- effort check -----------------------------------------------------------
def block_bootstrap_spearman(a, b, blocks, n_boot: int = BOOT, seed: int = 0) -> tuple[float, float, float]:
    """Spearman rho and a 95% interval from resampling whole spatial blocks."""
    a, b, blocks = np.asarray(a, float), np.asarray(b, float), np.asarray(blocks)
    rho = float(spearmanr(a, b)[0])
    uniq = np.unique(blocks)
    members = {u: np.flatnonzero(blocks == u) for u in uniq}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([members[u] for u in rng.choice(uniq, size=len(uniq), replace=True)])
        boots.append(spearmanr(a[idx], b[idx])[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return rho, float(lo), float(hi)


# --- figures ----------------------------------------------------------------
def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_importance(cmp: pd.DataFrame, path: Path):
    import matplotlib.pyplot as plt
    d = cmp.sort_values("shap_share")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7.5, 7.2), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    for yy, a, b in zip(y, d.shap_share, d.perm_share):
        ax.plot([a, b], [yy, yy], color=GRID, linewidth=2, zorder=1)
    # circle drawn larger than the diamond, so coinciding values stay visible as both
    ax.scatter(d.shap_share, y, s=95, color=BLUE, zorder=3, label="SHAP (mean |SHAP|)", edgecolor=SURFACE, linewidth=1.5)
    ax.scatter(d.perm_share, y, s=28, color=ORANGE, zorder=4, label="Permutation (RMSE increase)",
               edgecolor=SURFACE, linewidth=1, marker="D")
    ax.set_yticks(y, [LABELS.get(p, p) for p in d.index], color=INK_2, fontsize=8.5)
    ax.set_xlabel("Share of total importance", color=INK_2, fontsize=9)
    ax.set_title("Predictor importance: SHAP vs permutation (training cells)", color=INK, fontsize=11,
                 loc="left", pad=12)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_dependence(X: pd.DataFrame, S: pd.DataFrame, preds: list[str], curves: dict, path: Path):
    import matplotlib.pyplot as plt
    n = len(preds)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.3 * rows), facecolor=SURFACE)
    ylim = np.quantile(S[preds].to_numpy(), [0.002, 0.998])
    for ax, p in zip(axes.ravel(), preds):
        _style(ax)
        ax.scatter(X[p], S[p], s=5, color=BLUE, alpha=0.25, linewidth=0)
        c = plot_curve(X[p], S[p])
        ax.plot(c.x_median, c.shap_median, color=INK, linewidth=2)
        ax.axhline(0, color=INK_2, linewidth=0.8, linestyle=(0, (2, 3)))
        ax.set_ylim(*ylim)
        ax.set_xlabel(label(p), color=INK_2, fontsize=8.5)
        ax.set_title(LABELS.get(p, p), color=INK, fontsize=10, loc="left")
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("SHAP (species)", color=INK_2, fontsize=8.5)
    fig.suptitle("SHAP dependence: each dot a training cell; line = median in 5% bins (1% at the ends)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_effort(T, resid, built, shap_built, stats: dict, path: Path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    _style(a1)
    a1.scatter(T, resid, s=5, color=BLUE, alpha=0.3, linewidth=0)
    c = plot_curve(np.log10(T), resid)
    a1.plot(10 ** c.x_median, c.shap_median, color=INK, linewidth=2)
    a1.axhline(0, color=INK_2, linewidth=0.8, linestyle=(0, (2, 3)))
    a1.set_xscale("log")
    r = stats["residual_vs_T"]
    a1.set_title(f"Out-of-fold residual vs effort  (ρ = {r['rho']:.2f}, 95% CI {r['ci95'][0]:.2f} to {r['ci95'][1]:.2f})",
                 color=INK, fontsize=9.5, loc="left")
    a1.set_xlabel("Point-day sampling units, T (log scale)", color=INK_2, fontsize=8.5)
    a1.set_ylabel("Observed − predicted richness (species)", color=INK_2, fontsize=8.5)
    _style(a2)
    cmap = LinearSegmentedColormap.from_list("seq", SEQ_BLUE)
    sc_ = a2.scatter(built, shap_built, c=T, cmap=cmap, norm=LogNorm(vmin=T.min(), vmax=T.max()),
                     s=6, alpha=0.8, linewidth=0)
    a2.axhline(0, color=INK_2, linewidth=0.8, linestyle=(0, (2, 3)))
    a2.set_xlabel(label("lc_built"), color=INK_2, fontsize=8.5)
    a2.set_ylabel("SHAP for built-up fraction (species)", color=INK_2, fontsize=8.5)
    a2.set_title("Built-up fraction: SHAP, coloured by effort T", color=INK, fontsize=9.5, loc="left")
    cb = fig.colorbar(sc_, ax=a2, pad=0.01)
    cb.set_label("T (point-day units)", color=INK_2, fontsize=8.5)
    cb.ax.tick_params(colors=INK_2, labelsize=8)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# --- main -------------------------------------------------------------------
def main():
    import joblib
    import matplotlib
    matplotlib.use("Agg")

    if FROZEN.exists():
        verify_frozen()
    bundle = joblib.load(MODEL)
    preds = bundle["predictors"]
    table = pd.read_parquet(PROCESSED / "model_table.parquet")
    train = table[table.train_eligible].reset_index(drop=True)
    X = train[preds]
    sv, expected, model = shap_values(MODEL, X.to_numpy(float))
    err = check_additivity(sv, expected, model.predict(X.to_numpy(float)))
    S = pd.DataFrame(sv, columns=preds)
    S.insert(0, "cell_id", train["cell_id"].to_numpy())
    S.to_parquet(PROCESSED / "shap_train.parquet", index=False)
    S = S.drop(columns="cell_id")
    print(f"   SHAP on {len(S):,} training cells; additivity error {err:.1e}", flush=True)

    perm = pd.read_csv(TABLES / "rf_permutation_importance.csv").set_index("predictor")["importance_rmse_increase"]
    cmp = importance_comparison(S, perm)
    cmp.to_csv(TABLES / "importance_comparison.csv")
    rank_rho = float(spearmanr(cmp["shap_rank"], cmp["perm_rank"])[0])

    top = list(cmp.index[:TOP_N]) + [p for p in ALWAYS if p not in cmp.index[:TOP_N]]
    curves = {p: binned_curve(X[p], S[p]) for p in top}
    pd.concat([c.assign(predictor=p) for p, c in curves.items()]).to_csv(TABLES / "dependence_curves.csv", index=False)
    spread = float(S.abs().mean().mean())
    peaks = {p: interior_peak(curves[p], min_drop=spread) for p in top}

    # effort check
    oof = pd.read_parquet(PROCESSED / "oof_predictions.parquet")
    if not (oof["cell_id"].to_numpy() == train["cell_id"].to_numpy()).all():
        raise AssertionError("OOF rows do not line up with training cells")
    resid = oof["observed"].to_numpy() - oof["spatial_residual_rf"].to_numpy()
    T = train["T"].to_numpy(float)
    meta = json.loads((ROOT / "data" / "interim" / "grid_meta.json").read_text())
    import spatial_cv as sc
    blocks = sc.block_ids(train[["x", "y"]].to_numpy(float), 100_000, origin=(meta["x0"], meta["y0"]))
    effort = {}
    for name, a in (("observed_vs_T", train["richness_c090"]), ("residual_vs_T", resid),
                    ("lc_built_vs_T", train["lc_built"]), ("shap_lc_built_vs_T", S["lc_built"])):
        rho, lo, hi = block_bootstrap_spearman(a, T, blocks)
        effort[name] = {"rho": round(rho, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    report = {"shap_additivity_error": err, "expected_value": expected,
              "rank_agreement_spearman": round(rank_rho, 4), "top_predictors": top,
              "interior_peak_test_min_drop": spread, "peaks": peaks, "effort": effort}
    (TABLES / "interpretation_report.json").write_text(json.dumps(report, indent=2, default=float))

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig_importance(cmp, FIGURES / "shap_importance_comparison.png")
    fig_dependence(X, S, top, curves, FIGURES / "shap_dependence.png")
    fig_effort(T, resid, train["lc_built"].to_numpy(), S["lc_built"].to_numpy(), effort,
               FIGURES / "effort_check.png")

    with pd.option_context("display.width", 200):
        print(cmp.round(4).to_string())
    print(json.dumps({k: report[k] for k in ("rank_agreement_spearman", "peaks", "effort")}, indent=2, default=float))
    for p in top:
        print(f"\n{p}:\n" + curves[p][["x_low", "x_high", "x_median", "shap_median", "n"]].round(3).to_string(index=False))


def figures_only():
    """Redraw figures from saved outputs, without recomputing SHAP."""
    import matplotlib
    matplotlib.use("Agg")
    import joblib
    preds = joblib.load(MODEL)["predictors"]
    table = pd.read_parquet(PROCESSED / "model_table.parquet")
    train = table[table.train_eligible].reset_index(drop=True)
    S = pd.read_parquet(PROCESSED / "shap_train.parquet")
    if not (S.cell_id.to_numpy() == train.cell_id.to_numpy()).all():
        raise AssertionError("saved SHAP rows do not line up with training cells")
    S = S.drop(columns="cell_id")
    X = train[preds]
    rep_ = json.loads((TABLES / "interpretation_report.json").read_text())
    cmp = pd.read_csv(TABLES / "importance_comparison.csv", index_col=0)
    top = rep_["top_predictors"]
    curves = {p: binned_curve(X[p], S[p]) for p in top}
    oof = pd.read_parquet(PROCESSED / "oof_predictions.parquet")
    resid = oof["observed"].to_numpy() - oof["spatial_residual_rf"].to_numpy()
    fig_importance(cmp, FIGURES / "shap_importance_comparison.png")
    fig_dependence(X, S, top, curves, FIGURES / "shap_dependence.png")
    fig_effort(train["T"].to_numpy(float), resid, train["lc_built"].to_numpy(), S["lc_built"].to_numpy(),
               rep_["effort"], FIGURES / "effort_check.png")


if __name__ == "__main__":
    figures_only() if "--figures-only" in sys.argv else main()
