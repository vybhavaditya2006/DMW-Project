"""
Figures for the whole pipeline
==============================
Draws every v2 figure from the files the other modules save (reports/*.csv,
outputs/*.csv), so figures can be redrawn in seconds without re-running the
analysis. All figures share one style (apply_style) and go to outputs/plots_v2/.

Colours: a colour-blind-safe categorical palette used in a fixed order
(blue, orange, aqua, yellow, ...); a single-hue blue ramp for magnitudes
(e.g. lift); no dual-axis charts. Where four colours share a scatter, each
also gets its own marker shape so identity never relies on colour alone.

Usage:
    python -m src.visualize
"""

import logging
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette

from src import FIGURES_DIR, OUTPUTS_DIR, PLOTS, REPORTS_DIR
from src.clustering import knee_index
from src.profiling import load_raw
from src.wrangling import FINAL_OUTLIER_RULES, OUTLIER_METHODS, parse_sqft

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# STYLE
# ──────────────────────────────────────────────────────────────────────────────
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 8,
        "axes.labelsize": 10, "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 2, "lines.markersize": 6.5,
        "legend.frameon": False, "legend.fontsize": 9,
        "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    })


def figure(title: str, subtitle: str = "", ncols: int = 1, nrows: int = 1, size=(10, 5.5), **kw):
    """A figure with a left-aligned bold title and an optional grey subtitle."""
    fig, axes = plt.subplots(nrows, ncols, figsize=size, **kw)
    h = size[1]  # offsets in inches, so title spacing does not depend on the figure's height
    fig.suptitle(title, x=0.01, y=1 - 0.12 / h, ha="left", va="top", fontsize=14, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.01, 1 - 0.45 / h, subtitle, ha="left", va="top", fontsize=9.5, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.8 if subtitle else 0.5) / h))
    return fig, axes


def save(fig, key: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / PLOTS[key]
    fig.savefig(path)
    plt.close(fig)
    return path


def thousands(ax, axis="y"):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


# ──────────────────────────────────────────────────────────────────────────────
# WRANGLING
# ──────────────────────────────────────────────────────────────────────────────
def plot_distributions() -> Path:
    """Before (raw, only total_sqft parsed) vs after (clean) for price, sqft and price per sqft."""
    raw = load_raw()
    before = pd.DataFrame({"price": raw["price"], "total_sqft": raw["total_sqft"].map(lambda v: parse_sqft(v)[0])})
    before["price_per_sqft"] = before["price"] * 1e5 / before["total_sqft"]
    after = pd.read_csv(OUTPUTS_DIR / "clean_data.csv")
    panels = [("price", "Price (lakhs, log scale)"), ("total_sqft", "Total area (sq ft, log scale)"),
              ("price_per_sqft", "Price per sq ft (₹, log scale)")]
    fig, axes = figure("Distributions before and after wrangling",
                       "Raw data with only total_sqft parsed, vs outputs/clean_data.csv. Log-spaced bins; each axis "
                       "spans the raw 0.1–99.9th percentiles, so a few extreme raw values fall outside it.",
                       ncols=3, size=(15, 5))
    for ax, (col, label) in zip(axes, panels):
        b, a = before[col].dropna(), after[col]
        b = b[b > 0]
        lo, hi = min(b.quantile(0.001), a.min()), max(b.quantile(0.999), a.max())
        outside = int(((b < lo) | (b > hi)).sum())
        bins = np.logspace(np.log10(lo), np.log10(hi), 50)
        ax.hist(b.clip(lo, hi), bins=bins, color=SERIES[0], alpha=0.45,
                label=f"Before: n={len(b):,}, skew {b.skew():.1f}")
        ax.hist(a, bins=bins, histtype="step", color=SERIES[1], linewidth=2,
                label=f"After: n={len(a):,}, skew {a.skew():.1f}")
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ax.set_xlabel(label)
        ax.set_ylabel("Listings")
        ax.set_title(f"{col}" + (f"  ({outside} raw values beyond the axis)" if outside else ""))
        ax.legend(loc="best")
    return save(fig, "distributions")


def plot_waterfall() -> Path:
    log_df = pd.read_csv(REPORTS_DIR / "wrangling_log.csv")
    labels = ["Raw CSV"] + [textwrap.fill(t, 16) for t in log_df["title"]] + ["Clean data"]
    fig, ax = figure("Rows kept at each wrangling step",
                     f"{log_df['rows_before'].iloc[0]:,} raw listings → {log_df['rows_after'].iloc[-1]:,} clean "
                     f"listings. Orange = rows removed by the step.", size=(13, 5.5))
    x = np.arange(len(labels))
    ax.bar(x[0], log_df["rows_before"].iloc[0], color=SERIES[0], width=0.6)
    for i, r in enumerate(log_df.itertuples(), start=1):
        removed = r.rows_before - r.rows_after
        ax.bar(i, removed if removed else 0, bottom=r.rows_after, color=SERIES[1], width=0.6)
        ax.plot([i - 0.3, i + 0.3], [r.rows_after] * 2, color=MUTED, linewidth=1)  # level after the step
        ax.annotate(f"−{removed:,}" if removed else "0", (i, r.rows_before), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=9, color=INK if removed else MUTED)
    ax.bar(x[-1], log_df["rows_after"].iloc[-1], color=SERIES[0], width=0.6)
    for i, v in [(0, log_df["rows_before"].iloc[0]), (len(x) - 1, log_df["rows_after"].iloc[-1])]:
        ax.annotate(f"{v:,}", (i, v), xytext=(0, 4), textcoords="offset points", ha="center", fontweight="bold")
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.set_ylabel("Listings")
    ax.set_ylim(0, log_df["rows_before"].iloc[0] * 1.08)
    thousands(ax)
    ax.grid(axis="x", visible=False)
    return save(fig, "waterfall")


def plot_outliers() -> Path:
    flags = pd.read_csv(OUTPUTS_DIR / "outlier_flags.csv").drop(columns="raw_row").astype(bool)
    methods = list(OUTLIER_METHODS)
    families = list(dict.fromkeys(v[0] for v in OUTLIER_METHODS.values()))
    fig, (ax, hm) = figure("Outlier detection: nine methods compared",
                           f"{len(flags):,} listings before removal. Methods marked ✓ form the final strategy; the "
                           "others are kept as diagnostic flags only.", ncols=2, size=(15, 6),
                           gridspec_kw={"width_ratios": [1, 1.1]})
    order = flags[methods].sum().sort_values().index
    for i, m in enumerate(order):
        fam = OUTLIER_METHODS[m][0]
        used = m in FINAL_OUTLIER_RULES
        ax.barh(i, flags[m].sum(), color=SERIES[families.index(fam)], height=0.65,
                edgecolor=INK if used else "none", linewidth=1.2)
        ax.annotate(f"{flags[m].sum():,} ({flags[m].mean():.1%}){'  ✓' if used else ''}", (flags[m].sum(), i),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=9)
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel("Listings flagged")
    ax.set_title("Rows flagged by each method")
    ax.set_xlim(0, flags.sum().max() * 1.3)
    thousands(ax, "x")
    ax.legend(handles=[matplotlib.patches.Patch(color=SERIES[i], label=f) for i, f in enumerate(families)],
              loc="lower right", title="Family")

    jac = pd.DataFrame([[(flags[a] & flags[b]).sum() / max((flags[a] | flags[b]).sum(), 1) for b in order]
                        for a in order], index=order, columns=order)
    im = hm.imshow(jac, cmap=BLUE_RAMP, vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(jac.to_numpy()):
        hm.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color="white" if v > 0.55 else INK)
    hm.set_xticks(range(len(order)), order, rotation=45, ha="right", fontsize=8.5)
    hm.set_yticks(range(len(order)), order, fontsize=8.5)
    hm.grid(False)
    hm.set_title("Agreement between methods (Jaccard overlap of flagged rows)")
    fig.colorbar(im, ax=hm, shrink=0.8, label="Jaccard (0 = no overlap, 1 = identical)")
    return save(fig, "outliers")


# ──────────────────────────────────────────────────────────────────────────────
# MODELLING
# ──────────────────────────────────────────────────────────────────────────────
def plot_models() -> Path:
    comp = pd.read_csv(REPORTS_DIR / "model_comparison.csv")
    order = comp.groupby("model")["cv_r2_mean"].max().sort_values().index
    fig, axes = figure("Model comparison: 5-fold cross-validation on the training split",
                       "Point = mean across folds, bar = ± 1 std. Each model is fitted to price and to log price "
                       "(log-log); errors are always in lakhs.", ncols=2, size=(14, 5.5), sharey=True)
    for ax, (metric, label) in zip(axes, [("r2", "CV R² (higher is better)"), ("mae", "CV MAE, lakhs (lower is better)")]):
        for j, (target, name) in enumerate([("price", "price target"), ("log_price", "log-price target")]):
            sub = comp[comp["target"] == target].set_index("model").loc[order]
            y = np.arange(len(order)) + (j - 0.5) * 0.3
            ax.errorbar(sub[f"cv_{metric}_mean"], y, xerr=sub[f"cv_{metric}_std"], fmt=MARKERS[j],
                        color=SERIES[j], capsize=3, label=name, markersize=7)
        ax.set_xlabel(label)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(order)), order)
    axes[0].set_xlim(left=max(0, (comp["cv_r2_mean"] - comp["cv_r2_std"]).min() - 0.03))
    axes[0].legend(loc="lower right")
    axes[0].set_title("R²")
    axes[1].set_title("Mean absolute error")
    return save(fig, "models")


def plot_predictions() -> Path:
    pred = pd.read_csv(OUTPUTS_DIR / "test_predictions.csv")
    actual, predicted = pred["actual"], pred["predicted"]
    r2 = 1 - ((actual - predicted) ** 2).sum() / ((actual - actual.mean()) ** 2).sum()
    mae = (actual - predicted).abs().mean()
    pct = (actual - predicted) / predicted * 100
    fig, (ax, rs) = figure("Best model on the held-out test set",
                           f"{len(pred):,} listings never seen in training or tuning. R² {r2:.3f}, "
                           f"MAE {mae:.2f} lakhs.", ncols=2, size=(14, 6))
    ax.scatter(actual, predicted, s=10, alpha=0.35, color=SERIES[0], edgecolors="none")
    lims = [min(actual.min(), predicted.min()) * 0.9, max(actual.max(), predicted.max()) * 1.1]
    ax.plot(lims, lims, color=INK, linestyle="--", linewidth=1.2, label="perfect prediction")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual price (lakhs, log scale)")
    ax.set_ylabel("Predicted price (lakhs, log scale)")
    ax.set_title("Predicted vs actual")
    ax.legend(loc="upper left")

    clip = 150
    outside = int((pct.abs() > clip).sum())
    rs.scatter(predicted, pct.clip(-clip, clip), s=10, alpha=0.35, color=SERIES[0], edgecolors="none")
    rs.axhline(0, color=INK, linewidth=1.2)
    for q, style in [(0.25, ":"), (0.75, ":")]:
        rs.axhline(pct.quantile(q), color=SERIES[1], linestyle=style, linewidth=1.5)
    rs.annotate(f"middle 50% of errors: {pct.quantile(0.25):+.0f}% to {pct.quantile(0.75):+.0f}%",
                (0.02, 0.95), xycoords="axes fraction", color=SERIES[1], fontsize=9)
    rs.set_xscale("log")
    rs.set_xlabel("Predicted price (lakhs, log scale)")
    rs.set_ylabel(f"Error: (actual − predicted) / predicted, % (clipped at ±{clip}%)")
    rs.set_title(f"Residuals ({outside} points beyond ±{clip}% drawn at the edge)")
    return save(fig, "predictions")


# ──────────────────────────────────────────────────────────────────────────────
# CLUSTERING
# ──────────────────────────────────────────────────────────────────────────────
def plot_k_selection() -> Path:
    ksel = pd.read_csv(REPORTS_DIR / "kmeans_k_selection.csv")
    k = int(pd.read_csv(REPORTS_DIR / "ward_linkage.csv")["chosen_k"].iloc[0])
    elbow = int(ksel["k"].iloc[knee_index(ksel["inertia"])])
    fig, axes = figure("K-Means: choosing the number of clusters",
                       f"Chosen k = {k}: lowest Davies-Bouldin among the k whose silhouette is within 10% of the "
                       f"best. Elbow of the inertia curve at k = {elbow}.", ncols=3, size=(15, 4.6))
    for ax, (col, label, better) in zip(axes, [("inertia", "Inertia (within-cluster sum of squares)", "elbow"),
                                               ("silhouette", "Silhouette score", "higher is better"),
                                               ("davies_bouldin", "Davies-Bouldin index", "lower is better")]):
        ax.plot(ksel["k"], ksel[col], marker="o", color=SERIES[0])
        ax.axvline(k, color=SERIES[1], linestyle="--", linewidth=1.5, label=f"chosen k = {k}")
        ax.set_xlabel("k (number of clusters)")
        ax.set_ylabel(label)
        ax.set_title(f"{label.split(' (')[0]} ({better})")
        ax.set_xticks(ksel["k"])
    axes[0].scatter([elbow], ksel.set_index("k").loc[[elbow], "inertia"], s=140, facecolors="none",
                    edgecolors=INK, linewidths=1.5, label=f"elbow k = {elbow}", zorder=3)
    axes[0].legend()
    return save(fig, "k_selection")


def plot_k_distance() -> Path:
    kd = pd.read_csv(REPORTS_DIR / "dbscan_k_distance.csv")
    eps, ms = kd["eps"].iloc[0], int(kd["min_samples"].iloc[0])
    noise = int((kd["k_distance"] > eps).sum())
    fig, ax = figure("DBSCAN: choosing eps from the k-distance curve",
                     f"Distance from each location to its {ms}th nearest neighbour, sorted. eps is set at the knee; "
                     f"the {noise} locations above it cannot be core points.", size=(10, 5.2))
    ax.plot(kd["rank"], kd["k_distance"], color=SERIES[0])
    ax.axhline(eps, color=SERIES[1], linestyle="--", linewidth=1.5, label=f"eps = {eps:.2f} (knee)")
    ax.fill_between(kd["rank"], eps, kd["k_distance"], where=kd["k_distance"] > eps, color=SERIES[1], alpha=0.15)
    ax.set_xlabel("Locations, sorted by k-distance")
    ax.set_ylabel(f"Distance to {ms}th nearest neighbour (scaled units)")
    ax.legend(loc="upper left")
    return save(fig, "k_distance")


def plot_dendrogram() -> Path:
    lk = pd.read_csv(REPORTS_DIR / "ward_linkage.csv")
    k = int(lk["chosen_k"].iloc[0])
    Z = lk[["left", "right", "distance", "size"]].to_numpy()
    cut = (Z[-k, 2] + Z[-(k - 1), 2]) / 2  # between the merges that leave k and k-1 clusters
    fig, ax = figure("Agglomerative clustering (Ward linkage): dendrogram",
                     f"Last 30 merges of {len(Z) + 1} locations; leaf labels in brackets are cluster sizes. "
                     f"The dashed line cuts the tree into k = {k} clusters.", size=(13, 5.5))
    set_link_color_palette(SERIES[:k])
    dendrogram(Z, truncate_mode="lastp", p=30, color_threshold=cut, above_threshold_color=MUTED, ax=ax,
               leaf_font_size=8.5)
    set_link_color_palette(None)
    ax.axhline(cut, color=INK, linestyle="--", linewidth=1.2, label=f"cut for k = {k}")
    ax.set_xlabel("Locations (cluster sizes in brackets)")
    ax.set_ylabel("Ward merge distance")
    ax.grid(axis="x", visible=False)
    ax.legend()
    return save(fig, "dendrogram")


def plot_clusters() -> Path:
    """Small multiples, one panel per segment (highlighted over the others in grey).

    Four colours in one scatter fail the palette's all-pairs checks (orange vs yellow are too close even
    for normal vision), so each segment gets its own panel in the same colour instead.
    """
    loc = pd.read_csv(REPORTS_DIR / "location_clusters.csv")
    load = pd.read_csv(REPORTS_DIR / "pca_loadings.csv", index_col="feature")
    order = loc.groupby("segment")["mean_pps"].mean().sort_values(ascending=False).index
    v1, v2 = load["pc1_variance"].iloc[0], load["pc2_variance"].iloc[0]
    ncols = 2
    nrows = int(np.ceil(len(order) / ncols))
    fig, axes = figure("Bengaluru locations by market segment (K-Means, shown with PCA)",
                       f"Each point is a location; size grows with its listing count. PC1 and PC2 keep {v1 + v2:.0%} "
                       "of the variance of the 5 scaled location features. Arrows: direction of each feature.",
                       ncols=ncols, nrows=nrows, size=(14, 5.2 * nrows), sharex=True, sharey=True)
    reach = np.abs(loc[["pc1", "pc2"]]).max().max() * 0.5
    size = lambda d: 12 + 2.2 * np.sqrt(d["listings"])
    for ax, seg in zip(axes.flat, order):
        s, rest = loc[loc["segment"] == seg], loc[loc["segment"] != seg]
        ax.scatter(rest["pc1"], rest["pc2"], s=size(rest), color=GRID, edgecolors="none")
        ax.scatter(s["pc1"], s["pc2"], s=size(s), color=SERIES[0], alpha=0.85, edgecolors=SURFACE, linewidths=0.6)
        prof = s[["mean_pps", "median_sqft", "listings", "pct_ready"]].mean()
        ax.set_title(f"{seg}: {len(s)} locations", fontsize=10.5)
        ax.annotate(f"avg ₹{prof['mean_pps']:,.0f}/sq ft · {prof['median_sqft']:,.0f} sq ft · "
                    f"{prof['listings']:.0f} listings · {prof['pct_ready']:.0%} ready",
                    (0.01, 0.97), xycoords="axes fraction", fontsize=8.5, color=INK_2, va="top")
        for j, r in enumerate(s.nlargest(4, "listings").itertuples()):
            ax.annotate(r.location, (r.pc1, r.pc2), xytext=(5, 4 if j % 2 else -9), textcoords="offset points",
                        fontsize=8, color=INK)
        for f, r in load.iterrows():
            ax.annotate("", (r["pc1"] * reach, r["pc2"] * reach), (0, 0),
                        arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1})
    # Label the arrows once, in the first panel; arrows ending at almost the same point share one label.
    groups: list[list[str]] = []
    for f in load.index:
        near = next((g for g in groups if np.hypot(*(load.loc[f, ["pc1", "pc2"]] - load.loc[g[0], ["pc1", "pc2"]]))
                     < 0.15), None)
        (near.append(f) if near else groups.append([f]))
    for g in groups:
        tip = load.loc[g, ["pc1", "pc2"]].mean() * reach * 1.12
        axes.flat[0].annotate(" / ".join(g), tuple(tip), ha="left" if tip["pc1"] >= 0 else "right",
                              va="bottom" if tip["pc2"] >= 0 else "top", fontsize=8, color=INK_2)
    for ax in axes.flat[len(order):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(f"PC1 ({v1:.0%}): price per sq ft, size, bedrooms →")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"PC2 ({v2:.0%}): ↑ ready to move · ↓ listing volume")
    return save(fig, "clusters")


# ──────────────────────────────────────────────────────────────────────────────
# ASSOCIATION RULES
# ──────────────────────────────────────────────────────────────────────────────
def plot_rules() -> Path:
    rules = pd.read_csv(REPORTS_DIR / "association_rules_all.csv")
    top = pd.read_csv(REPORTS_DIR / "association_rules.csv")
    fig, ax = figure("Association rules: support vs confidence, coloured by lift",
                     f"All {len(rules):,} non-redundant rules with lift ≥ 1.2 and confidence ≥ 50% (one per itemset). "
                     f"Ringed: the {len(top)} selected rules; numbers: the top 5.", size=(11, 6.5))
    sc = ax.scatter(rules["support"] * 100, rules["confidence"] * 100, c=rules["lift"], cmap=BLUE_RAMP, s=22,
                    alpha=0.85, edgecolors="none")
    ax.scatter(top["support"] * 100, top["confidence"] * 100, s=70, facecolors="none", edgecolors=INK,
               linewidths=1.1, label="selected top rules")
    for r in top.head(5).itertuples():
        ax.annotate(str(r.rank), (r.support * 100, r.confidence * 100), xytext=(6, 4), textcoords="offset points",
                    fontsize=9, fontweight="bold")
    fig.colorbar(sc, ax=ax, label="Lift (× more likely than by chance)")
    ax.set_xscale("log")
    ticks = [t for t in (2, 3, 5, 10, 20, 30, 50, 75) if t <= rules["support"].max() * 100 * 1.1]
    ax.set_xticks(ticks, [f"{t}%" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Support: share of all listings covered by the rule (log scale)")
    ax.set_ylabel("Confidence: P(then | if), %")
    ax.legend(loc="lower right")
    return save(fig, "rules")


# ──────────────────────────────────────────────────────────────────────────────
# FEATURES
# ──────────────────────────────────────────────────────────────────────────────
def plot_importance() -> Path:
    imp = pd.read_csv(REPORTS_DIR / "feature_importance.csv", index_col="feature").sort_values("perm_r2_drop")
    fig, axes = figure("Feature importance for the best model",
                       "Built-in gain is measured while training; permutation importance shuffles one input on the "
                       "held-out test set. Location's dummies are summed / shuffled together.",
                       ncols=3, size=(15, 4.6), sharey=True)
    y = np.arange(len(imp))
    axes[0].barh(y, imp["gain_share"], color=SERIES[0], height=0.6)
    axes[0].set_xlabel("Share of total gain")
    axes[0].set_title("Built-in (XGBoost gain)")
    axes[1].barh(y, imp["perm_r2_drop"], xerr=imp["perm_r2_drop_std"], color=SERIES[1], height=0.6, capsize=3)
    axes[1].set_xlabel("Drop in test R² when shuffled")
    axes[1].set_title("Permutation: R² drop")
    axes[2].barh(y, imp["perm_mae_increase"], xerr=imp["perm_mae_increase_std"], color=SERIES[1], height=0.6,
                 capsize=3)
    axes[2].set_xlabel("Increase in test MAE when shuffled (lakhs)")
    axes[2].set_title("Permutation: MAE increase")
    axes[0].set_yticks(y, imp.index)
    for ax in axes:
        ax.axvline(0, color=INK_2, linewidth=0.8)
        ax.grid(axis="y", visible=False)
    return save(fig, "importance")


def plot_selection() -> Path:
    sel = pd.read_csv(REPORTS_DIR / "feature_selection.csv")
    base = sel.iloc[0]
    fig, ax = figure("Feature selection: can fewer columns keep the accuracy?",
                     f"5-fold CV R² (± 1 std) of the best model with only k of the {int(base['k'])} encoded columns. "
                     "Selection is redone inside every training fold.", size=(10, 5.5))
    for i, (method, g) in enumerate(sel.iloc[1:].groupby("method", sort=False)):
        ax.errorbar(g["k"], g["cv_r2_mean"], yerr=g["cv_r2_std"], marker=MARKERS[i], color=SERIES[i], capsize=3,
                    label=method, markersize=7)
    ax.axhline(base["cv_r2_mean"], color=INK, linestyle="--", linewidth=1.2,
               label=f"all {int(base['k'])} columns (R² {base['cv_r2_mean']:.3f})")
    ax.axhspan(base["cv_r2_mean"] - 0.01, base["cv_r2_mean"], color=MUTED, alpha=0.18, label="within 0.01 R²")
    ks = sorted(sel["k"].iloc[1:].unique())
    ax.set_xscale("log")
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.minorticks_off()
    ax.set_xlabel("Number of encoded columns kept, k (log scale)")
    ax.set_ylabel("5-fold CV R²")
    ax.legend(loc="lower right")
    return save(fig, "selection")


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
ALL_PLOTS = [plot_distributions, plot_waterfall, plot_outliers, plot_models, plot_predictions, plot_k_selection,
             plot_k_distance, plot_dendrogram, plot_clusters, plot_rules, plot_importance, plot_selection]


def run_visualize() -> list[Path]:
    apply_style()
    saved = []
    for fn in ALL_PLOTS:
        try:
            path = fn()
        except FileNotFoundError as e:
            log.warning("Skipped %s: input missing (%s). Run the stage that produces it first.",
                        fn.__name__, Path(e.filename).name if e.filename else e)
            continue
        saved.append(path)
        log.info("Saved figure: %s", path.name)
    return saved


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_visualize()


if __name__ == "__main__":
    main()
