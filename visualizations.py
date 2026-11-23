"""
Presentation-Ready Visualizations - Bengaluru House Price Dataset
==================================================================
Generates three clean, well-labeled plots:
  1. Scatter plot: total_sqft vs price
  2. Bar chart: average price by BHK
  3. Heatmap: correlation between numerical features

All plots saved to output_plots/ directory.

Usage:
    python visualizations.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ------------------------------------------------------------------------------
# 1. LOAD & CLEAN DATASET
# ------------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_CANDIDATES = ["Bengaluru_House_Data.csv", "bengaluru_house_prices.csv"]
df = None
for name in CSV_CANDIDATES:
    path = os.path.join(SCRIPT_DIR, name)
    if os.path.isfile(path):
        df = pd.read_csv(path)
        break
if df is None:
    raise FileNotFoundError("Dataset not found.")

# Clean
df.drop(columns=[c for c in ["area_type", "society", "balcony", "availability"]
                 if c in df.columns], inplace=True)
if "size" in df.columns:
    df["bhk"] = df["size"].str.extract(r"(\d+)").astype(float)
    df.drop(columns=["size"], inplace=True)

def convert_sqft(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        if isinstance(x, str) and "-" in x:
            parts = x.split("-")
            try:
                return (float(parts[0].strip()) + float(parts[1].strip())) / 2.0
            except (ValueError, IndexError):
                return np.nan
        return np.nan

df["total_sqft"] = df["total_sqft"].apply(convert_sqft)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
df["price_per_sqft"] = (df["price"] * 100000) / df["total_sqft"]

print(f"[OK] Dataset loaded and cleaned. Shape: {df.shape}")

# ------------------------------------------------------------------------------
# GLOBAL STYLE
# ------------------------------------------------------------------------------
COLORS = {
    "bg":         "#FAFBFD",
    "text":       "#2C3E50",
    "accent":     "#3498DB",
    "accent2":    "#E74C3C",
    "grid":       "#E8ECF1",
    "bar_grad":   ["#2980B9", "#3498DB", "#5DADE2", "#85C1E9",
                   "#AED6F1", "#D4E6F1"],
}

plt.rcParams.update({
    "figure.facecolor":    COLORS["bg"],
    "axes.facecolor":      COLORS["bg"],
    "axes.edgecolor":      COLORS["grid"],
    "axes.labelcolor":     COLORS["text"],
    "text.color":          COLORS["text"],
    "xtick.color":         COLORS["text"],
    "ytick.color":         COLORS["text"],
    "grid.color":          COLORS["grid"],
    "grid.linewidth":      0.6,
    "font.family":         "sans-serif",
    "font.size":           11,
    "axes.titlesize":      15,
    "axes.labelsize":      12,
    "axes.titleweight":    "bold",
    "figure.dpi":          150,
})
sns.set_style("whitegrid")

# ==============================================================================
# PLOT 1  -  Scatter Plot: Total Sqft vs Price
# ==============================================================================
print("\n[1/3] Generating scatter plot: total_sqft vs price ...")

fig, ax = plt.subplots(figsize=(11, 7))

# Filter to remove extreme outliers for cleaner visual
plot_df = df[(df["total_sqft"] <= df["total_sqft"].quantile(0.98)) &
             (df["price"] <= df["price"].quantile(0.98))]

scatter = ax.scatter(
    plot_df["total_sqft"], plot_df["price"],
    c=plot_df["price"], cmap="coolwarm",
    s=18, alpha=0.55, edgecolors="none",
    vmin=plot_df["price"].quantile(0.05),
    vmax=plot_df["price"].quantile(0.95),
)

# Trend line
z = np.polyfit(plot_df["total_sqft"], plot_df["price"], 1)
p = np.poly1d(z)
x_line = np.linspace(plot_df["total_sqft"].min(), plot_df["total_sqft"].max(), 200)
ax.plot(x_line, p(x_line), color=COLORS["accent2"], linewidth=2.2,
        linestyle="--", label=f"Linear Trend  (r = {plot_df['total_sqft'].corr(plot_df['price']):.3f})")

# Colour bar
cbar = plt.colorbar(scatter, ax=ax, pad=0.02, shrink=0.85)
cbar.set_label("Price (Lakhs)", fontsize=11)

ax.set_title("Total Square Feet vs Property Price")
ax.set_xlabel("Total Square Feet")
ax.set_ylabel("Price (Lakhs)")
ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

plt.tight_layout()
path1 = os.path.join(OUTPUT_DIR, "viz_01_sqft_vs_price.png")
plt.savefig(path1, bbox_inches="tight")
plt.close()
print(f"  [SAVED] {path1}")

# ==============================================================================
# PLOT 2  -  Bar Chart: Average Price by BHK
# ==============================================================================
print("\n[2/3] Generating bar chart: average price by BHK ...")

# Focus on BHK 1-6 (the meaningful, well-populated groups)
bhk_range = [1, 2, 3, 4, 5, 6]
bhk_df = df[df["bhk"].isin(bhk_range)]
bhk_avg = bhk_df.groupby("bhk")["price"].mean()
bhk_med = bhk_df.groupby("bhk")["price"].median()
bhk_cnt = bhk_df.groupby("bhk")["price"].count()

fig, ax = plt.subplots(figsize=(10, 7))

bar_colors = sns.color_palette("Blues_d", n_colors=len(bhk_range))[::-1]
bars = ax.bar(
    [str(int(b)) for b in bhk_avg.index],
    bhk_avg.values,
    color=bar_colors,
    edgecolor="white",
    linewidth=1.2,
    width=0.6,
    zorder=3,
)

# Value labels on each bar
for bar, avg, med, cnt in zip(bars, bhk_avg.values, bhk_med.values, bhk_cnt.values):
    # Average label at top
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4,
            f"Rs {avg:,.0f}L",
            ha="center", va="bottom", fontsize=10.5, fontweight="bold",
            color=COLORS["text"])
    # Count label inside bar near bottom
    ax.text(bar.get_x() + bar.get_width() / 2, 8,
            f"n={cnt:,}",
            ha="center", va="bottom", fontsize=9,
            color="white", fontstyle="italic")

ax.set_title("Average Property Price by Number of Bedrooms (BHK)")
ax.set_xlabel("BHK (Number of Bedrooms)")
ax.set_ylabel("Average Price (Lakhs)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.set_ylim(0, bhk_avg.max() * 1.18)
ax.grid(axis="y", alpha=0.4, zorder=0)
ax.grid(axis="x", visible=False)

plt.tight_layout()
path2 = os.path.join(OUTPUT_DIR, "viz_02_avg_price_by_bhk.png")
plt.savefig(path2, bbox_inches="tight")
plt.close()
print(f"  [SAVED] {path2}")

# ==============================================================================
# PLOT 3  -  Heatmap: Correlation Between Numerical Features
# ==============================================================================
print("\n[3/3] Generating correlation heatmap ...")

numeric_cols = ["total_sqft", "bath", "price", "bhk", "price_per_sqft"]
corr = df[numeric_cols].corr()

# Readable labels
label_map = {
    "total_sqft":     "Total Sqft",
    "bath":           "Bathrooms",
    "price":          "Price",
    "bhk":            "BHK",
    "price_per_sqft": "Price/Sqft",
}
corr_display = corr.rename(index=label_map, columns=label_map)

fig, ax = plt.subplots(figsize=(9, 7.5))

mask = np.triu(np.ones_like(corr_display, dtype=bool))

sns.heatmap(
    corr_display, mask=mask,
    annot=True, fmt=".2f",
    cmap="RdYlBu_r", center=0,
    vmin=-1, vmax=1,
    square=True,
    linewidths=2, linecolor="white",
    cbar_kws={"shrink": 0.78, "label": "Pearson Correlation"},
    annot_kws={"size": 13, "fontweight": "bold"},
    ax=ax,
)

ax.set_title("Correlation Between Numerical Features", pad=18)
ax.tick_params(axis="both", labelsize=12)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
plt.setp(ax.get_yticklabels(), rotation=0)

plt.tight_layout()
path3 = os.path.join(OUTPUT_DIR, "viz_03_correlation_heatmap.png")
plt.savefig(path3, bbox_inches="tight")
plt.close()
print(f"  [SAVED] {path3}")

# ==============================================================================
# DONE
# ==============================================================================
print("\n" + "=" * 70)
print("  [OK] All 3 visualizations saved to:")
print(f"       {OUTPUT_DIR}")
print()
print(f"  1. {os.path.basename(path1)}")
print(f"  2. {os.path.basename(path2)}")
print(f"  3. {os.path.basename(path3)}")
print("=" * 70)
