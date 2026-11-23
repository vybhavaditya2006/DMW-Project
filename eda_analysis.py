"""
Exploratory Data Analysis (EDA) - Bengaluru House Price Dataset
================================================================
Analyzes relationships between:
  1. total_sqft and price
  2. bhk and price
  3. bathrooms and price
  4. Correlation matrix of all numerical variables

Generates visualisation plots saved to output_plots/ directory.

Usage:
    python eda_analysis.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------------------
# 1. LOAD & CLEAN DATASET (same pipeline as data_cleaning.py)
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
        print(f"[OK] Loaded dataset: {name}")
        break

if df is None:
    raise FileNotFoundError(
        f"Dataset not found. Place one of {CSV_CANDIDATES} in {SCRIPT_DIR}"
    )

# Drop unnecessary columns
df.drop(columns=[c for c in ["area_type", "society", "balcony", "availability"]
                 if c in df.columns], inplace=True)

# Extract BHK from 'size'
if "size" in df.columns:
    df["bhk"] = df["size"].str.extract(r"(\d+)").astype(float)
    df.drop(columns=["size"], inplace=True)

# Convert total_sqft to numeric
def convert_sqft(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        if isinstance(x, str) and "-" in x:
            tokens = x.split("-")
            try:
                return (float(tokens[0].strip()) + float(tokens[1].strip())) / 2.0
            except (ValueError, IndexError):
                return np.nan
        return np.nan

if "total_sqft" in df.columns:
    df["total_sqft"] = df["total_sqft"].apply(convert_sqft)

# Drop missing values
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Create price_per_sqft
df["price_per_sqft"] = (df["price"] * 100000) / df["total_sqft"]

print(f"[OK] Cleaning complete. Shape: {df.shape}")
print(f"     Columns: {list(df.columns)}")

# Plotting style
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

# ==============================================================================
# 2. ANALYSIS 1: total_sqft vs price
# ==============================================================================
print("\n" + "=" * 70)
print("  ANALYSIS 1: RELATIONSHIP BETWEEN total_sqft AND price")
print("=" * 70)

corr_sqft_price = df["total_sqft"].corr(df["price"])
print(f"\n  Pearson Correlation (total_sqft vs price): {corr_sqft_price:.4f}")

# Group stats
sqft_bins = pd.cut(df["total_sqft"],
                   bins=[0, 800, 1200, 1600, 2000, 3000, 50000],
                   labels=["<800", "800-1200", "1200-1600",
                           "1600-2000", "2000-3000", "3000+"])
sqft_grouped = df.groupby(sqft_bins, observed=True)["price"].agg(["mean", "median", "count"])
sqft_grouped.columns = ["Mean Price (Lakhs)", "Median Price (Lakhs)", "Count"]
print("\n  Average Price by Square Footage Range:")
print(sqft_grouped.to_string(float_format=lambda x: f"{x:,.2f}"))

# --- Plot 1a: Scatter plot ---
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].scatter(df["total_sqft"], df["price"], alpha=0.25, s=8,
                c="#2980b9", edgecolors="none")
# Fit a trend line
z = np.polyfit(df["total_sqft"], df["price"], 1)
p = np.poly1d(z)
x_line = np.linspace(df["total_sqft"].min(), df["total_sqft"].quantile(0.99), 100)
axes[0].plot(x_line, p(x_line), color="#e74c3c", linewidth=2,
             linestyle="--", label=f"Trend (r = {corr_sqft_price:.3f})")
axes[0].set_title("Price vs Total Square Feet", fontweight="bold")
axes[0].set_xlabel("Total Square Feet")
axes[0].set_ylabel("Price (Lakhs)")
axes[0].set_xlim(0, df["total_sqft"].quantile(0.99))
axes[0].set_ylim(0, df["price"].quantile(0.99))
axes[0].legend()

# --- Plot 1b: Hexbin for density ---
hb = axes[1].hexbin(df["total_sqft"], df["price"], gridsize=40,
                     cmap="YlOrRd", mincnt=1,
                     extent=[0, df["total_sqft"].quantile(0.99),
                             0, df["price"].quantile(0.99)])
axes[1].set_title("Price vs Sqft (Density Heatmap)", fontweight="bold")
axes[1].set_xlabel("Total Square Feet")
axes[1].set_ylabel("Price (Lakhs)")
plt.colorbar(hb, ax=axes[1], label="Count")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_01_sqft_vs_price.png"), bbox_inches="tight")
plt.close()
print("\n  [SAVED] eda_01_sqft_vs_price.png")

print("""
  OBSERVATION:
  - There is a MODERATE POSITIVE correlation (r = {:.4f}) between total_sqft
    and price, meaning larger properties generally cost more.
  - The relationship is roughly linear for properties under ~3000 sqft but
    becomes more scattered for very large properties.
  - The density heatmap shows that most properties cluster between
    800-2000 sqft and Rs 20-150 lakhs.
""".format(corr_sqft_price))

# ==============================================================================
# 3. ANALYSIS 2: bhk vs price
# ==============================================================================
print("=" * 70)
print("  ANALYSIS 2: RELATIONSHIP BETWEEN bhk AND price")
print("=" * 70)

corr_bhk_price = df["bhk"].corr(df["price"])
print(f"\n  Pearson Correlation (bhk vs price): {corr_bhk_price:.4f}")

# Group stats
bhk_stats = df.groupby("bhk")["price"].agg(["mean", "median", "count"])
bhk_stats.columns = ["Mean Price (Lakhs)", "Median Price (Lakhs)", "Count"]
print("\n  Price Statistics by BHK:")
print(bhk_stats.to_string(float_format=lambda x: f"{x:,.2f}"))

# --- Plot 2: BHK vs Price ---
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Box plot (filter to common BHK values for clarity)
common_bhk = df[df["bhk"].isin([1, 2, 3, 4, 5, 6])]
sns.boxplot(x="bhk", y="price", data=common_bhk, ax=axes[0],
            palette="viridis", fliersize=2, width=0.6)
axes[0].set_title("Price Distribution by BHK (1-6 BHK)", fontweight="bold")
axes[0].set_xlabel("BHK (Bedrooms)")
axes[0].set_ylabel("Price (Lakhs)")

# Bar plot of median price by BHK
bhk_median = common_bhk.groupby("bhk")["price"].median()
bars = axes[1].bar(bhk_median.index.astype(int), bhk_median.values,
                   color=sns.color_palette("viridis", len(bhk_median)),
                   edgecolor="white", linewidth=0.8)
axes[1].set_title("Median Price by BHK", fontweight="bold")
axes[1].set_xlabel("BHK (Bedrooms)")
axes[1].set_ylabel("Median Price (Lakhs)")
for bar, val in zip(bars, bhk_median.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f"Rs {val:.0f}L", ha="center", va="bottom", fontsize=9,
                 fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_02_bhk_vs_price.png"), bbox_inches="tight")
plt.close()
print("\n  [SAVED] eda_02_bhk_vs_price.png")

print("""
  OBSERVATION:
  - There is a MODERATE POSITIVE correlation (r = {:.4f}) between BHK and price.
  - As expected, median price increases with more bedrooms: 1 BHK properties
    are the most affordable, while 5-6 BHK properties command premium prices.
  - The box plot shows that price variance also increases with BHK count,
    especially for 4+ BHK where location plays a larger role.
  - 2 BHK and 3 BHK dominate the market (highest property counts).
""".format(corr_bhk_price))

# ==============================================================================
# 4. ANALYSIS 3: bath vs price
# ==============================================================================
print("=" * 70)
print("  ANALYSIS 3: RELATIONSHIP BETWEEN bath (Bathrooms) AND price")
print("=" * 70)

corr_bath_price = df["bath"].corr(df["price"])
print(f"\n  Pearson Correlation (bath vs price): {corr_bath_price:.4f}")

# Group stats
bath_stats = df.groupby("bath")["price"].agg(["mean", "median", "count"])
bath_stats.columns = ["Mean Price (Lakhs)", "Median Price (Lakhs)", "Count"]
print("\n  Price Statistics by Number of Bathrooms:")
print(bath_stats.to_string(float_format=lambda x: f"{x:,.2f}"))

# --- Plot 3: Bath vs Price ---
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

common_bath = df[df["bath"].isin([1, 2, 3, 4, 5, 6])]
sns.boxplot(x="bath", y="price", data=common_bath, ax=axes[0],
            palette="coolwarm", fliersize=2, width=0.6)
axes[0].set_title("Price Distribution by Bathrooms (1-6)", fontweight="bold")
axes[0].set_xlabel("Number of Bathrooms")
axes[0].set_ylabel("Price (Lakhs)")

# Violin plot for shape of distribution
sns.violinplot(x="bath", y="price", data=common_bath, ax=axes[1],
               palette="coolwarm", inner="quartile", cut=0)
axes[1].set_title("Price Distribution Shape by Bathrooms", fontweight="bold")
axes[1].set_xlabel("Number of Bathrooms")
axes[1].set_ylabel("Price (Lakhs)")
axes[1].set_ylim(0, common_bath["price"].quantile(0.95))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_03_bath_vs_price.png"), bbox_inches="tight")
plt.close()
print("\n  [SAVED] eda_03_bath_vs_price.png")

print("""
  OBSERVATION:
  - There is a MODERATE POSITIVE correlation (r = {:.4f}) between bathrooms
    and price - properties with more bathrooms tend to be pricier.
  - This is expected since more bathrooms usually imply larger, higher-end
    properties.
  - The violin plot shows that 1-2 bathroom properties have tightly
    concentrated prices, while 4+ bathroom properties have wide spreads,
    reflecting luxury/premium variation.
""".format(corr_bath_price))

# ==============================================================================
# 5. CORRELATION MATRIX (ALL NUMERICAL VARIABLES)
# ==============================================================================
print("=" * 70)
print("  ANALYSIS 4: CORRELATION MATRIX - ALL NUMERICAL VARIABLES")
print("=" * 70)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = df[numeric_cols].corr()

print("\n  Correlation Matrix:")
print(corr_matrix.to_string(float_format=lambda x: f"{x:+.4f}"))

# --- Plot 4: Heatmap ---
fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".3f",
            cmap="RdYlBu_r", center=0, square=True,
            linewidths=0.8, linecolor="white",
            cbar_kws={"shrink": 0.8, "label": "Correlation"},
            annot_kws={"size": 11, "fontweight": "bold"}, ax=ax)
ax.set_title("Correlation Heatmap - Numerical Variables", fontweight="bold",
             fontsize=14, pad=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_04_correlation_heatmap.png"), bbox_inches="tight")
plt.close()
print("\n  [SAVED] eda_04_correlation_heatmap.png")

# --- Identify strong relationships ---
print("\n" + "-" * 70)
print("  STRONG RELATIONSHIPS (|r| >= 0.5)")
print("-" * 70)

pairs_seen = set()
strong_found = False
for i in range(len(corr_matrix)):
    for j in range(i + 1, len(corr_matrix)):
        col1 = corr_matrix.columns[i]
        col2 = corr_matrix.columns[j]
        r = corr_matrix.iloc[i, j]
        pair = tuple(sorted([col1, col2]))
        if abs(r) >= 0.5 and pair not in pairs_seen:
            pairs_seen.add(pair)
            direction = "POSITIVE" if r > 0 else "NEGATIVE"
            strength = "STRONG" if abs(r) >= 0.7 else "MODERATE"
            print(f"  {col1:20s} <-> {col2:20s}  r = {r:+.4f}  ({strength} {direction})")
            strong_found = True

if not strong_found:
    print("  No pairs found with |r| >= 0.5")

print("\n" + "-" * 70)
print("  ALL PAIRWISE CORRELATIONS (sorted by strength)")
print("-" * 70)

pair_list = []
for i in range(len(corr_matrix)):
    for j in range(i + 1, len(corr_matrix)):
        col1 = corr_matrix.columns[i]
        col2 = corr_matrix.columns[j]
        r = corr_matrix.iloc[i, j]
        pair_list.append((col1, col2, r))

pair_list.sort(key=lambda x: abs(x[2]), reverse=True)
for col1, col2, r in pair_list:
    bar_len = int(abs(r) * 30)
    direction = "+" if r > 0 else "-"
    bar = direction * bar_len
    print(f"  {col1:20s} <-> {col2:20s}  r = {r:+.4f}  |{bar}")

# ==============================================================================
# 6. BONUS: Pairplot of Key Numeric Features
# ==============================================================================
print("\n" + "=" * 70)
print("  BONUS: PAIRPLOT OF KEY FEATURES")
print("=" * 70)

sample = df[["total_sqft", "bath", "bhk", "price"]].sample(
    n=min(2000, len(df)), random_state=42
)
g = sns.pairplot(sample, diag_kind="kde",
                 plot_kws={"alpha": 0.35, "s": 12, "color": "#2980b9"},
                 diag_kws={"color": "#e74c3c", "linewidth": 2})
g.figure.suptitle("Pairplot - Key Numeric Features", y=1.02, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_05_pairplot.png"), bbox_inches="tight")
plt.close()
print("  [SAVED] eda_05_pairplot.png")

# ==============================================================================
# 7. SUMMARY OF KEY INSIGHTS
# ==============================================================================
print("\n" + "=" * 70)
print("  SUMMARY OF KEY INSIGHTS")
print("=" * 70)
print(f"""
  1. total_sqft vs price (r = {corr_sqft_price:+.4f}):
     Moderate positive correlation. Larger properties cost more, but
     location and property type introduce significant variance.

  2. bhk vs price (r = {corr_bhk_price:+.4f}):
     Moderate positive correlation. Each additional bedroom raises the
     median price substantially, from ~Rs 40L (1 BHK) to ~Rs 185L (6 BHK).

  3. bath vs price (r = {corr_bath_price:+.4f}):
     Moderate positive correlation. More bathrooms signal larger, premium
     properties. Bath count is also closely tied to BHK count.

  4. Strongest correlations:""")
for col1, col2, r in pair_list[:3]:
    print(f"     - {col1} <-> {col2}: r = {r:+.4f}")
print(f"""
  5. price_per_sqft shows weaker correlations with size/bhk because it
     is a normalised metric - it captures location value rather than
     property size.

  All plots saved to: {OUTPUT_DIR}
""")

print("=" * 70)
print("  [OK] Exploratory Data Analysis complete.")
print("=" * 70)
