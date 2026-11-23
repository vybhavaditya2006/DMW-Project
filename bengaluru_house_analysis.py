"""
Bengaluru House Price Analysis & Prediction
============================================
This script performs end-to-end data analysis on the Bengaluru house price
dataset, including:
  1. Data Loading & Exploration
  2. Data Cleaning & Preprocessing
  3. Feature Engineering
  4. Exploratory Data Analysis (EDA) with Visualisations
  5. Linear Regression Modelling & Evaluation

Required dataset : Bengaluru_House_Data.csv  (or bengaluru_house_prices.csv)
Required packages: pandas, numpy, matplotlib, seaborn, scikit-learn

Usage:
    python bengaluru_house_analysis.py
"""

# ──────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")

# Plotting defaults
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "figure.dpi": 120,
    "font.size": 11,
})

# Create output directory for saved figures
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  BENGALURU HOUSE PRICE ANALYSIS & PREDICTION")
print("=" * 70)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_CANDIDATES = ["Bengaluru_House_Data.csv", "bengaluru_house_prices.csv"]

df = None
for name in CSV_CANDIDATES:
    path = os.path.join(SCRIPT_DIR, name)
    if os.path.isfile(path):
        df = pd.read_csv(path)
        print(f"\n✅ Loaded dataset: {name}")
        break

if df is None:
    raise FileNotFoundError(
        f"Dataset not found. Place one of {CSV_CANDIDATES} in {SCRIPT_DIR}"
    )

# ──────────────────────────────────────────────────────────────────────────────
# 2. INITIAL DATA EXPLORATION
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("  STEP 1: DATA EXPLORATION")
print("─" * 70)

print(f"\n📊 Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nColumn names:\n  {list(df.columns)}")

print("\n── First 5 rows ──")
print(df.head().to_string())

print("\n── Data types ──")
print(df.dtypes.to_string())

print("\n── Statistical summary (numeric columns) ──")
print(df.describe().to_string())

print("\n── Missing values ──")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Count": missing, "Percent": missing_pct})
print(missing_df[missing_df["Count"] > 0].to_string())

# ──────────────────────────────────────────────────────────────────────────────
# 3. DATA CLEANING & PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("  STEP 2: DATA CLEANING & PREPROCESSING")
print("─" * 70)

# 3a. Drop columns that won't help the model
cols_to_drop = [c for c in ["society", "availability"] if c in df.columns]
if cols_to_drop:
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"  • Dropped columns: {cols_to_drop}")

# 3b. Extract BHK from the 'size' column
if "size" in df.columns:
    df["bhk"] = df["size"].str.extract(r"(\d+)").astype(float)
    df.drop(columns=["size"], inplace=True)
    print("  • Extracted 'bhk' from 'size' column")

# 3c. Convert total_sqft to numeric (handle ranges like "1000 - 1500")
def convert_sqft(x):
    """Convert total_sqft to a single float value."""
    try:
        return float(x)
    except (ValueError, TypeError):
        if isinstance(x, str) and "-" in x:
            tokens = x.split("-")
            try:
                return (float(tokens[0].strip()) + float(tokens[1].strip())) / 2
            except (ValueError, IndexError):
                return np.nan
        return np.nan

if "total_sqft" in df.columns:
    df["total_sqft"] = df["total_sqft"].apply(convert_sqft)
    print("  • Converted 'total_sqft' to numeric (ranges → mean)")

# 3d. Drop rows with missing critical values
initial_len = len(df)
df.dropna(subset=["total_sqft", "bath", "price", "bhk"], inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"  • Dropped {initial_len - len(df)} rows with missing critical values")

# Ensure numeric types
for col in ["total_sqft", "bath", "price", "bhk"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.dropna(subset=["total_sqft", "bath", "price", "bhk"], inplace=True)
df.reset_index(drop=True, inplace=True)

# ──────────────────────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("  STEP 3: FEATURE ENGINEERING")
print("─" * 70)

# 4a. Price per square foot
df["price_per_sqft"] = (df["price"] * 100000) / df["total_sqft"]  # price is in lakhs
print("  • Created 'price_per_sqft' feature")

# 4b. Consolidate rare locations (< 10 occurrences) into 'Other'
if "location" in df.columns:
    location_counts = df["location"].value_counts()
    rare_locations = location_counts[location_counts < 10].index
    df["location"] = df["location"].apply(
        lambda x: "Other" if x in rare_locations else x
    )
    print(f"  • Consolidated {len(rare_locations)} rare locations into 'Other'")

# 4c. Remove outliers — sqft per BHK too low (< 300 sqft/BHK)
before = len(df)
df = df[df["total_sqft"] / df["bhk"] >= 300]
df.reset_index(drop=True, inplace=True)
print(f"  • Removed {before - len(df)} rows where sqft/BHK < 300 (outliers)")

# 4d. Remove price_per_sqft outliers (per-location mean ± 1 std)
def remove_pps_outliers(dataframe):
    """Remove price-per-sqft outliers within each location."""
    frames = []
    for _loc, sub in dataframe.groupby("location"):
        mean_ = sub["price_per_sqft"].mean()
        std_ = sub["price_per_sqft"].std()
        if pd.isna(std_) or std_ == 0:
            frames.append(sub)
            continue
        reduced = sub[(sub["price_per_sqft"] >= (mean_ - std_)) &
                      (sub["price_per_sqft"] <= (mean_ + std_))]
        frames.append(reduced)
    return pd.concat(frames, ignore_index=True)

before = len(df)
df = remove_pps_outliers(df)
print(f"  • Removed {before - len(df)} price-per-sqft outliers")

# 4e. Remove rows where bath > bhk + 2
before = len(df)
df = df[df["bath"] <= df["bhk"] + 2]
df.reset_index(drop=True, inplace=True)
print(f"  • Removed {before - len(df)} rows where bath > bhk + 2")

print(f"\n📊 Cleaned dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

# ──────────────────────────────────────────────────────────────────────────────
# 5. EXPLORATORY DATA ANALYSIS — VISUALISATIONS
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("  STEP 4: EXPLORATORY DATA ANALYSIS")
print("─" * 70)

# ── Figure 1: Distribution of House Prices ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df["price"], bins=50, color="#3498db", edgecolor="white", alpha=0.85)
axes[0].set_title("Distribution of House Prices (Lakhs)", fontweight="bold")
axes[0].set_xlabel("Price (Lakhs)")
axes[0].set_ylabel("Frequency")

axes[1].hist(np.log1p(df["price"]), bins=50, color="#e74c3c", edgecolor="white", alpha=0.85)
axes[1].set_title("Log-Transformed Price Distribution", fontweight="bold")
axes[1].set_xlabel("log(1 + Price)")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_price_distribution.png"), bbox_inches="tight")
plt.show()
print("  📈 Saved: 01_price_distribution.png")

# ── Figure 2: Price vs Total Sqft ──
plt.figure(figsize=(12, 6))
plt.scatter(df["total_sqft"], df["price"], alpha=0.3, s=10, c="#2ecc71")
plt.title("Price vs Total Square Feet", fontweight="bold")
plt.xlabel("Total Square Feet")
plt.ylabel("Price (Lakhs)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_price_vs_sqft.png"), bbox_inches="tight")
plt.show()
print("  📈 Saved: 02_price_vs_sqft.png")

# ── Figure 3: BHK Distribution & Price by BHK ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

bhk_counts = df["bhk"].value_counts().sort_index()
axes[0].bar(bhk_counts.index.astype(int), bhk_counts.values, color="#9b59b6", edgecolor="white")
axes[0].set_title("Number of Properties by BHK", fontweight="bold")
axes[0].set_xlabel("BHK")
axes[0].set_ylabel("Count")

sns.boxplot(x="bhk", y="price", data=df[df["bhk"] <= 6], ax=axes[1],
            palette="viridis", fliersize=2)
axes[1].set_title("Price Distribution by BHK (≤ 6 BHK)", fontweight="bold")
axes[1].set_xlabel("BHK")
axes[1].set_ylabel("Price (Lakhs)")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_bhk_analysis.png"), bbox_inches="tight")
plt.show()
print("  📈 Saved: 03_bhk_analysis.png")

# ── Figure 4: Bathrooms vs Price ──
plt.figure(figsize=(10, 5))
sns.boxplot(x="bath", y="price", data=df[df["bath"] <= 6],
            palette="coolwarm", fliersize=2)
plt.title("Price Distribution by Number of Bathrooms (≤ 6)", fontweight="bold")
plt.xlabel("Bathrooms")
plt.ylabel("Price (Lakhs)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_bath_vs_price.png"), bbox_inches="tight")
plt.show()
print("  📈 Saved: 04_bath_vs_price.png")

# ── Figure 5: Top 10 Locations by Number of Properties ──
plt.figure(figsize=(12, 5))
top_locations = df["location"].value_counts().head(10)
sns.barplot(x=top_locations.values, y=top_locations.index, palette="magma")
plt.title("Top 10 Locations by Property Count", fontweight="bold")
plt.xlabel("Number of Properties")
plt.ylabel("Location")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_top_locations.png"), bbox_inches="tight")
plt.show()
print("  📈 Saved: 05_top_locations.png")

# ── Figure 6: Average Price by Top 10 Locations ──
plt.figure(figsize=(12, 5))
avg_price_loc = df.groupby("location")["price"].mean().sort_values(ascending=False).head(10)
sns.barplot(x=avg_price_loc.values, y=avg_price_loc.index, palette="rocket")
plt.title("Top 10 Locations by Average House Price", fontweight="bold")
plt.xlabel("Average Price (Lakhs)")
plt.ylabel("Location")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "06_avg_price_locations.png"), bbox_inches="tight")
plt.show()
print("  📈 Saved: 06_avg_price_locations.png")

# ── Figure 7: Area Type Distribution ──
if "area_type" in df.columns:
    plt.figure(figsize=(8, 5))
    area_counts = df["area_type"].value_counts()
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f1c40f", "#9b59b6"]
    plt.pie(area_counts.values, labels=area_counts.index, autopct="%1.1f%%",
            colors=colors[:len(area_counts)], startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    plt.title("Distribution by Area Type", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_area_type_distribution.png"), bbox_inches="tight")
    plt.show()
    print("  📈 Saved: 07_area_type_distribution.png")

# ── Figure 8: Correlation Heatmap ──
plt.figure(figsize=(8, 6))
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, cmap="RdYlBu_r", center=0,
            fmt=".2f", square=True, linewidths=0.5,
            cbar_kws={"shrink": 0.8})
plt.title("Correlation Heatmap (Numeric Features)", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "08_correlation_heatmap.png"), bbox_inches="tight")
plt.show()
print("  📈 Saved: 08_correlation_heatmap.png")

# ── Figure 9: Pairplot of Key Features ──
print("  ⏳ Generating pairplot (this may take a moment)...")
sample_df = df[["total_sqft", "bath", "bhk", "price"]].sample(
    n=min(2000, len(df)), random_state=42
)
g = sns.pairplot(sample_df, diag_kind="kde",
                 plot_kws={"alpha": 0.4, "s": 15, "color": "#3498db"},
                 diag_kws={"color": "#e74c3c"})
g.figure.suptitle("Pairplot — Key Numeric Features", y=1.02, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "09_pairplot.png"), bbox_inches="tight")
plt.show()
print("  📈 Saved: 09_pairplot.png")

# ──────────────────────────────────────────────────────────────────────────────
# 6. PREPARE DATA FOR MODELLING
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("  STEP 5: LINEAR REGRESSION MODELLING")
print("─" * 70)

# Encode categorical features
model_df = df.copy()
label_encoders = {}

for col in model_df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    model_df[col] = le.fit_transform(model_df[col].astype(str))
    label_encoders[col] = le
    print(f"  • Label-encoded '{col}' ({len(le.classes_)} unique values)")

# Drop price_per_sqft (derived from target — would cause leakage)
if "price_per_sqft" in model_df.columns:
    model_df.drop(columns=["price_per_sqft"], inplace=True)
    print("  • Dropped 'price_per_sqft' (derived from target)")

# Define features (X) and target (y)
X = model_df.drop(columns=["price"])
y = model_df["price"]

print(f"\n  Features used: {list(X.columns)}")
print(f"  Target: price")
print(f"  Samples: {len(X)}")

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  Training set: {len(X_train)} samples")
print(f"  Test set:     {len(X_test)} samples")

# ──────────────────────────────────────────────────────────────────────────────
# 7. TRAIN & EVALUATE LINEAR REGRESSION
# ──────────────────────────────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Metrics
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)

print("\n  ┌──────────────────────────────────────┐")
print("  │     LINEAR REGRESSION RESULTS        │")
print("  ├──────────────────────────────────────┤")
print(f"  │  Train R² Score : {train_r2:>12.4f}       │")
print(f"  │  Test  R² Score : {test_r2:>12.4f}       │")
print(f"  │  MAE            : {mae:>12.2f} lakhs │")
print(f"  │  RMSE           : {rmse:>12.2f} lakhs │")
print("  └──────────────────────────────────────┘")

# Feature coefficients
coeff_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)
print("\n  Feature Coefficients:")
print(coeff_df.to_string(index=False))

# ── Figure 10: Actual vs Predicted Prices ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(y_test, y_pred_test, alpha=0.4, s=15, c="#3498db", edgecolors="none")
max_val = max(y_test.max(), y_pred_test.max())
axes[0].plot([0, max_val], [0, max_val], "r--", lw=1.5, label="Perfect Prediction")
axes[0].set_title("Actual vs Predicted Price", fontweight="bold")
axes[0].set_xlabel("Actual Price (Lakhs)")
axes[0].set_ylabel("Predicted Price (Lakhs)")
axes[0].legend()

residuals = y_test - y_pred_test
axes[1].hist(residuals, bins=50, color="#e74c3c", edgecolor="white", alpha=0.85)
axes[1].axvline(x=0, color="black", linestyle="--", lw=1)
axes[1].set_title("Residual Distribution", fontweight="bold")
axes[1].set_xlabel("Residual (Actual − Predicted)")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "10_model_evaluation.png"), bbox_inches="tight")
plt.show()
print("  📈 Saved: 10_model_evaluation.png")

# ── Figure 11: Feature Importance (Coefficients) ──
plt.figure(figsize=(10, 5))
colors = ["#2ecc71" if c >= 0 else "#e74c3c" for c in coeff_df["Coefficient"]]
plt.barh(coeff_df["Feature"], coeff_df["Coefficient"], color=colors, edgecolor="white")
plt.title("Feature Coefficients (Linear Regression)", fontweight="bold")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.axvline(x=0, color="gray", linestyle="--", lw=0.8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "11_feature_coefficients.png"), bbox_inches="tight")
plt.show()
print("  📈 Saved: 11_feature_coefficients.png")

# ──────────────────────────────────────────────────────────────────────────────
# 8. PREDICTION EXAMPLES
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("  STEP 6: SAMPLE PREDICTIONS")
print("─" * 70)

sample_indices = X_test.index[:5]
for idx in sample_indices:
    actual = y.loc[idx]
    predicted = model.predict(X.loc[[idx]])[0]
    print(f"  Row {idx:>5d}  |  Actual: {actual:>8.2f} lakhs  |  Predicted: {predicted:>8.2f} lakhs  |  Diff: {abs(actual-predicted):>7.2f}")

# ──────────────────────────────────────────────────────────────────────────────
# DONE
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  ✅  ANALYSIS COMPLETE")
print(f"  📁  All plots saved in: {OUTPUT_DIR}")
print("=" * 70)
