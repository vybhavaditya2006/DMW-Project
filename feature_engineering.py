"""
Feature Engineering - Price Per Square Foot
============================================
Loads and cleans the Bengaluru house price dataset, then creates a new
feature 'price_per_sqft' using:

    price_per_sqft = (price * 100000) / total_sqft

Displays:
  - Updated dataset with the new column (first 10 rows)
  - Summary statistics of 'price_per_sqft'
  - Explanation of why this feature is useful

Usage:
    python feature_engineering.py
"""

import os
import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# 1. LOAD & CLEAN DATASET  (same steps as data_cleaning.py)
# ------------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
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

# Drop rows with missing values
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"[OK] Cleaning complete. Shape: {df.shape}")

# ------------------------------------------------------------------------------
# 2. CREATE 'price_per_sqft' FEATURE
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  FEATURE ENGINEERING: price_per_sqft")
print("=" * 70)

# price is in lakhs (1 lakh = 100,000 rupees)
# total_sqft is in square feet
# => price_per_sqft gives the price in rupees per square foot
df["price_per_sqft"] = (df["price"] * 100000) / df["total_sqft"]

print("\n  Formula: price_per_sqft = (price * 100000) / total_sqft")
print(f"  [+] Created 'price_per_sqft' column successfully")
print(f"  Updated columns: {list(df.columns)}")
print(f"  Updated shape  : {df.shape}")

# ------------------------------------------------------------------------------
# 3. DISPLAY UPDATED DATASET
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  UPDATED DATASET (First 10 Rows)")
print("=" * 70)
print()
print(df.head(10).to_string())

# ------------------------------------------------------------------------------
# 4. SUMMARY STATISTICS OF price_per_sqft
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  SUMMARY STATISTICS: price_per_sqft (Rs per sq.ft)")
print("=" * 70)

stats = df["price_per_sqft"].describe()
print(f"""
  Count  : {stats['count']:,.0f}
  Mean   : Rs {stats['mean']:,.2f} per sq.ft
  Std    : Rs {stats['std']:,.2f}
  Min    : Rs {stats['min']:,.2f} per sq.ft
  25%    : Rs {stats['25%']:,.2f} per sq.ft
  50%    : Rs {stats['50%']:,.2f} per sq.ft  (Median)
  75%    : Rs {stats['75%']:,.2f} per sq.ft
  Max    : Rs {stats['max']:,.2f} per sq.ft
""")

# Also show a few location-level averages for context
print("-" * 70)
print("  AVERAGE price_per_sqft BY TOP 10 LOCATIONS (by property count)")
print("-" * 70)
top_locations = df["location"].value_counts().head(10).index
loc_avg = (df[df["location"].isin(top_locations)]
           .groupby("location")["price_per_sqft"]
           .agg(["mean", "median", "count"])
           .sort_values("mean", ascending=False))
loc_avg.columns = ["Mean (Rs/sqft)", "Median (Rs/sqft)", "Properties"]
print()
print(loc_avg.to_string(float_format=lambda x: f"{x:,.2f}"))

# ------------------------------------------------------------------------------
# 5. WHY IS price_per_sqft USEFUL?
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  WHY IS 'price_per_sqft' USEFUL IN REAL ESTATE ANALYSIS?")
print("=" * 70)
print("""
  1. NORMALISED COMPARISON ACROSS PROPERTIES
     Properties vary widely in size (500 sqft studio vs 5000 sqft villa).
     Raw price alone is misleading - a Rs 50 lakh flat may be expensive
     for 400 sqft but cheap for 2000 sqft.  Price per sq.ft levels the
     playing field so every property can be compared on a common scale.

  2. LOCATION BENCHMARKING
     By averaging price_per_sqft within each locality, analysts can rank
     neighbourhoods from premium to affordable.  This helps buyers find
     value-for-money areas and helps sellers price competitively.

  3. OUTLIER DETECTION
     Unusually high or low price_per_sqft values often indicate data
     entry errors, mispriced listings, or genuinely exceptional deals.
     Filtering on this metric is a standard data-cleaning technique.

  4. TREND ANALYSIS OVER TIME
     When tracked across months or years, price_per_sqft reveals market
     trends (appreciation / depreciation) independent of the mix of
     property sizes being sold in each period.

  5. MODEL FEATURE & TARGET ENGINEERING
     It can serve as an additional feature for ML models, or as an
     alternative prediction target that generalises better across
     differently-sized properties.
""")

print("=" * 70)
print("  [OK] Feature engineering complete.")
print("=" * 70)
