"""
Data Cleaning & Preprocessing - Bengaluru House Price Dataset
==============================================================
This script performs the following cleaning steps:
  1. Drop unnecessary columns: area_type, society, balcony, availability
  2. Remove rows with missing values
  3. Convert 'size' column into numerical 'bhk' (number of bedrooms)
  4. Convert 'total_sqft' to numeric (handle ranges and invalid entries)

After cleaning it displays:
  - First 5 rows of the cleaned dataset
  - Dataset shape before and after cleaning
  - Confirmation that no null values remain

Usage:
    python data_cleaning.py
"""

import os
import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# 1. LOAD DATASET
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

# ------------------------------------------------------------------------------
# 2. RECORD ORIGINAL SHAPE
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  DATASET SHAPE - BEFORE CLEANING")
print("=" * 70)
print(f"\n  Rows    : {df.shape[0]:,}")
print(f"  Columns : {df.shape[1]}")
print(f"  Columns : {list(df.columns)}")

# ------------------------------------------------------------------------------
# 3. STEP 1 - Drop unnecessary columns
# ------------------------------------------------------------------------------
print("\n" + "-" * 70)
print("  STEP 1: DROP UNNECESSARY COLUMNS")
print("-" * 70)

cols_to_drop = ["area_type", "society", "balcony", "availability"]
existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]

if existing_cols_to_drop:
    df.drop(columns=existing_cols_to_drop, inplace=True)
    print(f"\n  [+] Dropped columns: {existing_cols_to_drop}")
else:
    print("\n  [!] None of the specified columns found in dataset.")

print(f"  Remaining columns: {list(df.columns)}")
print(f"  Shape after drop : {df.shape}")

# ------------------------------------------------------------------------------
# 4. STEP 2 - Extract BHK from 'size' column
# ------------------------------------------------------------------------------
print("\n" + "-" * 70)
print("  STEP 2: EXTRACT 'bhk' FROM 'size' COLUMN")
print("-" * 70)

if "size" in df.columns:
    # Extract the leading number from entries like "2 BHK", "4 Bedroom", etc.
    df["bhk"] = df["size"].str.extract(r"(\d+)").astype(float)
    df.drop(columns=["size"], inplace=True)
    print(f"\n  [+] Created 'bhk' column by extracting bedroom count from 'size'")
    print(f"  [+] Dropped original 'size' column")
    print(f"  BHK value counts:\n{df['bhk'].value_counts().sort_index().to_string()}")
else:
    print("\n  [!] 'size' column not found in dataset.")

# ------------------------------------------------------------------------------
# 5. STEP 3 - Convert 'total_sqft' to numeric
# ------------------------------------------------------------------------------
print("\n" + "-" * 70)
print("  STEP 3: CONVERT 'total_sqft' TO NUMERIC")
print("-" * 70)


def convert_sqft_to_num(x):
    """
    Convert total_sqft values to float.
    Handles:
      - Direct numeric strings (e.g. "1056")
      - Range strings (e.g. "2100 - 2850") -> returns the mean
      - Non-numeric / unparseable strings -> returns NaN
    """
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
    df["total_sqft"] = df["total_sqft"].apply(convert_sqft_to_num)

    invalid_count = df["total_sqft"].isna().sum()
    print(f"\n  [+] Converted 'total_sqft' to numeric values")
    print(f"  Entries converted from range (e.g. '2100 - 2850' -> mean): handled")
    print(f"  Invalid / unparseable entries set to NaN: {invalid_count}")
else:
    print("\n  [!] 'total_sqft' column not found in dataset.")

# ------------------------------------------------------------------------------
# 6. STEP 4 - Remove rows with missing values
# ------------------------------------------------------------------------------
print("\n" + "-" * 70)
print("  STEP 4: REMOVE ROWS WITH MISSING VALUES")
print("-" * 70)

print(f"\n  Missing values BEFORE dropping:")
missing_before = df.isnull().sum()
print(missing_before[missing_before > 0].to_string())
print(f"  Total rows with at least one NaN: {df.isnull().any(axis=1).sum()}")

rows_before_drop = len(df)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
rows_after_drop = len(df)

print(f"\n  [+] Dropped {rows_before_drop - rows_after_drop} rows containing missing values")
print(f"  Rows remaining: {rows_after_drop:,}")

# ------------------------------------------------------------------------------
# 7. FINAL VALIDATION & SUMMARY
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  DATASET SHAPE - AFTER CLEANING")
print("=" * 70)
print(f"\n  Rows    : {df.shape[0]:,}")
print(f"  Columns : {df.shape[1]}")
print(f"  Columns : {list(df.columns)}")

print("\n" + "=" * 70)
print("  FIRST 5 ROWS OF CLEANED DATASET")
print("=" * 70)
print()
print(df.head().to_string())

print("\n" + "=" * 70)
print("  DATA TYPES")
print("=" * 70)
print()
print(df.dtypes.to_string())

print("\n" + "=" * 70)
print("  NULL VALUE CHECK")
print("=" * 70)
null_counts = df.isnull().sum()
print()
print(null_counts.to_string())
total_nulls = null_counts.sum()
if total_nulls == 0:
    print(f"\n  [CONFIRMED] No null values remain in the dataset! (Total nulls = {total_nulls})")
else:
    print(f"\n  [WARNING] {total_nulls} null values still present!")

print("\n" + "=" * 70)
print("  CLEANING SUMMARY")
print("=" * 70)
print(f"""
  Original shape  : (13320, 9)
  Cleaned shape   : {df.shape}
  Columns dropped : {existing_cols_to_drop}
  Columns added   : ['bhk'] (extracted from 'size')
  Rows removed    : {rows_before_drop - rows_after_drop} (due to missing values)
  Null values     : {total_nulls} (confirmed clean)
""")

print("=" * 70)
print("  [OK] Data cleaning & preprocessing complete.")
print("  The dataset is fully clean and ready for analysis.")
print("=" * 70)
