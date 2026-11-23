"""
Data Exploration — Bengaluru House Price Dataset
=================================================
Loads 'Bengaluru_House_Data.csv' from the local directory and displays:
  • First 5 rows
  • Dataset info (data types & non-null counts)
  • Summary statistics of numerical columns
  • Total rows and columns
  • Missing values per column
  • Column names

Usage:
    python data_exploration.py
"""

import os
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATASET (strictly local file)
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "Bengaluru_House_Data.csv")

if not os.path.isfile(CSV_PATH):
    raise FileNotFoundError(
        f"Dataset not found at:\n  {CSV_PATH}\n"
        "Please ensure 'Bengaluru_House_Data.csv' is in the same folder as this script."
    )

df = pd.read_csv(CSV_PATH)

# ──────────────────────────────────────────────────────────────────────────────
# 2. DISPLAY RESULTS
# ──────────────────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    """Print a formatted section header."""
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


# ── 2a. Total rows and columns ───────────────────────────────────────────────
section("DATASET DIMENSIONS")
rows, cols = df.shape
print(f"\n  Total Rows    : {rows:,}")
print(f"  Total Columns : {cols}")

# ── 2b. Column names ─────────────────────────────────────────────────────────
section("COLUMN NAMES")
for i, col in enumerate(df.columns, start=1):
    print(f"  {i:>2}. {col}")

# ── 2c. First 5 rows ─────────────────────────────────────────────────────────
section("FIRST 5 ROWS")
print()
print(df.head().to_string(index=True))

# ── 2d. Complete dataset information ─────────────────────────────────────────
section("DATASET INFO (Data Types & Non-Null Counts)")
print()
# Build a custom info table for cleaner formatting than df.info()
info_records = []
for col in df.columns:
    non_null = df[col].notna().sum()
    info_records.append({
        "Column": col,
        "Non-Null Count": f"{non_null:,} of {rows:,}",
        "Dtype": str(df[col].dtype),
    })
info_df = pd.DataFrame(info_records)
print(info_df.to_string(index=False))
print(f"\n  Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

# ── 2e. Summary statistics (numerical columns) ──────────────────────────────
section("SUMMARY STATISTICS (Numerical Columns)")
print()
desc = df.describe().T  # transpose for readability
desc = desc.rename(columns={
    "count": "Count",
    "mean": "Mean",
    "std": "Std Dev",
    "min": "Min",
    "25%": "Q1 (25%)",
    "50%": "Median",
    "75%": "Q3 (75%)",
    "max": "Max",
})
# Format numbers to 2 decimal places
print(desc.to_string(float_format=lambda x: f"{x:,.2f}"))

# ── 2f. Missing values per column ────────────────────────────────────────────
section("MISSING VALUES PER COLUMN")
print()
missing_count = df.isnull().sum()
missing_pct = (missing_count / rows * 100).round(2)
missing_report = pd.DataFrame({
    "Missing Count": missing_count,
    "Missing %": missing_pct,
})
# Sort so worst columns are on top
missing_report = missing_report.sort_values("Missing Count", ascending=False)
print(missing_report.to_string())

total_missing = missing_count.sum()
total_cells = rows * cols
print(f"\n  Total missing cells : {total_missing:,} / {total_cells:,} "
      f"({total_missing / total_cells * 100:.2f}%)")

# ── Done ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  ✅  Data exploration complete.")
print("=" * 70 + "\n")
