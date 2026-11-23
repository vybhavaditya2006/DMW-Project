"""
Linear Regression Model - Bengaluru House Prices
================================================
Train a simple linear regression model to predict property 'price'.
Features: total_sqft, bath, bhk

Usage:
    python linear_regression.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------------------------
# 1. LOAD & CLEAN DATASET
# ------------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_CANDIDATES = ["Bengaluru_House_Data.csv", "bengaluru_house_prices.csv"]

df = None
for name in CSV_CANDIDATES:
    path = os.path.join(SCRIPT_DIR, name)
    if os.path.isfile(path):
        df = pd.read_csv(path)
        break

if df is None:
    raise FileNotFoundError("Dataset not found.")

def convert_sqft(x):
    try:
        return float(x)
    except:
        if isinstance(x, str) and "-" in x:
            parts = x.split("-")
            try:
                return (float(parts[0].strip()) + float(parts[1].strip())) / 2.0
            except:
                return np.nan
        return np.nan

if "size" in df.columns:
    df["bhk"] = df["size"].str.extract(r"(\d+)").astype(float)
if "total_sqft" in df.columns:
    df["total_sqft"] = df["total_sqft"].apply(convert_sqft)

# ------------------------------------------------------------------------------
# 2. SELECT FEATURES AND TARGET
# ------------------------------------------------------------------------------
print("Preparing dataset for modeling...")
# Select required columns
model_df = df[["total_sqft", "bath", "bhk", "price"]].copy()

# Drop missing values
model_df.dropna(inplace=True)
model_df.reset_index(drop=True, inplace=True)

# Features (X) and Target (y)
X = model_df[["total_sqft", "bath", "bhk"]]
y = model_df["price"]

print(f"Data prepared. Samples ready for training: {len(X)}")

# ------------------------------------------------------------------------------
# 3. SPLIT DATASET (80-20 Split)
# ------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Dataset split successfully: 80% Training ({len(X_train)} samples), 20% Testing ({len(X_test)} samples)")

# ------------------------------------------------------------------------------
# 4. TRAIN LINEAR REGRESSION MODEL
# ------------------------------------------------------------------------------
print("\nTraining Linear Regression Model...")
model = LinearRegression()
model.fit(X_train, y_train)

# ------------------------------------------------------------------------------
# 5. CONFIRMATION
# ------------------------------------------------------------------------------
print("\n=========================================================")
print(" [OK] Linear Regression Model Successfully Trained!")
print("=========================================================")
print("Model Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  - {feature:12s} : {coef:.4f}")
print(f"Model Intercept  : {model.intercept_:.4f}")
print("=========================================================\n")
