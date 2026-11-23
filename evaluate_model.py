"""
Evaluate Linear Regression Model - Bengaluru House Prices
=========================================================
Trains the linear regression model and runs evaluations on the test set.

Usage:
    python evaluate_model.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ------------------------------------------------------------------------------
# 1. LOAD, CLEAN, & PREPARE DATASET
# ------------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_CANDIDATES = ["Bengaluru_House_Data.csv", "bengaluru_house_prices.csv"]

df = None
for name in CSV_CANDIDATES:
    path = os.path.join(SCRIPT_DIR, name)
    if os.path.isfile(path):
        df = pd.read_csv(path)
        break

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

model_df = df[["total_sqft", "bath", "bhk", "price"]].copy()
model_df.dropna(inplace=True)
model_df.reset_index(drop=True, inplace=True)

X = model_df[["total_sqft", "bath", "bhk"]]
y = model_df["price"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# ------------------------------------------------------------------------------
# 2. PREDICT ON TEST DATA
# ------------------------------------------------------------------------------
# Generate predictions
y_pred = model.predict(X_test)

# ------------------------------------------------------------------------------
# 3. COMPUTE EVALUATION METRICS
# ------------------------------------------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=========================================================")
print(" MODEL EVALUATION INSIGHT")
print("=========================================================")
print(f"Average Error (MAE): Off by ~{mae:.2f} Lakhs per prediction")
print(f"Accuracy Score (R²): {r2*100:.1f}% of price differences explained")

if r2 < 0.6:
    print("\nInsight: The predictions are somewhat scattered from the actual prices.")
    print("This means total_sqft, bath, and bhk alone are NOT enough to reliably predict")
    print("the final price. Factors like 'Location' and 'Age of Property' likely play a huge role!")
else:
    print("\nInsight: The predictions track reasonably well with the actual prices.")

print("\n=========================================================")
print(" SAMPLE PREDICTIONS vs ACTUAL VALUES (Test Set Subset)")
print("=========================================================")
results = X_test.copy()
results['Actual Price (Lakhs)'] = y_test
results['Predicted Price (Lakhs)'] = np.round(y_pred, 2)
results['Difference'] = np.round(abs(y_test - y_pred), 2)

# Select 5 to 10 random samples to display
sample_display = results.sample(8, random_state=42)
print(sample_display.to_string())
print("=========================================================\n")
