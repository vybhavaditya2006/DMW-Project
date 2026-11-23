"""
Demo Prediction - Bengaluru House Prices
=========================================
Trains the linear regression model quickly and makes a single
prediction for demo purposes.

Usage:
    python demo_prediction.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------------------------
# 1. LOAD & TRAIN MODEL 
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

X_train = model_df[["total_sqft", "bath", "bhk"]]
y_train = model_df["price"]

model = LinearRegression()
model.fit(X_train, y_train)

# ------------------------------------------------------------------------------
# 2. RUN DEMO PREDICTION
# ------------------------------------------------------------------------------
# User specified inputs
sqft_input = 1200
bath_input = 2
bhk_input  = 2

input_data = pd.DataFrame({
    'total_sqft': [sqft_input],
    'bath': [bath_input],
    'bhk': [bhk_input]
})

predicted_price = model.predict(input_data)[0]

# ------------------------------------------------------------------------------
# 3. DISPLAY PRESENTATION
# ------------------------------------------------------------------------------
print("\n" + "*" * 60)
print("             PROPERTY PRICE PREDICTOR DEMO")
print("*" * 60)
print("\nINPUT DETAILS:")
print(f"  - Total Area       :  {sqft_input} sq. ft.")
print(f"  - Bathrooms        :  {bath_input}")
print(f"  - Bedrooms (BHK)   :  {bhk_input}")
print("\n------------------------------------------------------------")
print(f"PREDICTED HOUSE PRICE:  Rs. {predicted_price:.2f} Lakhs")
print("------------------------------------------------------------\n")
