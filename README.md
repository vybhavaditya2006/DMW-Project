# Bengaluru Real Estate Price Prediction

Data Mining & Data Wrangling project — Real Estate and Property Market Analytics (Bengaluru).

| Team member | SRN |
|---|---|
| Vitala SS | 24SUUBECS2451 |
| Vishwas SV | 24SUUBECS2448 |
| Vybhav Aditya | 24SUUBECS2455 |
| Yadunandan R | 24SUUBECS2458 |
| Vishnu Narasimha Murthy | 24SUUBECS2441 |
| Yadu R | 24SUUBECS2457 |

## Versions

- **v1.0 (this release)** — the baseline carried over from our Data Analysis & Visualization project: cleaning, EDA, plots, a linear-regression price model and a Streamlit dashboard.
- **v2.0 (in progress)** — a full data-wrangling pipeline and data-mining techniques (model comparison with cross-validation, clustering, association rules, feature importance).

## v1 contents

| File | Purpose |
|---|---|
| `Bengaluru_House_Data.csv` | Raw dataset (~13k listings) |
| `data_exploration.py` | First look at the data |
| `data_cleaning.py` | Basic cleaning (sqft conversion, drop missing) |
| `feature_engineering.py` | Price per sqft and location stats |
| `eda_analysis.py`, `visualizations.py` | EDA and charts (saved to `output_plots/`) |
| `linear_regression.py`, `evaluate_model.py`, `demo_prediction.py` | Linear regression model, evaluation, sample predictions |
| `bengaluru_house_analysis.py` | End-to-end analysis script |
| `app.py` | Streamlit dashboard |
| `generate_sample_data.py` | Synthetic sample data generator |

## Running

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python bengaluru_house_analysis.py
streamlit run app.py
```

## Tools

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly, Streamlit
