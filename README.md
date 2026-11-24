# Bengaluru Real Estate Price Prediction

Data Mining & Data Wrangling project: Real Estate and Property Market Analytics (Bengaluru).

| Team member | SRN |
|---|---|
| Vitala SS | 24SUUBECS2451 |
| Vishwas SV | 24SUUBECS2448 |
| Vybhav Aditya | 24SUUBECS2455 |
| Yadunandan R | 24SUUBECS2458 |
| Vishnu Narasimha Murthy | 24SUUBECS2441 |
| Yadu R | 24SUUBECS2457 |

## Versions

- **v1.0**: the baseline from our Data Analysis & Visualization project. It has basic cleaning, EDA plots, a linear-regression price model and a Streamlit dashboard (`app.py`).
- **v2.0 (this release)**: a logged data-wrangling pipeline plus data-mining methods. These are model comparison with cross-validation, clustering, association rules and feature importance. One command runs the pipeline and one dashboard shows the results (`app_v2.py`). The v1 files are kept unchanged so the two versions can be compared.

Dataset: `Bengaluru_House_Data.csv` (Kaggle, 13,320 listings × 9 columns, price in lakhs of rupees). `bengaluru_house_prices.csv` is a byte-identical copy, kept because v1 scripts fall back to it.

## v2 structure

```
run_pipeline.py        runs every stage below in order, from the raw CSV
app_v2.py              Streamlit dashboard; reads saved results, never retrains
src/
  profiling.py         raw-data audit: ydata-profiling report, missingno plots
  wrangling.py         8 logged cleaning steps -> outputs/clean_data.csv
  modeling.py          7 regressors x 2 targets, 5-fold CV, GridSearchCV, v1 comparison
  clustering.py        location segments: K-Means, DBSCAN, Agglomerative (Ward), PCA
  association.py       Apriori vs FP-Growth, association rules
  features.py          XGBoost gain vs permutation importance, SelectKBest, RFE
  visualize.py         every figure -> outputs/plots_v2/
outputs/
  clean_data.csv, outlier_flags.csv, test_predictions.csv, best_model.joblib   (generated, git-ignored)
  plots_v2/            14 figures
reports/
  wrangling_report.md, modeling_report.md, clustering_report.md,
  association_report.md, features_report.md     one write-up per stage
  *.csv                every table behind those reports
  v2_summary.md        results summary for the project report
  demo_script.md       5-7 minute dashboard walkthrough
  raw_profile.html, pipeline.log                (generated, git-ignored)
```

v1 files: `data_exploration.py`, `data_cleaning.py`, `feature_engineering.py`, `eda_analysis.py`, `visualizations.py`, `linear_regression.py`, `evaluate_model.py`, `demo_prediction.py`, `bengaluru_house_analysis.py`, `app.py`, `generate_sample_data.py` and `output_plots/`.

## How to run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python run_pipeline.py --fresh     # whole v2 pipeline from the raw CSV (about 20 minutes)
streamlit run app_v2.py            # v2 dashboard
```

- `--fresh` deletes the generated outputs and reports before running, so everything is rebuilt from the CSV.
- `--skip-profile` skips the slow ydata-profiling HTML report.
- Each stage can also run on its own, for example `python -m src.clustering`. Every stage reads the files the earlier stages saved.
- The whole run logs to the console and to `reports/pipeline.log`.
- On Windows, set `PYTHONIOENCODING=utf-8` if the console shows encoding errors.
- Everything uses `random_state=42`, so re-runs give the same numbers.

v1 still runs as before: `python bengaluru_house_analysis.py` and `streamlit run app.py`.

## Methods

| Stage | Methods | Tools |
|---|---|---|
| Profiling | Type, missing-value, duplicate and skew audit; nullity matrix and nullity-correlation heatmap | ydata-profiling, missingno, pandas |
| Wrangling | Text normalisation; de-duplication; regex parsing of `size`; range-to-midpoint and unit conversion (sq m, sq yd, acres, cents, guntha, perch, grounds); availability parsing; group-median vs KNN imputation (masked evaluation); target transform; 9 outlier methods compared (domain rules, z-score, IQR, Tukey far-out fence on log price/sq ft, per-location mean ± std, BHK inversion, Isolation Forest, LOF); rare-category grouping; quantile binning. Every step is logged. | pandas, NumPy, scikit-learn (KNNImputer, IsolationForest, LocalOutlierFactor), SciPy |
| Price prediction | Pipeline + ColumnTransformer (one-hot, scaling); Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, XGBoost; raw vs log-log target; 5-fold CV (R², MAE, RMSE); GridSearchCV on the top 2; 80/20 hold-out test; step-by-step comparison with v1 | scikit-learn, XGBoost, joblib |
| Market segments | Location-level feature table, log + standard scaling; K-Means (elbow, silhouette, Davies-Bouldin to choose k); DBSCAN (k-distance knee for eps); Agglomerative Ward clustering with a dendrogram; PCA for 2-D plots; segments named from their profiles | scikit-learn, SciPy |
| Association rules | Listings turned into item baskets; Apriori vs FP-Growth (same itemsets, timed); support, confidence, lift, leverage, conviction; redundant-rule pruning; diverse top-25 selection by Jaccard overlap | mlxtend |
| Feature importance | XGBoost gain vs permutation importance on the test set; SelectKBest (f_regression) and RFE inside the CV pipeline | scikit-learn, XGBoost |
| Visualisation & demo | 14 figures; 6-tab dashboard with a live predictor that uses the saved model | Matplotlib, Streamlit |

## Results summary

These numbers come from `python run_pipeline.py --fresh`. See `reports/v2_summary.md` for the full write-up.

**Wrangling: 13,320 raw rows → 11,992 clean rows** (1,328 removed, 10.0%), in 8 logged steps:
- 530 exact duplicates removed, after normalising text (1 more duplicate than in the raw file).
- 16 bare land plots with no size dropped.
- All 238 non-numeric `total_sqft` values converted. v1 turned the 46 values in other units into NaN and dropped them.
- Missing `bath`/`balcony` filled with the median for that BHK. KNN was 1.9% less accurate on masked values.
- `society` (41.7% missing) dropped.
- 782 outliers removed (6.1%): 747 broke the domain rules and 35 more fell outside ₹841–₹37,830 per sq ft.
- 972 rare locations (10 or fewer listings) grouped into `other`, leaving 215.

**Price prediction: tuned XGBoost (log-log), test R² 0.639 and MAE ₹29.33 L, against v1's R² 0.438 and MAE ₹47.10 L.** The typical error falls by 38%. Each change was added one at a time on the same test rows:

| Step | Test R² | Test MAE (lakhs) |
|---|---|---|
| v1 as shipped (LinearRegression, 3 numeric features) | 0.438 | 47.10 |
| Same model on v2 clean data | 0.449 | 42.02 |
| + location and area type (one-hot) | 0.509 | 37.90 |
| + log-log target | 0.517 | 32.12 |
| Tuned XGBoost | **0.639** | **29.33** |

**Market segments: K-Means picked k = 4 for 214 locations.** The silhouette is 0.258 and Davies-Bouldin 1.25, the lowest Davies-Bouldin of any k.

| Segment | Locations | Mean ₹/sq ft | Examples |
|---|---|---|---|
| Premium (more bedrooms, large homes) | 42 | 9,071 | Rajaji Nagar, Koramangala, Malleshwaram |
| Upper mid-market (high volume) | 39 | 5,712 | Whitefield, Sarjapur Road, Electronic City |
| Mid-market | 100 | 5,487 | Hulimavu, Jigani, Budigere |
| Budget (under construction) | 33 | 5,217 | Electronic City Phase II, Hoodi, Chandapura |

DBSCAN found one dense core and 19 noise locations. Bengaluru locations form a continuum, so the segments are a summary of it rather than natural groups.

**Association rules: 617 rules kept, with lift ≥ 1.2 and confidence ≥ 0.5.** Apriori and FP-Growth found identical itemsets; mlxtend's Apriori was faster on this data. Top rules:
- Super built-up, 1 BHK, Low price band → 1 bathroom: confidence 96%, lift 15.7.
- 4+ BHK, under construction → 4+ bathrooms: confidence 97%, lift 6.8.
- Built-up area, 4+ bathrooms → 4+ BHK: confidence 90%, lift 6.7.

**Feature importance:**
- Permutation importance ranks `total_sqft` first by far: test R² falls by 0.76 when it is shuffled.
- XGBoost's gain gives 92% to `location`, because its 215 dummy columns offer many split points. Gain measures how much the trees use a feature; permutation measures how much the model needs it.
- Keeping 100 of the 222 columns (SelectKBest) gives CV R² 0.676, against 0.697 with all of them.

## Tools

Python 3.12 · pandas · NumPy · SciPy · scikit-learn · XGBoost · mlxtend · ydata-profiling · missingno · Matplotlib · joblib · Streamlit. v1 also uses Seaborn and Plotly.
