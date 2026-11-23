# CLAUDE.md: Bengaluru Real Estate, Data Mining & Data Wrangling (v2)

Read this first. It describes the dataset, the data-quality problems measured in it, what v1 does, and the plan for v2.

## 1. Project in one paragraph

v1 (tag `v1.0`) is our Data Analysis & Visualisation project: basic cleaning, EDA plots, a linear-regression price model and a Streamlit app. v2 turns it into a **Data Mining & Data Wrangling** project: a logged, reproducible wrangling pipeline, model comparison with cross-validation, clustering, association rules and feature importance, all demo-able from one command and one dashboard. **Every step must log what it did and how many rows it touched**, because the team will be asked to explain *why*, not just *what*.

## 2. Rules

- **Do not modify v1 files.** These are: `app.py`, `bengaluru_house_analysis.py`, `data_cleaning.py`, `data_exploration.py`, `demo_prediction.py`, `eda_analysis.py`, `evaluate_model.py`, `feature_engineering.py`, `generate_sample_data.py`, `linear_regression.py`, `visualizations.py`, `output_plots/`. v2 code lives in `src/`, `run_pipeline.py` and `app_v2.py`.
- **Never invent numbers.** All metrics, counts and findings must come from running the code. (v1's `app.py` has made-up amenity/age multipliers and "yield projections"; v2 must not do this.)
- `random_state=42` everywhere, so runs are reproducible.
- Use `Bengaluru_House_Data.csv`. `bengaluru_house_prices.csv` is a byte-identical duplicate (checked with `cmp`), kept only because v1 scripts fall back to it.
- `generate_sample_data.py` writes a **synthetic** file over `Bengaluru_House_Data.csv`. Never run it; the CSV in the repo is the real Kaggle data.

## 3. Environment

- Windows, Python 3.12.10, virtualenv at `.venv/` (git-ignored).
- Install: `.venv\Scripts\python.exe -m pip install -r requirements.txt`
- ydata-profiling requires `pandas<3` and `matplotlib<=3.10`, so installing it downgraded pandas 3.0.2→2.3.3, numpy 2.4.4→2.3.5, scipy 1.17.1→1.16.3 and matplotlib 3.10.8→3.10.0. v1 scripts were re-run afterwards and still work. It also needs `setuptools<81` for `pkg_resources`.
- Installed versions (2026-11-23): pandas 2.3.3, numpy 2.3.5, scipy 1.16.3, scikit-learn 1.8.0, xgboost 3.4.1, mlxtend 0.23.4, ydata-profiling 4.18.4, missingno 0.5.2, joblib 1.5.3, streamlit 1.56.0.
- On Windows consoles set `PYTHONIOENCODING=utf-8` (v1 scripts print emoji/box characters).

## 4. Dataset: `Bengaluru_House_Data.csv`

13,320 rows × 9 columns, ~0.9 MB. Kaggle "Bengaluru House Price Data". One row per property listing.

| Column | Raw type | Meaning |
|---|---|---|
| `area_type` | text, 4 values | How area is measured: Super built-up (8,790), Built-up (2,418), Plot (2,025), Carpet (87) |
| `availability` | text, 81 values | `Ready To Move` (10,581), `Immediate Possession` (16), or a possession date like `18-Dec` (2,723) |
| `location` | text, 1,306 values | Locality name, free text |
| `size` | text | `"2 BHK"`, `"4 Bedroom"` or `"1 RK"` |
| `society` | text, 2,688 values | Housing society / project name (coded strings like `Coomee`) |
| `total_sqft` | **text** | Area in sq ft, but also ranges and other units (see below) |
| `bath` | float | Number of bathrooms |
| `balcony` | float | Number of balconies (0–3) |
| `price` | float | **Target.** Price in **lakhs** of rupees (1 lakh = ₹100,000) |

## 5. Data-quality problems found (measured 2026-11-23)

All counts are on the raw 13,320 rows.

**Missing values**
| Column | Missing | % |
|---|---|---|
| `society` | 5,502 | 41.3% |
| `balcony` | 609 | 4.6% |
| `bath` | 73 | 0.5% |
| `size` | 16 | 0.1% |
| `location` | 1 | <0.1% |

**Duplicates**
- 529 exact duplicate rows (909 rows belong to a duplicate group). 568 duplicates if `society` is ignored.

**`total_sqft` is stored as text** (247 non-numeric values)
- 201 ranges such as `2100 - 2850`.
- 46 values in other units: Sq. Meter 17, Sq. Yards 15, Acres 8, Cents 2, Guntha 2, Perch 1, Grounds 1. v1 turns all 46 into NaN and drops them; v2 should convert them (1 sq m = 10.7639 sq ft, 1 sq yd = 9, 1 acre = 43,560, 1 cent = 435.6, 1 guntha = 1,089, 1 perch = 272.25, 1 ground = 2,400).
- Impossible areas: 11 numeric values < 300 sq ft, 7 of them < 100 (e.g. a 4-bedroom at 1 sq ft, a 7 BHK at 5 sq ft).

**`size` is inconsistent text**
- Three suffixes: `BHK` 10,766, `Bedroom` 2,525, `RK` 13. `Bedroom` is used almost only for Plot Area (2,008 of 2,025 plot rows), so the suffix encodes property type.
- Extreme bedroom counts: 12 rows with > 10 BHK, max 43.

**Domain-rule violations**
- 744 rows with less than 300 sq ft per bedroom.
- 16 rows with more than bhk + 2 bathrooms; 20 rows with > 10 bathrooms (max 40).

**`location` is dirty free text**
- 95 values have leading/trailing spaces. Stripping + lower-casing + collapsing spaces reduces 1,306 → 1,282 unique values; 21 locations had 2+ spellings (e.g. `Anekal` / ` Anekal`, `Anand Nagar` / `Anand nagar`).
- Spelling variants that need fuzzy matching: `Kanakpura Road` (273), `Kanakapura` (42), `kanakapura road`, `Kanakapura  Rod`, `Kanakapur main road`, …; `Electronic City Phase II` vs `Electronics City Phase 1` vs `2Electronic City Phase II`. Some values have trailing commas.
- Long tail: 457 locations appear once; 1,041 locations have ≤ 10 listings (2,885 rows in total).

**Formatting noise**
- Every `area_type` value has a double space (`Super built-up  Area`).
- 1,144 `society` values have trailing spaces.
- `availability` mixes a status and a date; the `18-Dec` format has no explicit year/day meaning in the source.

**Target skew and outliers**
- `price`: min 8, median 72, mean 112.6, max 3,600 lakhs; skewness 8.06. 1,276 rows are IQR outliers. A log transform is needed for modelling.
- Price per sq ft (₹): median 5,434, but min 268 and max 12,000,000. 39 rows < ₹2,000/sq ft, 15 rows > ₹50,000/sq ft.

## 6. What v1 does (and its gaps)

The load + clean code is copy-pasted into each script. Steps: drop `area_type`, `society`, `balcony`, `availability`; `bhk` = first number in `size`; `total_sqft` ranges → mean, everything else → NaN; `dropna()`. Result: 13,200 rows (120 dropped).

**Measured v1 baseline** (`evaluate_model.py`: LinearRegression on `total_sqft`, `bath`, `bhk`, 80/20 split, `random_state=42`): **R² = 0.438, MAE = 47.10 lakhs.** v2 models are compared against this number.

v1 gaps that v2 fixes:
- No duplicate removal; no unit conversion; locations not normalised.
- Rows with missing values are dropped instead of imputed.
- `bengaluru_house_analysis.py` label-encodes `location` as an integer (an ordinal encoding with no meaning) and its outlier filters are not logged in a reusable way.
- Single train/test split, one model, no cross-validation, no log target.
- `app.py` retrains on every start and adds invented multipliers (amenities, property age, "yield 3.2–4.1%").

## 7. v2 plan

### Layout
```
src/
  __init__.py
  profiling.py      # raw-data audit
  wrangling.py      # logged cleaning steps
  features.py       # feature engineering + encodings + feature selection
  modeling.py       # model comparison with CV, tuning, importance
  clustering.py     # location segmentation
  association.py    # association-rule mining
  visualize.py      # all figure-making functions
run_pipeline.py     # runs everything in order, writes outputs/ and reports/
app_v2.py           # Streamlit dashboard that reads saved outputs (no training on start)
outputs/            # machine-readable artefacts: cleaned CSV, metrics CSV, model .joblib, figures/
reports/            # human-readable: profiling HTML, wrangling log, pipeline.log, summary
```

### Logging convention (the core of the project)
- `run_pipeline.py` sets up Python `logging` to the console and to `reports/pipeline.log`.
- Each wrangling step is a function that takes and returns a DataFrame and records one row in a **wrangling log**: `step`, `column`, `action`, `reason`, `rows_before`, `rows_after`, `rows_affected`. The log is saved to `reports/wrangling_log.csv` and shown in `app_v2.py`. This is the main thing to show in the viva.
- Every model/cluster/rule result is saved to `outputs/` with the parameters used.

### Module responsibilities
- **`profiling.py`**: ydata-profiling HTML report of the raw data (`reports/profile_raw.html`) and optionally of the cleaned data for a before/after comparison; missingno matrix and heatmap; a data-quality summary table reproducing §5.
- **`wrangling.py`**: strip/normalise text columns; normalise `location` (case, spaces, trailing commas) and map spelling variants with fuzzy matching (stdlib `difflib`, with a log of every mapping); parse `size` → `bhk` + `size_unit`; parse `total_sqft` (numbers, ranges → midpoint, unit conversion); split `availability` → `ready_to_move` (+ possession month); remove exact duplicates; impute `bath` and `balcony` (group median by `bhk` / `area_type`), keep `has_society` flag; apply domain rules (sqft/bhk ≥ 300, bath ≤ bhk + 2); statistical outliers on price per sq ft within each location (IQR); group rare locations (≤ 10 listings) into `other`.
- **`features.py`**: `price_per_sqft` (for analysis and clustering only; it is derived from the target, so it is **not** a model input); `log_price` target; one-hot for `area_type`; location target encoding fitted **inside** CV folds to avoid leakage; feature selection evidence with scipy (Spearman correlation, Kruskal–Wallis for price across `area_type`) and mutual information.
- **`modeling.py`**: compare LinearRegression (v1 baseline), Ridge, Lasso, DecisionTree, RandomForest, GradientBoosting and XGBoost with 5-fold `KFold` CV in sklearn `Pipeline`s; metrics R², MAE, RMSE (in lakhs, after back-transforming `log_price`); `GridSearchCV`/`RandomizedSearchCV` on the best one; final hold-out test; permutation importance; save best model with joblib to `outputs/model.joblib`.
- **`clustering.py`**: build a location-level table (median price per sq ft, median sqft, listing count, ready-to-move share, mean bhk); scale; KMeans with elbow + silhouette to choose k; PCA to 2-D for plotting; name the segments (e.g. premium / mid / budget) from the cluster centres. Optionally DBSCAN to flag anomalous listings.
- **`association.py`**: discretise listings into items (price band, bhk, sqft band, area type, ready/under construction, location segment from clustering); mlxtend Apriori + `association_rules`; report support, confidence and lift; save top rules.
- **`visualize.py`**: one function per figure, all saved to `outputs/figures/`; no `plt.show()`.
- **`run_pipeline.py`**: `profile → wrangle → features → model → cluster → associate → visualize`, with flags such as `--skip-profile` (profiling is the slowest step). Prints a final summary (rows in/out, best model vs v1 baseline).
- **`app_v2.py`**: tabs for Data Quality (raw profile + missingno), Wrangling Log (step-by-step table, before/after counts), Model Comparison (CV table, chosen model, importance), Location Segments (clusters), Association Rules, and Predict (uses `outputs/model.joblib`; only real model output, no invented multipliers).

### Demo flow
1. Show a raw-data problem (e.g. `total_sqft = "34.46Sq. Meter"` or the 43 BHK row).
2. `python run_pipeline.py` → the logged steps scroll past.
3. `app_v2.py`: wrangling log → model comparison vs v1 baseline (R² 0.438) → clusters → rules → a live prediction.
