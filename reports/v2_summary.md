# v2 Summary: Bengaluru Real Estate, Data Mining & Data Wrangling

This summary is written to be pasted into the project report. Every number comes from `python run_pipeline.py --fresh` (`random_state=42`). The per-stage reports in this folder hold the full tables.

## 1. Objective

v1 (our Data Analysis & Visualisation project) cleaned the Bengaluru house-price data with a few ad-hoc filters and fitted one linear regression (test R² 0.438). v2 rebuilds it as a data-mining and data-wrangling project:

1. A reproducible wrangling pipeline in which every step logs what it changed and why.
2. A comparison of seven regression models with cross-validation and tuning, measured against v1.
3. Three unsupervised methods: clustering locations into market segments, association-rule mining and feature-importance analysis.
4. A dashboard that presents the saved results and a live price predictor.

## 2. Dataset

`Bengaluru_House_Data.csv` (Kaggle) has 13,320 property listings and 9 columns: `area_type`, `availability`, `location`, `size`, `society`, `total_sqft`, `bath`, `balcony` and `price` (in lakhs of rupees).

Data-quality problems found in the raw file:

| Problem | Count |
|---|---|
| Exact duplicate rows | 529 |
| Missing `society` / `balcony` / `bath` / `size` / `location` | 5,502 / 609 / 73 / 16 / 1 |
| `total_sqft` stored as text: ranges (`2100 - 2850`) | 201 |
| `total_sqft` in other units (sq m, sq yd, acres, cents, guntha, perch, grounds) | 46 |
| `size` written three ways (`BHK`, `Bedroom`, `RK`) | 10,766 / 2,525 / 13 |
| Fewer than 300 sq ft per bedroom | 744 |
| More than bhk + 2 bathrooms | 16 |
| `location` with stray spaces / one-listing locations | 95 / 457 |
| Double space inside every `area_type` value | 13,320 |
| Price skewness (heavy right tail, max 3,600 lakhs) | 8.06 |

## 3. Data wrangling

Each step is a function wrapped in a `@logged_step` decorator. The decorator records rows, columns and nulls before and after the step, along with the decision and its reason. The log is saved to `reports/wrangling_log.csv` and explained step by step in `reports/wrangling_report.md`.

| # | Step | Rows before → after | What was done and why |
|---|---|---|---|
| 1 | Normalise text, drop duplicates | 13,320 → 12,790 | Fixed spacing and case in `area_type` and `location` *before* de-duplicating, so rows that differ only by spaces count as duplicates: 530 removed. Duplicates over-weight listings and can leak between the train and test sets. |
| 2 | Parse `size` → `bhk` | 12,790 → 12,774 | Extracted the bedroom count with a regex. The 16 rows with no size are all bare land plots with no bath or balcony data either, so they were dropped. |
| 3 | Convert `total_sqft` | 12,774 → 12,774 | Ranges became their midpoint and other units were converted to sq ft: all 238 non-numeric values recovered. (v1 dropped the unit values as NaN.) |
| 4 | Parse `availability` | 12,774 → 12,774 | Split into `is_ready_to_move`, `possession_year` and `possession_month`. The `18-Dec` prefix only takes values 14–22, so it is a year, not a day. |
| 5 | Missing values | 12,774 → 12,774 | Dropped `society`: 41.7% missing, 2,676 values, most appearing once. Imputed `bath`/`balcony` with the median for each BHK. On masked known values its MAE was 0.423 against 0.431 for KNNImputer, so the simpler method won. |
| 6 | Price features | 12,774 → 12,774 | `price_per_sqft` is used for outliers and clustering only; it is never a model input, because that would leak the target. `log_price` reduces skewness from 7.96 to 0.82. |
| 7 | Outliers | 12,774 → 11,992 | Nine methods compared; 782 rows (6.1%) removed. The two domain rules (< 300 sq ft per bedroom, > bhk + 2 baths) caught 747 rows. Tukey's far-out fence (3 × IQR) on log price per sq ft, ₹841–₹37,830, caught 35 more, such as giant land parcels at ₹2/sq ft. The other methods were rejected: per-location ±1 std would remove 17.7% of rows, the city-wide IQR removes genuinely expensive localities, and IsolationForest/LOF need a contamination rate we would have to guess. |
| 8 | Rare locations, price bands | 11,992 → 11,992 | 972 of 1,186 locations have 10 or fewer listings; their 2,741 rows (22.9%) were grouped as `other`, leaving 215 locations. `price_band` holds the price quartiles. |

**Result:** 13,320 → 11,992 rows (1,328 removed, 10.0%), with no missing values in any model column.

## 4. Price prediction

**Setup:**
- **Pipeline:** a scikit-learn `Pipeline` with a `ColumnTransformer`, so preprocessing is learned inside each CV fold with no leakage. `location` and `area_type` are one-hot encoded; `total_sqft`, `bath` and `bhk` are standardised.
- **Data split:** an 80/20 hold-out split. 5-fold `KFold` CV runs on the 80% training part.
- **Models:** LinearRegression, Ridge, Lasso, DecisionTree, RandomForest, GradientBoosting and XGBoost.
- **Targets:** each model is trained once on price and once on a log-log specification (log price, log `total_sqft`), with predictions converted back to lakhs.
- **Tuning:** the top two models were tuned with `GridSearchCV` (24 settings each); the winner was scored once on the untouched test set.

**Cross-validation (top 6 of 14 model–target combinations):**

| Model | Target | CV R² (mean ± std) | CV MAE (lakhs) |
|---|---|---|---|
| RandomForest | log | 0.685 ± 0.027 | 31.17 |
| XGBoost | log | 0.684 ± 0.030 | 30.60 |
| GradientBoosting | price | 0.681 ± 0.058 | 33.97 |
| XGBoost | price | 0.671 ± 0.089 | 31.87 |
| RandomForest | price | 0.670 ± 0.056 | 31.66 |
| GradientBoosting | log | 0.649 ± 0.035 | 32.85 |

The linear models reach CV R² of about 0.56. The tree ensembles capture non-linear effects and interactions such as location × size that a linear model cannot.

**Tuning:** XGBoost with log target, `learning_rate=0.03, max_depth=8, n_estimators=800, subsample=1.0`, reached CV R² 0.697. Tuned RandomForest reached 0.685.

**v1 → v2, one change at a time (same held-out test rows from step 1 on):**

| Step | Test R² | Test MAE (lakhs) | Test RMSE (lakhs) |
|---|---|---|---|
| 0. v1 as shipped (LinearRegression on sqft, bath, bhk) | 0.438 | 47.10 | 107.66 |
| 1. Same model, v2 clean data | 0.449 | 42.02 | 101.45 |
| 2. + location and area_type (one-hot) | 0.509 | 37.90 | 95.80 |
| 3. + log-log specification | 0.517 | 32.12 | 95.04 |
| 4. Tuned XGBoost (final model) | **0.639** | **29.33** | **82.12** |

The final model explains 64% of price variance against 44% for v1, and its average error falls from ₹47.1 L to ₹29.3 L (−38%). Step 1 also removes impossible listings from the test set, so part of its gain comes from an easier test set; steps 2–4 use the same rows. The model is saved as `outputs/best_model.joblib` and used by the dashboard's predictor.

## 5. Market segmentation (clustering)

**Data:** one row per location: 214 locations with more than 10 listings. Five features were used: mean price per sq ft, median size, mean BHK, listing count and share ready to move. The skewed features were log-transformed, then all were standardised.

**Choosing k for K-Means:**
- Silhouette is highest at k = 2 (0.284). k values within 10% of that are 2, 3 and 4.
- Among those, k = 4 has the lowest Davies-Bouldin index (1.25, also the lowest for any k).
- k = 2 only separates "expensive vs not". The PCA plot shows that k = 2 misses the split between large under-construction corridors and small established localities.

**Methods compared:**

| Method | Settings | Clusters | Silhouette | Davies-Bouldin |
|---|---|---|---|---|
| K-Means | k = 4, n_init = 20 | 4 | **0.258** | **1.25** |
| Agglomerative (Ward) | k = 4 | 4 | 0.223 | 1.28 |
| DBSCAN | eps = 1.51 (k-distance knee), min_samples = 10 | 1 + 19 noise | n/a | n/a |

**Segments found (K-Means):**

| Segment | Locations | Listings | Mean ₹/sq ft | Median sq ft | Ready to move | Examples |
|---|---|---|---|---|---|---|
| Premium (more bedrooms, large homes) | 42 | 1,113 | 9,071 | 1,884 | 90% | Rajaji Nagar, Koramangala, Malleshwaram |
| Upper mid-market (high volume) | 39 | 4,666 | 5,712 | 1,347 | 77% | Whitefield, Sarjapur Road, Electronic City |
| Mid-market | 100 | 2,224 | 5,487 | 1,234 | 91% | Hulimavu, Jigani, Budigere |
| Budget (under construction) | 33 | 1,248 | 5,217 | 1,269 | 48% | Electronic City Phase II, Hoodi, Chandapura |

**Interpretation:**
- The silhouette values are modest: Bengaluru's locations form a continuum rather than well-separated groups. DBSCAN finding a single dense core agrees with that.
- The segments are therefore a useful summary of the market, not natural categories.
- The three non-premium segments are close in price; they differ mainly in listing volume and readiness.

## 6. Association-rule mining

**Baskets:** each listing became a "basket" of 27 possible items: price band, size band, BHK, bathrooms, area type, ready or under construction, and market segment.

**Apriori vs FP-Growth:**
- Both are exact algorithms and found identical frequent itemsets at every support level tested (1%–10%).
- On this data mlxtend's vectorised Apriori was much faster than its pure-Python FP-Growth, because there are few items and the baskets are dense. FP-Growth's advantage appears with many sparse items.

**Rule filtering:**
- From the 1,511 itemsets with support ≥ 2%, 3,737 rules have lift ≥ 1.2 and confidence ≥ 0.5.
- 1,915 of them are redundant: a rule with fewer conditions and the same conclusion is at least as confident. Removing them leaves 1,822.
- Keeping single-item conclusions and one rule per itemset (its most confident) leaves **617 rules**. All 617 can be filtered in the dashboard.
- Ranking purely by lift fills the top 25 with rewordings of one pattern (cheap 1-BHK flats with 1 bathroom). So the top 25 are picked greedily down the lift ranking, skipping any rule whose listings overlap more than 0.5 (Jaccard) with a rule already chosen.

**The five best rules in plain English:**

1. Of the listings that are Super built-up area, 1 BHK and in the Low price band (9–50 lakhs), **96%** have 1 bathroom, against 6% of all listings: **15.66× as likely**. The rule covers 324 listings (2.7%).
2. Of the listings that are Plot area with 4 or more bathrooms, in the Premium price band (120–2,912 lakhs), 1,710+ sq ft and ready to move, **95%** are 4 or more BHK, against 13% of all listings: **7.03× as likely**. The rule covers 339 listings (2.8%).
3. Of the listings that are 4 or more BHK and under construction, **97%** have 4 or more bathrooms, against 14% of all listings: **6.82× as likely**. The rule covers 246 listings (2.1%).
4. Of the listings that are Built-up area with 4 or more bathrooms, **90%** are 4 or more BHK, against 13% of all listings: **6.71× as likely**. The rule covers 350 listings (2.9%).
5. Of the listings with 4 or more bathrooms, in a rare location and 1,710+ sq ft, **89%** are 4 or more BHK, against 13% of all listings: **6.60× as likely**. The rule covers 356 listings (3.0%).

Rules describe co-occurrence, not cause and effect.

## 7. Feature importance and selection

| Feature | XGBoost gain share | Permutation: test R² drop | Permutation: MAE increase (lakhs) |
|---|---|---|---|
| total_sqft | 0.054 | **0.756** | 43.08 |
| bath | 0.004 | 0.088 | 3.39 |
| area_type | 0.018 | 0.080 | 4.08 |
| location | **0.915** | 0.080 | 6.78 |
| bhk | 0.008 | −0.014 | 1.68 |

**Importance:**
- The two methods disagree, and the disagreement is informative.
- Gain is measured on the training data and credits `location` with 92%, because its 215 dummy columns offer many split points. Gain measures *use*.
- Permutation importance shuffles one input on the held-out test set. It shows the model *needs* `total_sqft` most by far.
- `bhk` scores about zero because `total_sqft` and `bath` already carry its information.

**Selection:** SelectKBest (f_regression) and RFE (XGBoost ranker) ran inside the CV pipeline.
- Keeping 100 of the 222 encoded columns with SelectKBest gives CV R² 0.676, against 0.697 with all columns.
- With 3 columns (sqft, bath, bhk) it gives 0.581.
- Location detail adds accuracy steadily, so the final model keeps every feature.

## 8. Tools

Python 3.12, pandas 2.3, NumPy 2.3, SciPy 1.16, scikit-learn 1.8, XGBoost 3.4, mlxtend 0.23, ydata-profiling 4.18, missingno 0.5, Matplotlib 3.10, joblib 1.5, Streamlit 1.56.

## 9. Limitations and future work

- **Asking prices:** the data holds listing (asking) prices, not sale prices, from a single snapshot with no listing date. The model cannot capture price changes over time.
- **Remaining error:** test R² 0.64 leaves a lot unexplained. Floor, age, amenities and exact coordinates are not in the data.
- **Grouped locations:** grouping 972 rare locations as `other` makes predictions for them coarse. Target encoding or geographic coordinates could help.
- **Weak cluster structure:** the low silhouette values mean the segment boundaries are soft.
- **Future work:** geocoding the locations for spatial features, quantile regression for prediction intervals, and SHAP values for per-prediction explanations.
