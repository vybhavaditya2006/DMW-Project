# Modelling report

Generated 2026-11-23 22:22 by `src/modeling.py` from `outputs/clean_data.csv`.

## Setup

- Data: 11,992 clean listings. Features: `location`, `area_type`, `total_sqft`, `bath`, `bhk`. Target: `price` (lakhs).
- Hold-out test set: 2,399 rows (20%, `random_state=42`), used only for the final scores. Everything else uses the 9,593 training rows.
- 5-fold cross-validation (shuffled, `random_state=42`), identical folds for every model.
- Pipeline: `ColumnTransformer` (OneHotEncoder for location/area_type, StandardScaler for total_sqft/bath/bhk) → model, all inside the CV so encoders and scalers never see the validation fold.
- `log_price` variant: a log-log specification. `TransformedTargetRegressor(func=np.log, inverse_func=np.exp)` makes the model learn log(price) while predictions and metrics stay in lakhs, and `total_sqft` is logged before scaling. (Log price with raw sqft made the linear models predict exponential growth with area and explode on the largest listings: CV R² ≈ −3,900 in the first run.)
- MAE and RMSE are in lakhs. `price_per_sqft` is not a feature (it is derived from the target).

## 1. Model comparison (5-fold CV, mean ± std across folds)

| model | target | cv_r2_mean | cv_r2_std | cv_mae_mean | cv_mae_std | cv_rmse_mean | cv_rmse_std | fit_seconds |
|---|---|---|---|---|---|---|---|---|
| RandomForest | log_price | 0.685 | 0.0271 | 31.17 | 1.54 | 82.77 | 7.07 | 4.95 |
| XGBoost | log_price | 0.684 | 0.0303 | 30.60 | 1.05 | 82.83 | 7.21 | 0.808 |
| GradientBoosting | price | 0.681 | 0.0581 | 33.97 | 1.16 | 82.82 | 7.31 | 0.341 |
| XGBoost | price | 0.671 | 0.0885 | 31.87 | 0.787 | 83.45 | 9.32 | 0.791 |
| RandomForest | price | 0.67 | 0.0563 | 31.66 | 1.52 | 84.28 | 6.74 | 4.42 |
| GradientBoosting | log_price | 0.649 | 0.0353 | 32.85 | 1.27 | 87.54 | 9.59 | 0.346 |
| Ridge | price | 0.559 | 0.0921 | 39.12 | 1.10 | 97.75 | 14.88 | 0.0107 |
| Lasso | price | 0.556 | 0.0924 | 39.33 | 1.11 | 98.07 | 14.70 | 0.143 |
| LinearRegression | price | 0.556 | 0.0923 | 39.38 | 1.11 | 98.09 | 14.69 | 0.0127 |
| DecisionTree | log_price | 0.522 | 0.0639 | 37.06 | 1.63 | 101.46 | 5.53 | 0.187 |
| DecisionTree | price | 0.462 | 0.106 | 37.20 | 1.21 | 107.25 | 8.46 | 0.19 |
| Ridge | log_price | 0.443 | 0.254 | 33.30 | 1.67 | 107.25 | 26.37 | 0.0114 |
| LinearRegression | log_price | 0.44 | 0.253 | 33.27 | 1.67 | 107.69 | 26.62 | 0.0147 |
| Lasso | log_price | 0.384 | 0.316 | 36.40 | 2.04 | 111.55 | 25.19 | 0.0157 |

## 2. Does predicting log_price help?

| model | r2_price | r2_log_price | mae_price | mae_log_price | r2_change | mae_change |
|---|---|---|---|---|---|---|
| RandomForest | 0.67 | 0.685 | 31.66 | 31.17 | 0.0153 | -0.489 |
| XGBoost | 0.671 | 0.684 | 31.87 | 30.60 | 0.0131 | -1.27 |
| GradientBoosting | 0.681 | 0.649 | 33.97 | 32.85 | -0.032 | -1.13 |
| DecisionTree | 0.462 | 0.522 | 37.20 | 37.06 | 0.0601 | -0.14 |
| Ridge | 0.559 | 0.443 | 39.12 | 33.30 | -0.116 | -5.82 |
| LinearRegression | 0.556 | 0.44 | 39.38 | 33.27 | -0.115 | -6.11 |
| Lasso | 0.556 | 0.384 | 39.33 | 36.40 | -0.172 | -2.93 |

log_price improves R² for 3 of 7 models and lowers MAE for 7 of 7.

## 3. Tuning the top 2 (GridSearchCV, same folds, refit on best mean R²)

| model | target | untuned_cv_r2 | untuned_cv_mae |
|---|---|---|---|
| RandomForest | log_price | 0.685 | 31.17 |
| XGBoost | log_price | 0.684 | 30.60 |

**RandomForest: top 5 of 24 parameter combinations**

| target | params | cv_r2_mean | cv_r2_std | cv_mae_mean | cv_mae_std | cv_rmse_mean | cv_rmse_std | rank_r2 |
|---|---|---|---|---|---|---|---|---|
| log_price | max_depth=None, max_features=1.0, min_samples_leaf=1, n_estimators=300 | 0.685 | 0.0271 | 31.17 | 1.54 | 82.77 | 7.07 | 1 |
| log_price | max_depth=25, max_features=1.0, min_samples_leaf=1, n_estimators=300 | 0.684 | 0.0269 | 31.38 | 1.43 | 82.81 | 6.92 | 2 |
| log_price | max_depth=25, max_features=1.0, min_samples_leaf=1, n_estimators=600 | 0.684 | 0.0283 | 31.35 | 1.43 | 82.92 | 7.25 | 3 |
| log_price | max_depth=None, max_features=1.0, min_samples_leaf=1, n_estimators=600 | 0.684 | 0.029 | 31.16 | 1.54 | 82.94 | 7.41 | 4 |
| log_price | max_depth=25, max_features=0.3, min_samples_leaf=1, n_estimators=300 | 0.671 | 0.0289 | 31.13 | 1.48 | 84.53 | 6.90 | 5 |

**XGBoost: top 5 of 24 parameter combinations**

| target | params | cv_r2_mean | cv_r2_std | cv_mae_mean | cv_mae_std | cv_rmse_mean | cv_rmse_std | rank_r2 |
|---|---|---|---|---|---|---|---|---|
| log_price | learning_rate=0.03, max_depth=8, n_estimators=800, subsample=1.0 | 0.697 | 0.0429 | 30.04 | 1.23 | 80.97 | 7.07 | 1 |
| log_price | learning_rate=0.06, max_depth=6, n_estimators=800, subsample=1.0 | 0.694 | 0.0295 | 29.87 | 1.04 | 81.57 | 7.39 | 2 |
| log_price | learning_rate=0.06, max_depth=8, n_estimators=400, subsample=1.0 | 0.693 | 0.048 | 30.01 | 1.21 | 81.37 | 7.49 | 3 |
| log_price | learning_rate=0.03, max_depth=8, n_estimators=400, subsample=0.8 | 0.693 | 0.0421 | 30.27 | 1.13 | 81.45 | 6.58 | 4 |
| log_price | learning_rate=0.03, max_depth=8, n_estimators=400, subsample=1.0 | 0.692 | 0.0381 | 30.62 | 1.20 | 81.62 | 6.67 | 5 |

## 4. Final model

Chosen by cross-validated R² (not by the test set): **XGBoost** with target `log_price`, parameters `learning_rate=0.03, max_depth=8, n_estimators=800, subsample=1.0`.

| cv_r2 | test_r2 | test_mae | test_rmse |
|---|---|---|---|
| 0.697 | 0.639 | 29.33 | 82.12 |

The saved `outputs/best_model.joblib` is this configuration refitted on all 11,992 clean rows; the test scores above come from the fit on the training rows only.

## 5. v1 vs v2

| step | data | cv_r2_mean | cv_r2_std | cv_mae_mean | cv_mae_std | cv_rmse_mean | cv_rmse_std | test_r2 | test_mae | test_rmse |
|---|---|---|---|---|---|---|---|---|---|---|
| 0. v1 as shipped (raw data, 3 numeric features, LinearRegression) | v1 dropna (13,201 rows) | 0.375 | 0.0916 | 45.77 | 1.40 | 117.39 | 13.29 | 0.438 | 47.10 | 107.66 |
| 1. same v1 model, v2 clean data | v2 clean (11,992 rows) | 0.501 | 0.0912 | 42.93 | 1.37 | 104.24 | 15.69 | 0.449 | 42.02 | 101.45 |
| 2. + location and area_type (one-hot) | v2 clean (11,992 rows) | 0.556 | 0.0923 | 39.38 | 1.11 | 98.09 | 14.69 | 0.509 | 37.90 | 95.80 |
| 3. + log-log (log_price target, log total_sqft) | v2 clean (11,992 rows) | 0.44 | 0.253 | 33.27 | 1.67 | 107.69 | 26.62 | 0.517 | 32.12 | 95.04 |
| 4. tuned best model (XGBRegressor) | v2 clean (11,992 rows) | 0.697 | 0.0429 | 30.04 | 1.23 | 80.97 | 7.07 | 0.639 | 29.33 | 82.12 |

- Overall, v1 as shipped → final v2 model: test R² 0.438 → 0.639, MAE 47.10 → 29.33 lakhs (error falls by 38%).
- **Step 1, wrangling** (same 3 features, same LinearRegression): test R² 0.438 → 0.449, MAE 47.10 → 42.02 lakhs (error falls by 11%). Caution: this step changes which rows are scored, not just their quality. Removing impossible listings also removes some of the hardest cases, so part of this change is an easier test set rather than a better model. Steps 2–4 use exactly the same rows, split and folds, so their changes are a fair comparison.
- **Step 2, location** (one-hot location and area_type): test R² 0.449 → 0.509, MAE 42.02 → 37.90 lakhs (error falls by 10%). v1 ignored location entirely, yet the same flat can cost several times more in a premium area.
- **Step 3, log-log** (log_price target, log total_sqft): test R² 0.509 → 0.517, MAE 37.90 → 32.12 lakhs (error falls by 15%). On the log scale each feature acts as a percentage change in price, which matches how prices behave, and the few very expensive listings stop dominating the squared-error fit.
- **Step 4, non-linear model + tuning**: test R² 0.517 → 0.639, MAE 32.12 → 29.33 lakhs (error falls by 9%). Trees can learn interactions a linear model cannot, e.g. that an extra 1,000 sq ft is worth far more in one location than in another.

## 6. `predict_price()` examples

| location | sqft | bath | bhk | area_type | predicted_lakhs |
|---|---|---|---|---|---|
| Whitefield | 1,200 | 2 | 2 | Super built-up Area | 59.82 |
| Indira Nagar | 1,200 | 2 | 2 | Super built-up Area | 99.51 |
| Electronic City | 1,200 | 2 | 2 | Super built-up Area | 43.47 |
| Electronic City | 2,000 | 3 | 3 | Super built-up Area | 117.04 |
| Some New Layout | 1,500 | 2 | 3 | Built-up Area | 81.51 |
