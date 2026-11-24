# Wrangling report

Generated 2026-11-24 20:50 by `src/wrangling.py` from `Bengaluru_House_Data.csv`.
Raw data: **13,320 rows × 9 columns**. Clean data: **11,992 rows × 15 columns** (1,328 rows removed, 10.0%).

Raw profile: [`raw_profile.html`](raw_profile.html). Missing-value plots: `outputs/plots_v2/00_missingno_matrix.png`, `outputs/plots_v2/00_missingno_heatmap.png`. Before/after distributions and a row-count waterfall: `outputs/plots_v2/01_distributions_before_after.png`, `outputs/plots_v2/02_wrangling_waterfall.png`.

## Summary

| step | title | rows_before | rows_after | rows_removed | nulls_before | nulls_after |
|---|---|---|---|---|---|---|
| 1 | Normalise text and drop exact duplicates | 13,320 | 12,790 | 530 | 6,201 | 6,023 |
| 2 | Parse size into bhk | 12,790 | 12,774 | 16 | 6,023 | 5,972 |
| 3 | Convert total_sqft to square feet | 12,774 | 12,774 | 0 | 5,972 | 5,972 |
| 4 | Parse availability | 12,774 | 12,774 | 0 | 5,972 | 26,314 |
| 5 | Handle missing values | 12,774 | 12,774 | 0 | 26,314 | 20,342 |
| 6 | Engineer price_per_sqft and log_price | 12,774 | 12,774 | 0 | 20,342 | 20,342 |
| 7 | Detect and remove outliers | 12,774 | 11,992 | 782 | 20,342 | 18,856 |
| 8 | Group rare locations and bin price | 11,992 | 11,992 | 0 | 18,856 | 18,856 |

## Step 1: Normalise text and drop exact duplicates

*Strip and collapse whitespace in every text column, unify location spellings that differ only by letter case, then drop rows that are exact duplicates.*

- Rows: 13,320 → 12,790 (530 removed). Columns: 10 → 10. Nulls: 6,201 → 6,023.
- Null counts changed: `society` 5,502 → 5,328, `balcony` 609 → 605
- Every `area_type` value had a double space ("Super built-up  Area"); 13,320 values fixed.
- Location case variants unified: 16 values; unique locations 1,286 → 1,273.
- Exact duplicate rows in the raw file: 529. After normalisation: 530 (removed, first copy kept).

**Decision:** Normalise before de-duplicating, so rows that differ only by stray spaces count as duplicates. Duplicates are removed because they over-weight some listings in every statistic and can leak between train and test sets.

**Values changed by whitespace normalisation**

| column | values_changed |
|---|---|
| area_type | 13,320 |
| availability | 0 |
| location | 504 |
| size | 0 |
| society | 1,155 |
| total_sqft | 0 |

## Step 2: Parse size into bhk

*Regex-parse `size` ("2 BHK", "4 Bedroom", "1 RK") into an integer `bhk` and keep the suffix as `size_unit`, because it encodes the property type.*

- Rows: 12,790 → 12,774 (16 removed). Columns: 10 → 11. Nulls: 6,023 → 5,972.
- Columns added: `bhk`, `size_unit`
- Columns dropped: `size`
- Null counts changed: `size` 16 → 0, `society` 5,328 → 5,325, `bath` 73 → 57, `balcony` 605 → 589
- 16 rows have no size. All are Plot Area listings (availability: Immediate Possession) with bath missing in 16 and balcony missing in 16: they are bare land plots, not houses.
- 12 listings have more than 10 bedrooms (max 43); handled in step 7.

**Decision:** Drop the rows without a size: a house-price model cannot use a listing with no bedrooms, bathrooms or balconies, and guessing all three for empty land would invent data.

**size_unit by area_type**

| area_type | BHK | Bedroom | RK |
|---|---|---|---|
| Built-up Area | 1,885 | 509 | 4 |
| Carpet Area | 82 | 5 | 0 |
| Plot Area | 1 | 1,972 | 0 |
| Super built-up Area | 8,305 | 2 | 9 |

**bhk distribution**

| bhk | count |
|---|---|
| 1 | 638 |
| 2 | 5,244 |
| 3 | 4,655 |
| 4 | 1,398 |
| 5 | 350 |
| 6 | 221 |
| 7 | 99 |
| 8 | 89 |
| 9 | 54 |
| 10 | 14 |
| 11 | 4 |
| 12 | 1 |
| 13 | 1 |
| 14 | 1 |
| 16 | 1 |
| 18 | 1 |
| 19 | 1 |
| 27 | 1 |
| 43 | 1 |

## Step 3: Convert total_sqft to square feet

*Turn the text column `total_sqft` into a number: plain numbers as they are, ranges ("2100 - 2850") to their midpoint, other units to square feet.*

- Rows: 12,774 → 12,774 (0 removed). Columns: 11 → 11. Nulls: 5,972 → 5,972.
- No unparseable values: every total_sqft was converted.
- 238 values needed conversion; v1 kept the ranges but turned all 46 unit values into NaN and dropped them.

**Decision:** Ranges become their midpoint (the best single estimate of a range). Units use standard conversion factors; Guntha, Cents and Grounds are Indian land units.

**How each value was parsed**

| how | rows | example_raw | example_sqft |
|---|---|---|---|
| number | 12,536 | 1056 | 1,056 |
| range (midpoint) | 192 | 2100 - 2850 | 2,475 |
| unit: Sq. Meter | 17 | 34.46Sq. Meter | 370.92 |
| unit: Sq. Yards | 15 | 1100Sq. Yards | 9,900 |
| unit: Acres | 8 | 5.31Acres | 231,304 |
| unit: Guntha | 2 | 24Guntha | 26,136 |
| unit: Cents | 2 | 3Cents | 1,307 |
| unit: Grounds | 1 | 1Grounds | 2,400 |
| unit: Perch | 1 | 4125Perch | 1,123,031 |

**Conversion factors used (sq ft per unit)**

| unit | sqft_per_unit |
|---|---|
| Sq. Meter | 10.76 |
| Sq. Yards | 9.00 |
| Acres | 43,560 |
| Perch | 272.25 |
| Cents | 435.60 |
| Guntha | 1,089 |
| Grounds | 2,400 |

## Step 4: Parse availability

*Split `availability` into `is_ready_to_move` (1/0) and, for properties not yet ready, `possession_year` and `possession_month` from values like "18-Dec".*

- Rows: 12,774 → 12,774 (0 removed). Columns: 11 → 13. Nulls: 5,972 → 26,314.
- Columns added: `is_ready_to_move`, `possession_year`, `possession_month`
- Columns dropped: `availability`
- Null counts changed: `possession_year` 0 → 10,171, `possession_month` 0 → 10,171
- Date-like values: 2,603. The 2-digit prefix only takes values 14–22 while all 12 months occur, so it is a year ("18-Dec" = Dec 2018), not a day of the month.
- `possession_year`/`possession_month` are empty for ready properties by design (there is no future date), which is why the null count rises in this step.

**Decision:** "Immediate Possession" is treated as ready to move. (All 16 such rows in the raw file were the land plots removed in step 2.)

**Availability categories**

| category | rows |
|---|---|
| Ready To Move | 10,171 |
| Immediate Possession | 0 |
| future possession date | 2,603 |
| unrecognised | 0 |

## Step 5: Handle missing values

*Drop `society`, fill the missing `location` with the mode, and impute `bath` and `balcony` with the better of per-bhk median and KNN imputation.*

- Rows: 12,774 → 12,774 (0 removed). Columns: 13 → 12. Nulls: 26,314 → 20,342.
- Columns dropped: `society`
- Null counts changed: `location` 1 → 0, `society` 5,325 → 0, `bath` 57 → 0, `balcony` 589 → 0
- `society`: 41.7% missing and 2,676 distinct values among 7,449 known rows (median 1 listing per society; 1,523 societies appear once).
- raw_row 568: location filled with 'Anantapura' (mode of society 'Grare S').
- `bath`: 57 missing; the two methods give different values for 19 of them.
- `balcony`: 589 missing; the two methods give different values for 192 of them.

**Decision:** Drop `society` (41.7% missing and 2,676 distinct values among 7,449 known rows (median 1 listing per society; 1,523 societies appear once)): too much is missing to impute honestly, and one-hot or target encoding thousands of tiny groups would overfit. For bath/balcony, mean MAE is 0.423 (median by bhk) vs 0.431 (KNN), so KNN is 1.9% less accurate. Rule: use KNN only if it is more than 5% better, otherwise prefer the simpler, deterministic median. Chosen: **median_by_bhk**.

**Missing values at the start of this step**

| column | missing | pct |
|---|---|---|
| location | 1 | 0 |
| society | 5,325 | 41.70 |
| bath | 57 | 0.4 |
| balcony | 589 | 4.60 |
| possession_year | 10,171 | 79.60 |
| possession_month | 10,171 | 79.60 |

**Imputer comparison (10% of known values hidden, 5 repeats)**

| column | method | mae | exact_match |
|---|---|---|---|
| balcony | knn_k5 | 0.605 | 0.473 |
| balcony | median_by_bhk | 0.574 | 0.491 |
| bath | knn_k5 | 0.258 | 0.773 |
| bath | median_by_bhk | 0.272 | 0.762 |

**Imputed values on the actually-missing rows**

| value | bath (median) | bath (knn) | balcony (median) | balcony (knn) |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 6 |
| 1 | 7 | 7 | 56 | 136 |
| 2 | 1 | 7 | 521 | 388 |
| 3 | 24 | 16 | 12 | 59 |
| 4 | 22 | 16 | 0 | 0 |
| 5 | 3 | 8 | 0 | 0 |
| 6 | 0 | 1 | 0 | 0 |
| 7 | 0 | 2 | 0 | 0 |

## Step 6: Engineer price_per_sqft and log_price

*Add `price_per_sqft` in rupees (price is in lakhs, 1 lakh = ₹100,000) and `log_price`, the natural log of price, to tame its right skew.*

- Rows: 12,774 → 12,774 (0 removed). Columns: 12 → 14. Nulls: 20,342 → 20,342.
- Columns added: `price_per_sqft`, `log_price`
- Skewness of price 7.96 → log_price 0.82.

**Decision:** `price_per_sqft` is used for outlier detection, clustering and analysis only. It is computed from the price, so it must never be a model input (target leakage).

**Distribution**

| feature | count | mean | std | min | 25% | 50% | 75% | max | skew |
|---|---|---|---|---|---|---|---|---|---|
| price | 12,774 | 114.34 | 151.51 | 8.00 | 50.00 | 73.00 | 121.00 | 3,600 | 7.96 |
| log_price | 12,774 | 4.41 | 0.726 | 2.08 | 3.91 | 4.29 | 4.80 | 8.19 | 0.82 |
| price_per_sqft | 12,774 | 8,023 | 108,493 | 2.26 | 4,296 | 5,482 | 7,401 | 12,000,000 | 106.51 |

## Step 7: Detect and remove outliers

*Flag outliers with nine methods side by side, compare what each one catches, then remove only the rows flagged by the chosen strategy.*

- Rows: 12,774 → 11,992 (782 removed). Columns: 14 → 14. Nulls: 20,342 → 18,856.
- Null counts changed: `possession_year` 10,171 → 9,428, `possession_month` 10,171 → 9,428
- Per-row flags saved to `outputs/outlier_flags.csv`.
- Median price_per_sqft of all rows: ₹5,482; of rows flagged by iqr_pps: ₹15,833; by rule_location_pps: ₹7,742.
- z-score uses the mean and std, which the extreme rows themselves inflate (max price_per_sqft ₹12,000,000), so it flags only 5 rows.

**Decision:** Remove 782 rows (6.1%) flagged by either of two kinds of evidence. (1) The two physical domain rules (747 rows): less than 300 sq ft per bedroom, or more than bhk + 2 bathrooms. These are almost certainly data errors whatever the location. (2) Tukey's far-out fence on log price_per_sqft (47 rows, 35 not already caught by the domain rules): outside ₹841 … ₹37,830 per sq ft. The domain rules only catch areas that are too small; this catches the opposite errors too, such as land parcels converted from Acres/Perch (a 2-BHK on 1.3 million sq ft at ₹2/sq ft). It uses quartiles, so the extreme rows cannot inflate it the way they inflate the z-score, and the log scale makes it symmetric for a right-skewed price. The other methods are kept as diagnostic flags, not used for removal: per-location mean ± 1 std would remove 2,265 rows (17.7%), because about a third of any distribution lies outside ±1 std even when nothing is wrong; the city-wide IQR (1,206 rows) mostly removes genuinely expensive localities; the bhk-inversion rule (2,182 rows) assumes bigger flats cannot be cheaper per sq ft, which is not true in general; IsolationForest and LOF need a contamination rate that we would have to guess (5% here), so the number they flag is our choice, not a finding.

**Rows flagged by each method**

| method | family | rule | flagged | pct | also_in_final |
|---|---|---|---|---|---|
| iqr_pps | statistical | price_per_sqft outside Q1 − 1.5·IQR … Q3 + 1.5·IQR (whole city) | 1,206 | 9.40 | 418 |
| zscore_pps | statistical | \|z-score of price_per_sqft\| > 3 (whole city) | 5 | 0 | 5 |
| iqr3_log_pps | statistical | log price_per_sqft outside Q1 − 3·IQR … Q3 + 3·IQR (Tukey far-out fence) | 47 | 0.4 | 47 |
| isolation_forest | ML, multivariate | IsolationForest on log sqft, bhk, bath, log price_per_sqft (5% contamination) | 639 | 5.00 | 284 |
| lof | ML, multivariate | LocalOutlierFactor, 20 neighbours, same features (5% contamination) | 639 | 5.00 | 103 |
| rule_sqft_per_bhk | domain rule | total_sqft / bhk < 300 | 738 | 5.80 | 738 |
| rule_bath | domain rule | bath > bhk + 2 | 16 | 0.1 | 16 |
| rule_location_pps | domain rule | price_per_sqft outside its location's mean ± 1 std | 2,265 | 17.70 | 408 |
| rule_bhk_pps | domain rule | n-BHK cheaper per sqft than the mean (n−1)-BHK in the same location (>5 listings) | 2,182 | 17.10 | 18 |

**Overlap: rows flagged by both methods**

| method | iqr_pps | zscore_pps | iqr3_log_pps | isolation_forest | lof | rule_sqft_per_bhk | rule_bath | rule_location_pps | rule_bhk_pps |
|---|---|---|---|---|---|---|---|---|---|
| iqr_pps | 1,206 | 5 | 23 | 302 | 128 | 402 | 10 | 582 | 6 |
| zscore_pps | 5 | 5 | 5 | 5 | 5 | 5 | 0 | 3 | 0 |
| iqr3_log_pps | 23 | 5 | 47 | 41 | 31 | 10 | 2 | 35 | 2 |
| isolation_forest | 302 | 5 | 41 | 639 | 103 | 247 | 15 | 289 | 18 |
| lof | 128 | 5 | 31 | 103 | 639 | 75 | 14 | 189 | 107 |
| rule_sqft_per_bhk | 402 | 5 | 10 | 247 | 75 | 738 | 7 | 379 | 16 |
| rule_bath | 10 | 0 | 2 | 15 | 14 | 7 | 16 | 8 | 0 |
| rule_location_pps | 582 | 3 | 35 | 289 | 189 | 379 | 8 | 2,265 | 258 |
| rule_bhk_pps | 6 | 0 | 2 | 18 | 107 | 16 | 0 | 258 | 2,182 |

**How many methods flag each row**

| methods_flagging | rows |
|---|---|
| 0 | 7,349 |
| 1 | 3,848 |
| 2 | 1,054 |
| 3 | 359 |
| 4 | 133 |
| 5 | 17 |
| 6 | 11 |
| 7 | 3 |

**Rows flagged by the most methods**

| raw_row | location | bhk | total_sqft | bath | price | price_per_sqft | methods |
|---|---|---|---|---|---|---|---|
| 4,086 | Sarjapur Road | 4 | 1.00 | 4 | 120.00 | 12,000,000 | 7 |
| 4,972 | Srirampuram | 7 | 5.00 | 7 | 115.00 | 2,300,000 | 7 |
| 11,558 | Whitefield | 4 | 60.00 | 4 | 218.00 | 363,333 | 7 |
| 349 | Suragajakkanahalli | 3 | 11.00 | 3 | 74.00 | 672,727 | 6 |
| 1,020 | Weavers Colony | 1 | 15.00 | 1 | 30.00 | 200,000 | 6 |

## Step 8: Group rare locations and bin price

*Replace locations with 10 or fewer listings by "other", and bin price into four quartile bands: Low, Mid, High, Premium.*

- Rows: 11,992 → 11,992 (0 removed). Columns: 14 → 15. Nulls: 18,856 → 18,856.
- Columns added: `price_band`
- 972 of 1,186 locations have ≤ 10 listings; their 2,741 rows (22.9%) are now 'other'. Locations left: 215 (including 'other').

**Decision:** Locations with very few listings give unreliable statistics and add hundreds of near-empty one-hot columns. Quartile bands give four equal-sized groups for association rules and the dashboard.

**Price bands (lakhs)**

| band | from | to | rows |
|---|---|---|---|
| Low | 9.00 | 49.98 | 2,998 |
| Mid | 49.98 | 70.00 | 3,047 |
| High | 70.00 | 119.00 | 2,954 |
| Premium | 119.00 | 2,912 | 2,993 |

## Output: `outputs/clean_data.csv`

| column | dtype | nulls | example |
|---|---|---|---|
| raw_row | int64 | 0 | 0 |
| area_type | object | 0 | Super built-up Area |
| location | object | 0 | Electronic City Phase II |
| total_sqft | float64 | 0 | 1056.0 |
| bath | int64 | 0 | 2 |
| balcony | int64 | 0 | 1 |
| price | float64 | 0 | 39.07 |
| bhk | int64 | 0 | 2 |
| size_unit | object | 0 | BHK |
| is_ready_to_move | int64 | 0 | 0 |
| possession_year | Int64 | 9,428 | 2019 |
| possession_month | Int64 | 9,428 | 12 |
| price_per_sqft | float64 | 0 | 3699.810606060606 |
| log_price | float64 | 0 | 3.6653549090669792 |
| price_band | category | 0 | Low |
