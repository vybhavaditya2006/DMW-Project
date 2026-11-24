# Demo script: v2 dashboard walkthrough (5–7 minutes)

**Before the demo:**
- Run `python run_pipeline.py --fresh` well in advance; it takes about 20 minutes.
- Start the dashboard with `streamlit run app_v2.py` and open each tab once, so the cached tables load instantly.
- Close other heavy apps.
- Keep a screen recording as a backup.

**Shape of the walkthrough:**
- Six tabs, about one minute each.
- *Do* = what to click. *Say* = the talking points; use your own words.
- Every tab starts with a "What this step does" card (purple bar on the left). Read it out briefly, then go to the details.

---

## 0. Opening (20 s)

**Say:**
- "v1 was last semester's project: basic cleaning and one linear regression, test R² 0.44."
- "v2 turns it into a data-mining and data-wrangling pipeline. One command, `run_pipeline.py`, goes from the raw CSV to every result, and every step logs what it did and why."
- "This dashboard only *reads* the saved results. Nothing is retrained when the page loads."

## 1. Raw Data (50 s)

**Do:**
- Point at the four headline numbers (13,320 listings, missing cells, 529 duplicates).
- Point at the data-quality problems table.
- Show the missingno matrix.

**Say:**
- "Before cleaning anything we profiled the raw file with **ydata-profiling** and **missingno**."
- "The problems are real and counted:
  - `total_sqft` is text: 201 ranges like '2100 - 2850' and 46 values in sq metres, sq yards, acres, even guntha and perch.
  - `size` is written three ways: BHK, Bedroom, RK.
  - Locations have stray spaces and spelling variants.
  - There are 744 listings with less than 300 sq ft per bedroom."
- "The missingno matrix shows `society` is 41% empty. The heatmap shows which columns go missing together: all 73 listings with no `bath` also have no `balcony`."
- "Every decision in the next tab answers one of these problems."

## 2. Wrangling (70 s)

**Do:**
- Show the step log table.
- In **Explain a step**, pick **Step 7** and read the decision.
- Scroll to the waterfall, then the outlier-methods table.

**Say:**
- "Cleaning is eight small functions. A logging decorator records rows, columns and nulls before and after each one, plus the reason. That's this table, saved to `reports/wrangling_log.csv`."
- "Some key decisions:
  - We normalise text *before* removing duplicates, so 'Anekal' and ' Anekal' count as the same: 530 duplicates removed.
  - Unit values are converted, not dropped like in v1.
  - For missing bathrooms and balconies we **tested two imputers** by hiding known values. The median per BHK beat KNNImputer, so we kept the simpler one."
- "For outliers we compared **nine methods**: domain rules, z-score, IQR, Tukey's far-out fence, per-location standard deviation, Isolation Forest and LOF, among others."
  - "We remove only what physical rules or the far-out fence on log price per sq ft flag: 782 rows, 6.1%."
  - "Per-location ±1 std would remove 18% of perfectly normal data."
  - "Isolation Forest needs us to *guess* the contamination rate, so the count it flags is our choice, not a finding."
- "The waterfall shows 13,320 rows becoming 11,992."

## 3. Price Prediction (80 s)

**Do:**
- Point at the four metric cards.
- In the predictor, keep **Whitefield, 1,500 sq ft, 3 BHK, 3 bathrooms** and press **PREDICT PRICE**.
- Then scroll to the v1 → v2 table.

**Say:**
- "Seven models: Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting and XGBoost. Each is trained on price and on log price."
- "They sit in a scikit-learn **Pipeline with a ColumnTransformer**: one-hot for location, scaling for numbers. The encoders are learned inside each cross-validation fold, so nothing leaks from the validation data."
- "We compare them with **5-fold cross-validation** on 80% of the data, tune the top two with **GridSearchCV**, and score the winner once on the untouched 20%."
- On the prediction: "About ₹81 lakhs. The range under it is the middle half of the model's real errors on test listings, so it is honest about uncertainty. Below it are actual listings like this one for context."
- On the v1 → v2 table: "We added one change at a time so we know what helped."
  - "Clean data: MAE 47 → 42 lakhs."
  - "Adding location: → 38."
  - "Log-log target: → 32."
  - "Tuned XGBoost: **R² 0.639, MAE ₹29.3 L, 38% less error than v1**."
- **If asked why log-log:** "Prices are heavily right-skewed (skewness 8). Logging both price and area makes the relationship close to linear and stops a few mansions dominating the loss."

## 4. Market Segments (70 s)

**Do:**
- In **Find a location**, pick **Koramangala**, then **Whitefield**.
- Show the segment profiles and the 2×2 cluster plot.
- Point briefly at the method comparison table.

**Say:**
- "Here the unit is a *location*, not a listing: 214 locations described by price per sq ft, size, bedrooms, listing volume and the share ready to move. Skewed features are logged and everything is standardised."
- "We ran three clustering methods: **K-Means**, **DBSCAN** and **Agglomerative (Ward)**."
- "To choose k we used the elbow, silhouette and **Davies-Bouldin**. Silhouette alone says k = 2, which just separates expensive from not. k = 4 is within 10% of the best silhouette and has the lowest Davies-Bouldin of all k."
- "The four segments:
  - Premium: Koramangala, Rajaji Nagar, about ₹9,000/sq ft.
  - Upper mid-market: high volume, like Whitefield.
  - Mid-market.
  - A budget, mostly under-construction group: Electronic City Phase II, Chandapura."
- "Honest finding: silhouette is only about 0.26, and DBSCAN finds one dense core plus 19 noise locations. Bengaluru is a continuum, so these segments summarise it; they are not hard boundaries."
- "The plots use **PCA** to show the five features in 2-D."

## 5. Association Rules (60 s)

**Do:**
- Type `price=Premium` in the filter box.
- Drag **Minimum lift** to 3.
- Scroll to the five plain-English rules.

**Say:**
- "Each listing becomes a shopping basket of items: price band, size band, BHK, bathrooms, area type, ready or under construction, and its market segment."
- "We mined it with **Apriori** and **FP-Growth**. Both are exact and gave identical itemsets."
- "Interestingly, Apriori was far faster here. FP-Growth wins with many sparse items, but we have only 27 items and every basket is dense."
- "Support is how common a rule is, confidence is how often it holds, and lift is how many times more likely than chance."
- "We pruned redundant rules and kept 617; the top 25 are chosen so they don't repeat each other."
- Read rule 1: "Of the 1-BHK, super built-up, low-price listings, 96% have one bathroom, against 6% overall: about 16× more likely."
- "These are co-occurrences, not causes."

## 6. Feature Importance (50 s)

**Do:**
- Show the importance table and plot.
- Show the feature-selection table.

**Say:**
- "We measured importance two ways, and they disagree, which is the interesting part."
- "XGBoost's built-in **gain** gives location 92%. But location has 215 dummy columns, so the trees have many places to split on it: gain measures *use*."
- "**Permutation importance** shuffles one column on the test set and measures the damage. Shuffling area drops R² by 0.76, so area is what the model really *needs*."
- "Then **SelectKBest** and **RFE**, run inside cross-validation, test whether fewer columns would do. Keeping 100 of 222 columns gives R² 0.676 against 0.697, so we keep all features."

## 7. Close (15 s)

**Say:**
- "To sum up: 13,320 messy listings cleaned to 11,992 with every step logged; a model with 38% less error than v1; four market segments; and 617 association rules. All of it is reproducible from one command with a fixed random seed."
- "Limitations: these are asking prices with no dates, and the data has no floor, age or amenity information."

---

## Likely viva questions

| Question | Answer |
|---|---|
| Why not use `price_per_sqft` as a model input? | It is computed from the price, so using it would leak the target. We use it only for outliers and clustering. |
| Why median imputation instead of KNN? | We tested both on masked known values. The median per BHK was slightly more accurate (MAE 0.423 vs 0.431), and it is simpler and deterministic. |
| Why 5-fold CV and a hold-out test? | CV picks and tunes models using the training data only. The test set is touched once at the end, so its score is not optimistic. |
| Why group rare locations as `other`? | 972 locations have 10 or fewer listings. Their averages are unreliable and they would add hundreds of near-empty one-hot columns. |
| Is R² 0.64 good? | It is 0.20 above v1 on the same task. The rest is variance the data cannot explain: floor, age, amenities and exact position are missing. |
| Why does DBSCAN find only one cluster? | The locations have no separate dense regions. That agrees with the low silhouette and is itself a finding. |
| Why is FP-Growth slower here? | Few items and dense baskets suit mlxtend's vectorised Apriori. Its FP-Growth is pure Python. |
| Why does the app not retrain? | `run_pipeline.py` saves the model with joblib and the app loads it, so the demo is fast and shows exactly the results in the reports. |
