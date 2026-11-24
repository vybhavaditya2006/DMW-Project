"""
Price modelling
===============
Compares seven regressors on outputs/clean_data.csv, tests whether predicting
log(price) helps, tunes the two best with GridSearchCV, and measures how far
v2 improves on the v1 linear regression.

Protocol (so the numbers can be trusted):
  - 20% of the clean data is held out as a test set and only scored at the end.
  - All comparison and tuning use 5-fold cross-validation on the other 80%,
    with the same folds for every model.
  - Preprocessing lives inside the Pipeline, so encoders and scalers are fitted
    on the training folds only (no leakage from the validation fold).
  - With a log target, TransformedTargetRegressor converts predictions back to
    lakhs, so every metric (R², MAE, RMSE) is on the same scale.

Outputs:
  reports/model_comparison.csv   CV results, 7 models x {price, log_price}
  reports/tuning_results.csv     GridSearchCV results for the top 2 models
  reports/v1_vs_v2.csv           step-by-step comparison with the v1 model
  reports/modeling_report.md     all of the above, explained
  outputs/best_model.joblib      fitted pipeline + metadata for predict_price()
  outputs/test_predictions.csv   held-out test rows with actual and predicted price

Usage:
    python -m src.modeling
"""

import logging
import re
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src import OUTPUTS_DIR, RANDOM_STATE, REPORTS_DIR
from src.profiling import load_raw
from src.wrangling import md_table

log = logging.getLogger(__name__)

CLEAN_CSV = OUTPUTS_DIR / "clean_data.csv"
BEST_MODEL_PATH = OUTPUTS_DIR / "best_model.joblib"

CATEGORICAL = ["location", "area_type"]
NUMERIC = ["total_sqft", "bath", "bhk"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "price"

TEST_SIZE = 0.2
N_FOLDS = 5
SCORING = {"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE CONSTRUCTION
# ──────────────────────────────────────────────────────────────────────────────
def make_preprocessor(categorical=CATEGORICAL, numeric=NUMERIC, log_sqft: bool = False) -> ColumnTransformer:
    """One-hot encode the categories, standardise the numbers (total_sqft logged first if log_sqft)."""
    plain = [c for c in numeric if not (log_sqft and c == "total_sqft")]
    transformers = [("num", StandardScaler(), plain)]
    if log_sqft:
        transformers.append(("log_sqft", Pipeline([("log", FunctionTransformer(np.log, feature_names_out="one-to-one")),
                                                   ("scale", StandardScaler())]), ["total_sqft"]))
    if categorical:
        # handle_unknown="ignore": a location unseen in training becomes all zeros instead of an error
        transformers.insert(0, ("cat", OneHotEncoder(handle_unknown="ignore"), categorical))
    return ColumnTransformer(transformers)


def make_pipeline(model, log_target: bool, categorical=CATEGORICAL, numeric=NUMERIC) -> TransformedTargetRegressor:
    """Preprocessor + model, wrapped so the model can learn log(price) but still predict lakhs.

    The log variant is a log-log specification: log(price) from log(total_sqft).
    With log(price) but raw sqft, a linear model makes price grow exponentially
    with area and explodes on the largest listings (first run: CV R² ≈ −3,900).
    Trees are unaffected, because a log does not change the order of values.

    The wrapper is always present (identity when log_target is False), so tuning
    parameter names are the same either way: regressor__model__<param>.
    """
    pipe = Pipeline([("prep", make_preprocessor(categorical, numeric, log_sqft=log_target)), ("model", model)])
    if log_target:
        return TransformedTargetRegressor(regressor=pipe, func=np.log, inverse_func=np.exp)
    return TransformedTargetRegressor(regressor=pipe)


def candidate_models() -> dict:
    """The seven models compared, at sensible default settings."""
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        # A small alpha: the default (1.0) zeroes almost every coefficient on the log scale.
        "Lasso": Lasso(alpha=0.001, max_iter=20_000),
        "DecisionTree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "XGBoost": XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6,
                                n_jobs=-1, random_state=RANDOM_STATE),
    }


PARAM_GRIDS = {
    "LinearRegression": {},
    "Ridge": {"alpha": [0.01, 0.1, 1, 10, 100]},
    "Lasso": {"alpha": [0.0001, 0.001, 0.01, 0.1]},
    "DecisionTree": {"max_depth": [8, 12, 16, None], "min_samples_leaf": [1, 5, 10, 20]},
    "RandomForest": {"n_estimators": [300, 600], "max_depth": [None, 25],
                     "min_samples_leaf": [1, 2, 4], "max_features": [0.3, 1.0]},
    "GradientBoosting": {"n_estimators": [300, 600], "learning_rate": [0.05, 0.1],
                         "max_depth": [3, 5], "subsample": [0.8, 1.0]},
    "XGBoost": {"n_estimators": [400, 800], "learning_rate": [0.03, 0.06],
                "max_depth": [4, 6, 8], "subsample": [0.8, 1.0]},
}


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def folds() -> KFold:
    return KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def cv_scores(estimator, X, y) -> dict:
    """5-fold CV: mean and std of R², MAE and RMSE (MAE/RMSE in lakhs)."""
    res = cross_validate(estimator, X, y, cv=folds(), scoring=SCORING, n_jobs=1)
    out = {}
    for name in SCORING:
        values = res[f"test_{name}"] * (1 if name == "r2" else -1)  # sklearn negates error scores
        out[f"cv_{name}_mean"], out[f"cv_{name}_std"] = values.mean(), values.std()
    out["fit_seconds"] = res["fit_time"].mean()
    return out


def test_scores(fitted, X, y) -> dict:
    pred = fitted.predict(X)
    return {"test_r2": r2_score(y, pred), "test_mae": mean_absolute_error(y, pred),
            "test_rmse": root_mean_squared_error(y, pred)}


def load_clean(path: Path = CLEAN_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    log.info("Loaded %s: %d rows", path.name, len(df))
    return df


def split(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=TEST_SIZE, random_state=RANDOM_STATE)
    log.info("Train %d rows, test %d rows (held out until the end)", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────────────────────────────────────
# 1. MODEL COMPARISON
# ──────────────────────────────────────────────────────────────────────────────
def compare_models(X_train, y_train) -> pd.DataFrame:
    """Cross-validate every model with both targets: price and log(price)."""
    rows = []
    for name, model in candidate_models().items():
        for log_target in (False, True):
            start = time.perf_counter()
            scores = cv_scores(make_pipeline(model, log_target), X_train, y_train)
            rows.append({"model": name, "target": "log_price" if log_target else "price", **scores})
            log.info("%-16s %-9s R² %.3f ± %.3f  MAE %6.2f ± %.2f  RMSE %6.2f ± %.2f  (%.1fs)",
                     name, rows[-1]["target"], scores["cv_r2_mean"], scores["cv_r2_std"],
                     scores["cv_mae_mean"], scores["cv_mae_std"], scores["cv_rmse_mean"],
                     scores["cv_rmse_std"], time.perf_counter() - start)
    return pd.DataFrame(rows).sort_values("cv_r2_mean", ascending=False).reset_index(drop=True)


def log_target_effect(comparison: pd.DataFrame) -> pd.DataFrame:
    """For each model: CV R² and MAE with price vs log_price as the target."""
    wide = comparison.pivot(index="model", columns="target", values=["cv_r2_mean", "cv_mae_mean"])
    out = pd.DataFrame({
        "r2_price": wide[("cv_r2_mean", "price")],
        "r2_log_price": wide[("cv_r2_mean", "log_price")],
        "mae_price": wide[("cv_mae_mean", "price")],
        "mae_log_price": wide[("cv_mae_mean", "log_price")],
    })
    out["r2_change"] = out["r2_log_price"] - out["r2_price"]
    out["mae_change"] = out["mae_log_price"] - out["mae_price"]
    return out.sort_values("r2_log_price", ascending=False).rename_axis("model").reset_index()


def top_models(comparison: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    """Best target variant of each model, then the n models with the highest mean CV R²."""
    best_variant = comparison.loc[comparison.groupby("model")["cv_r2_mean"].idxmax()]
    return best_variant.sort_values("cv_r2_mean", ascending=False).head(n).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# 2. TUNING
# ──────────────────────────────────────────────────────────────────────────────
def tune(name: str, log_target: bool, X_train, y_train) -> tuple[GridSearchCV, pd.DataFrame]:
    """GridSearchCV over PARAM_GRIDS[name] on the same folds, refitting on the best mean R²."""
    model = candidate_models()[name]
    grid = {f"regressor__model__{k}": v for k, v in PARAM_GRIDS[name].items()}
    # Parallelise across grid points only for models that are not already multi-threaded.
    n_jobs = 1 if "n_jobs" in model.get_params() else -1
    search = GridSearchCV(make_pipeline(model, log_target), grid, cv=folds(), scoring=SCORING,
                          refit="r2", n_jobs=n_jobs)
    start = time.perf_counter()
    search.fit(X_train, y_train)
    log.info("Tuned %s (%d combinations x %d folds) in %.0fs: best CV R² %.4f with %s",
             name, len(search.cv_results_["params"]), N_FOLDS, time.perf_counter() - start,
             search.best_score_, short_params(search.best_params_))

    res = pd.DataFrame(search.cv_results_)
    table = pd.DataFrame({
        "model": name,
        "target": "log_price" if log_target else "price",
        "params": res["params"].map(short_params),
        "cv_r2_mean": res["mean_test_r2"], "cv_r2_std": res["std_test_r2"],
        "cv_mae_mean": -res["mean_test_mae"], "cv_mae_std": res["std_test_mae"],
        "cv_rmse_mean": -res["mean_test_rmse"], "cv_rmse_std": res["std_test_rmse"],
        "rank_r2": res["rank_test_r2"],
    }).sort_values("rank_r2")
    return search, table


def short_params(params: dict) -> str:
    return ", ".join(f"{k.split('__')[-1]}={v}" for k, v in params.items())


# ──────────────────────────────────────────────────────────────────────────────
# 3. V1 BASELINE AND STEP-BY-STEP COMPARISON
# ──────────────────────────────────────────────────────────────────────────────
def v1_sqft(x):
    """v1's total_sqft conversion, copied from evaluate_model.py: ranges -> mean, anything else -> NaN."""
    try:
        return float(x)
    except (ValueError, TypeError):
        if isinstance(x, str) and "-" in x:
            parts = x.split("-")
            try:
                return (float(parts[0].strip()) + float(parts[1].strip())) / 2.0
            except (ValueError, IndexError):
                return np.nan
        return np.nan


def v1_data() -> pd.DataFrame:
    """The data exactly as v1's evaluate_model.py prepares it (raw CSV, dropna, no other cleaning)."""
    df = load_raw()
    df["bhk"] = df["size"].str.extract(r"(\d+)").astype(float)
    df["total_sqft"] = df["total_sqft"].apply(v1_sqft)
    return df[["total_sqft", "bath", "bhk", "price"]].dropna().reset_index(drop=True)


def compare_with_v1(clean: pd.DataFrame, X_train, X_test, y_train, y_test, best_estimator) -> pd.DataFrame:
    """Add one change at a time, so each gain can be attributed to its cause."""
    rows = []

    # Step 0: v1 exactly as shipped: raw data, 80/20 split (random_state=42), test-set score.
    v1 = v1_data()
    v1_X, v1_y = v1[NUMERIC], v1[TARGET]
    a, b, c, d = train_test_split(v1_X, v1_y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    v1_model = LinearRegression().fit(a, c)
    rows.append({"step": "0. v1 as shipped (raw data, 3 numeric features, LinearRegression)",
                 "data": f"v1 dropna ({len(v1):,} rows)", **cv_scores(LinearRegression(), v1_X, v1_y),
                 **test_scores(v1_model, b, d)})

    # Steps 1-3 use the clean data, the same split and the same folds.
    ladder = [
        ("1. same v1 model, v2 clean data", make_pipeline(LinearRegression(), False, [], NUMERIC)),
        ("2. + location and area_type (one-hot)", make_pipeline(LinearRegression(), False)),
        ("3. + log-log (log_price target, log total_sqft)", make_pipeline(LinearRegression(), True)),
    ]
    for label, est in ladder:
        rows.append({"step": label, "data": f"v2 clean ({len(clean):,} rows)",
                     **cv_scores(est, X_train, y_train), **test_scores(est.fit(X_train, y_train), X_test, y_test)})

    model_name = type(best_estimator.regressor.named_steps["model"]).__name__
    rows.append({"step": f"4. tuned best model ({model_name})", "data": f"v2 clean ({len(clean):,} rows)",
                 **cv_scores(best_estimator, X_train, y_train), **test_scores(best_estimator, X_test, y_test)})

    out = pd.DataFrame(rows).drop(columns="fit_seconds")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 4. SAVED MODEL AND PREDICTION
# ──────────────────────────────────────────────────────────────────────────────
def save_bundle(estimator, clean: pd.DataFrame, info: dict, path: Path = BEST_MODEL_PATH) -> Path:
    """Save the fitted pipeline together with what predict_price() needs to validate its inputs."""
    bundle = {
        "model": estimator,
        "features": FEATURES,
        "locations": sorted(clean["location"].unique()),
        "area_types": sorted(clean["area_type"].unique()),
        "trained_rows": len(clean),
        "created": datetime.now().isoformat(timespec="seconds"),
        **info,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    log.info("Saved %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path


_bundle_cache: dict | None = None


def load_model(path: Path = BEST_MODEL_PATH) -> dict:
    global _bundle_cache
    if _bundle_cache is None:
        _bundle_cache = joblib.load(path)
    return _bundle_cache


def _key(text: str) -> str:
    """Same normalisation as wrangling step 1: trim spaces/commas, collapse spaces, ignore case."""
    return re.sub(r"\s+", " ", str(text).strip().strip(",").strip()).lower()


def predict_price(location: str, sqft: float, bath: int, bhk: int,
                  area_type: str = "Super built-up Area") -> float:
    """Predicted price in lakhs for one property, using outputs/best_model.joblib.

    A location the model never saw (or one grouped as rare) is treated as "other".
    """
    bundle = load_model()
    if sqft <= 0 or bath < 1 or bhk < 1:
        raise ValueError("sqft must be > 0, bath and bhk must be >= 1")

    areas = {_key(a): a for a in bundle["area_types"]}
    if _key(area_type) not in areas:
        raise ValueError(f"area_type must be one of {bundle['area_types']}")
    locations = {_key(loc): loc for loc in bundle["locations"]}
    loc = locations.get(_key(location))
    if loc is None:
        log.warning("Location %r not in the training data; using 'other'", location)
        loc = "other"
    if sqft / bhk < 300 or bath > bhk + 2:
        log.warning("Input breaks the cleaning rules (sqft/bhk < 300 or bath > bhk + 2); "
                    "the model never saw such listings, so treat the prediction with caution")

    row = pd.DataFrame([{"location": loc, "area_type": areas[_key(area_type)],
                         "total_sqft": float(sqft), "bath": bath, "bhk": bhk}])[bundle["features"]]
    return round(float(bundle["model"].predict(row)[0]), 2)


# ──────────────────────────────────────────────────────────────────────────────
# REPORT
# ──────────────────────────────────────────────────────────────────────────────
def explain_v1_gap(v1v2: pd.DataFrame) -> list[str]:
    """Plain-language explanation of each step's contribution, computed from the table."""
    r = v1v2.set_index(v1v2["step"].str[0])

    def change(a: str, b: str) -> str:
        """'test R² x → y, MAE x → y (better/worse)' between two steps."""
        better = r.at[b, "test_mae"] < r.at[a, "test_mae"]
        return (f"test R² {r.at[a, 'test_r2']:.3f} → {r.at[b, 'test_r2']:.3f}, MAE {r.at[a, 'test_mae']:.2f} → "
                f"{r.at[b, 'test_mae']:.2f} lakhs ({'error falls' if better else 'error rises'} by "
                f"{abs(1 - r.at[b, 'test_mae'] / r.at[a, 'test_mae']):.0%})")

    return [
        f"Overall, v1 as shipped → final v2 model: {change('0', '4')}.",
        f"**Step 1, wrangling** (same 3 features, same LinearRegression): {change('0', '1')}. Caution: this step "
        "changes which rows are scored, not just their quality. Removing impossible listings also removes some of "
        "the hardest cases, so part of this change is an easier test set rather than a better model. Steps 2–4 "
        "use exactly the same rows, split and folds, so their changes are a fair comparison.",
        f"**Step 2, location** (one-hot location and area_type): {change('1', '2')}. v1 ignored location "
        "entirely, yet the same flat can cost several times more in a premium area.",
        f"**Step 3, log-log** (log_price target, log total_sqft): {change('2', '3')}. On the log scale each "
        "feature acts as a percentage change in price, which matches how prices behave, and the few very "
        "expensive listings stop dominating the squared-error fit."
        + (" R² moves the other way: exp(predicted log price) estimates the *median* price for given features, "
           "not the mean, so the most expensive listings are under-predicted, and R²/RMSE square exactly those "
           "large errors. MAE, the typical error, is the better guide to what a buyer would experience."
           if (r.at["3", "test_mae"] < r.at["2", "test_mae"]) != (r.at["3", "test_r2"] > r.at["2", "test_r2"]) else ""),
        f"**Step 4, non-linear model + tuning**: {change('3', '4')}. Trees can learn interactions a linear "
        "model cannot, e.g. that an extra 1,000 sq ft is worth far more in one location than in another.",
    ]


def write_report(path: Path, clean, X_train, X_test, comparison, effect, top2, tuning, final, v1v2, examples) -> Path:
    cols = ["model", "target", "cv_r2_mean", "cv_r2_std", "cv_mae_mean", "cv_mae_std",
            "cv_rmse_mean", "cv_rmse_std", "fit_seconds"]
    lines = [
        "# Modelling report",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `src/modeling.py` from `outputs/clean_data.csv`.",
        "",
        "## Setup",
        "",
        f"- Data: {len(clean):,} clean listings. Features: {', '.join(f'`{c}`' for c in FEATURES)}. Target: `price` (lakhs).",
        f"- Hold-out test set: {len(X_test):,} rows ({TEST_SIZE:.0%}, `random_state={RANDOM_STATE}`), "
        "used only for the final scores. Everything else uses the "
        f"{len(X_train):,} training rows.",
        f"- {N_FOLDS}-fold cross-validation (shuffled, `random_state={RANDOM_STATE}`), identical folds for every model.",
        "- Pipeline: `ColumnTransformer` (OneHotEncoder for location/area_type, StandardScaler for "
        "total_sqft/bath/bhk) → model, all inside the CV so encoders and scalers never see the validation fold.",
        "- `log_price` variant: a log-log specification. `TransformedTargetRegressor(func=np.log, "
        "inverse_func=np.exp)` makes the model learn log(price) while predictions and metrics stay in lakhs, and "
        "`total_sqft` is logged before scaling. (Log price with raw sqft made the linear models predict "
        "exponential growth with area and explode on the largest listings: CV R² ≈ −3,900 in the first run.)",
        "- MAE and RMSE are in lakhs. `price_per_sqft` is not a feature (it is derived from the target).",
        "",
        "## 1. Model comparison (5-fold CV, mean ± std across folds)",
        "",
        md_table(comparison[cols]),
        "",
        "## 2. Does predicting log_price help?",
        "",
        md_table(effect),
        "",
        f"log_price improves R² for {(effect['r2_change'] > 0).sum()} of {len(effect)} models and lowers MAE for "
        f"{(effect['mae_change'] < 0).sum()} of {len(effect)}.",
        "",
        "## 3. Tuning the top 2 (GridSearchCV, same folds, refit on best mean R²)",
        "",
        md_table(top2[["model", "target", "cv_r2_mean", "cv_mae_mean"]].rename(
            columns={"cv_r2_mean": "untuned_cv_r2", "cv_mae_mean": "untuned_cv_mae"})),
        "",
    ]
    for name, table in tuning.groupby("model", sort=False):
        lines += [f"**{name}: top 5 of {len(table)} parameter combinations**", "",
                  md_table(table.head(5).drop(columns="model")), ""]
    lines += [
        "## 4. Final model",
        "",
        f"Chosen by cross-validated R² (not by the test set): **{final['model']}** with target `{final['target']}`, "
        f"parameters `{final['params']}`.",
        "",
        md_table(pd.DataFrame([final["scores"]])),
        "",
        "The saved `outputs/best_model.joblib` is this configuration refitted on all "
        f"{len(clean):,} clean rows; the test scores above come from the fit on the training rows only.",
        "",
        "## 5. v1 vs v2",
        "",
        md_table(v1v2),
        "",
        *[f"- {line}" for line in explain_v1_gap(v1v2)],
        "",
        "## 6. `predict_price()` examples",
        "",
        md_table(examples),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def run_modeling(clean: pd.DataFrame | None = None) -> dict:
    clean = load_clean() if clean is None else clean
    X_train, X_test, y_train, y_test = split(clean)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("1. Comparing %d models x 2 targets with %d-fold CV", len(candidate_models()), N_FOLDS)
    comparison = compare_models(X_train, y_train)
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    effect = log_target_effect(comparison)

    top2 = top_models(comparison)
    log.info("2. Tuning the top 2: %s", ", ".join(f"{m} ({t})" for m, t in zip(top2["model"], top2["target"])))
    searches, tables = {}, []
    for _, row in top2.iterrows():
        searches[row["model"]], table = tune(row["model"], row["target"] == "log_price", X_train, y_train)
        tables.append(table)
    tuning = pd.concat(tables, ignore_index=True)
    tuning.to_csv(REPORTS_DIR / "tuning_results.csv", index=False)

    best_name = max(searches, key=lambda m: searches[m].best_score_)
    best = searches[best_name]
    target = top2.set_index("model").at[best_name, "target"]
    scores = test_scores(best.best_estimator_, X_test, y_test)
    log.info("Final model %s (%s): CV R² %.4f, test R² %.4f, test MAE %.2f lakhs",
             best_name, target, best.best_score_, scores["test_r2"], scores["test_mae"])
    X_test.assign(actual=y_test, predicted=best.best_estimator_.predict(X_test)).to_csv(
        OUTPUTS_DIR / "test_predictions.csv", index=False)
    final = {"model": best_name, "target": target, "params": short_params(best.best_params_),
             "scores": {"cv_r2": best.best_score_, **scores}}

    log.info("3. Comparing with v1")
    v1v2 = compare_with_v1(clean, X_train, X_test, y_train, y_test, best.best_estimator_)
    v1v2.to_csv(REPORTS_DIR / "v1_vs_v2.csv", index=False)
    for line in explain_v1_gap(v1v2):
        log.info("    %s", line.replace("**", ""))

    log.info("4. Refitting %s on all clean rows and saving", best_name)
    deployed = make_pipeline(candidate_models()[best_name], target == "log_price")
    deployed.set_params(**best.best_params_).fit(clean[FEATURES], clean[TARGET])
    save_bundle(deployed, clean, {"model_name": best_name, "target": target, "params": final["params"],
                                  "best_params": best.best_params_, "scores": final["scores"]})

    global _bundle_cache
    _bundle_cache = None  # make predict_price() pick up the file just written
    examples = pd.DataFrame([
        {"location": loc, "sqft": s, "bath": ba, "bhk": bh, "area_type": at}
        for loc, s, ba, bh, at in [
            ("Whitefield", 1200, 2, 2, "Super built-up Area"),
            ("Indira Nagar", 1200, 2, 2, "Super built-up Area"),
            ("Electronic City", 1200, 2, 2, "Super built-up Area"),
            ("Electronic City", 2000, 3, 3, "Super built-up Area"),
            ("Some New Layout", 1500, 2, 3, "Built-up Area"),
        ]])
    examples["predicted_lakhs"] = [predict_price(**r) for r in examples.to_dict("records")]

    write_report(REPORTS_DIR / "modeling_report.md", clean, X_train, X_test,
                 comparison, effect, top2, tuning, final, v1v2, examples)
    log.info("Saved reports/model_comparison.csv, tuning_results.csv, v1_vs_v2.csv, modeling_report.md")
    return {"comparison": comparison, "tuning": tuning, "final": final, "v1_vs_v2": v1v2}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_modeling()


if __name__ == "__main__":
    main()
