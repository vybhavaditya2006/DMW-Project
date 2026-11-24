"""
Feature importance and selection
================================
For the best model from src/modeling.py (same type, target and tuned settings):

  1. Built-in importance: XGBoost's gain share per encoded column, with the
     location_* and area_type_* dummies summed back into one feature each.
  2. Permutation importance on the held-out test set: shuffle one input
     column and measure how much R² falls / MAE rises. Shuffling the raw
     `location` column moves all its dummies together, so location is
     scored as a single feature.
  3. Feature selection: SelectKBest (univariate F-test) and RFE (recursive
     elimination with a smaller XGBoost) for several k, placed inside the
     pipeline so the selection is redone in every CV training fold.

Outputs:
  reports/feature_importance.csv          aggregated built-in + permutation importance
  reports/feature_importance_columns.csv  built-in importance of the top individual encoded columns
  reports/feature_selection.csv           CV accuracy for each method and k
  reports/features_report.md              the above, explained
Figures are drawn from these files by src/visualize.py.

Usage:
    python -m src.features
"""

import ast
import logging
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src import PLOTS, RANDOM_STATE, REPORTS_DIR
from src.modeling import (BEST_MODEL_PATH, CATEGORICAL, FEATURES, candidate_models, cv_scores,
                          load_clean, make_preprocessor, split, test_scores)
from src.wrangling import md_table

log = logging.getLogger(__name__)

PERMUTATION_REPEATS = 10
SELECTION_KS = [3, 5, 10, 25, 50, 100]
# "Similar accuracy": CV R² at most 0.01 below and CV MAE at most 2% above the all-features model.
R2_TOLERANCE, MAE_TOLERANCE = 0.01, 0.02


# ──────────────────────────────────────────────────────────────────────────────
# THE BEST MODEL'S CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
def parse_params(text: str) -> dict:
    """'learning_rate=0.03, max_depth=8' -> {'regressor__model__learning_rate': 0.03, ...}"""
    params = {}
    for part in text.split(", "):
        key, value = part.split("=", 1)
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        params[f"regressor__model__{key}"] = value
    return params


def best_config(path: Path = BEST_MODEL_PATH) -> tuple[str, bool, dict]:
    """Model name, whether it uses the log target, and its tuned parameters, from the saved bundle."""
    bundle = joblib.load(path)
    params = bundle.get("best_params") or parse_params(bundle["params"])
    return bundle["model_name"], bundle["target"] == "log_price", params


def build_model(selector=None) -> TransformedTargetRegressor:
    """The best model's pipeline, optionally with a feature selector between preprocessing and model."""
    name, log_target, params = best_config()
    steps = [("prep", make_preprocessor(log_sqft=log_target))]
    if selector is not None:
        steps.append(("select", selector))
    steps.append(("model", candidate_models()[name]))
    est = (TransformedTargetRegressor(regressor=Pipeline(steps), func=np.log, inverse_func=np.exp)
           if log_target else TransformedTargetRegressor(regressor=Pipeline(steps)))
    return est.set_params(**params)


def original_feature(encoded: str) -> str:
    """'cat__location_Whitefield' -> 'location', 'log_sqft__total_sqft' -> 'total_sqft'."""
    name = encoded.split("__", 1)[1]
    for col in CATEGORICAL:
        if name.startswith(f"{col}_"):
            return col
    return name


# ──────────────────────────────────────────────────────────────────────────────
# IMPORTANCE
# ──────────────────────────────────────────────────────────────────────────────
def builtin_importance(fitted) -> tuple[pd.Series, pd.DataFrame]:
    """XGBoost gain share per encoded column, and summed per original feature."""
    names = fitted.regressor_.named_steps["prep"].get_feature_names_out()
    gain = fitted.regressor_.named_steps["model"].feature_importances_
    columns = pd.DataFrame({"column": [n.split("__", 1)[1] for n in names],
                            "feature": [original_feature(n) for n in names],
                            "gain_share": gain}).sort_values("gain_share", ascending=False)
    return columns.groupby("feature")["gain_share"].sum(), columns


def permutation(fitted, X_test, y_test) -> pd.DataFrame:
    """Mean and std (over repeats) of the drop in R² and the rise in MAE when one input column is shuffled."""
    res = permutation_importance(fitted, X_test, y_test, scoring=["r2", "neg_mean_absolute_error"],
                                 n_repeats=PERMUTATION_REPEATS, random_state=RANDOM_STATE, n_jobs=1)
    return pd.DataFrame({
        "perm_r2_drop": res["r2"].importances_mean,
        "perm_r2_drop_std": res["r2"].importances_std,
        # sklearn's score is −MAE, so the importance (score drop) equals the MAE increase in lakhs
        "perm_mae_increase": res["neg_mean_absolute_error"].importances_mean,
        "perm_mae_increase_std": res["neg_mean_absolute_error"].importances_std,
    }, index=pd.Index(X_test.columns, name="feature"))


# ──────────────────────────────────────────────────────────────────────────────
# SELECTION
# ──────────────────────────────────────────────────────────────────────────────
def rfe_selector(k: int) -> RFE:
    """RFE ranks columns with a smaller, faster XGBoost and removes 10% of them per round."""
    ranker = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=RANDOM_STATE)
    return RFE(ranker, n_features_to_select=k, step=0.1)


def kept_features(fitted) -> str:
    """Which original features survive the selector, e.g. 'location (7 dummies), total_sqft, bath'."""
    pipe = fitted.regressor_
    names = pipe.named_steps["prep"].get_feature_names_out()[pipe.named_steps["select"].get_support()]
    feats = pd.Series([original_feature(n) for n in names]).value_counts()
    return ", ".join(f"{f} ({feats[f]} dummies)" if f in CATEGORICAL else f for f in FEATURES if f in feats)


def feature_selection(X_train, y_train, n_columns: int) -> pd.DataFrame:
    rows = []
    start = time.perf_counter()
    rows.append({"method": "all features", "k": n_columns, **cv_scores(build_model(), X_train, y_train),
                 "kept": "everything"})
    log.info("All %d columns: CV R² %.4f, MAE %.2f", n_columns, rows[-1]["cv_r2_mean"], rows[-1]["cv_mae_mean"])
    for method, make in [("SelectKBest (f_regression)", lambda k: SelectKBest(f_regression, k=k)),
                         ("RFE (XGBoost ranker)", rfe_selector)]:
        for k in SELECTION_KS:
            est = build_model(make(k))
            scores = cv_scores(est, X_train, y_train)
            kept = kept_features(est.fit(X_train, y_train))
            rows.append({"method": method, "k": k, **scores, "kept": kept})
            log.info("%-26s k=%3d: CV R² %.4f ± %.4f, MAE %.2f (%s)", method, k,
                     scores["cv_r2_mean"], scores["cv_r2_std"], scores["cv_mae_mean"], kept)
    log.info("Feature selection took %.0fs", time.perf_counter() - start)
    return pd.DataFrame(rows)


def smallest_similar(selection: pd.DataFrame) -> pd.DataFrame:
    """For each method, the smallest k whose CV accuracy is similar to using all features."""
    base = selection.iloc[0]
    ok = selection[(selection["cv_r2_mean"] >= base["cv_r2_mean"] - R2_TOLERANCE)
                   & (selection["cv_mae_mean"] <= base["cv_mae_mean"] * (1 + MAE_TOLERANCE))
                   & (selection["method"] != "all features")]
    return ok.loc[ok.groupby("method")["k"].idxmin()] if len(ok) else ok


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def run_features(clean: pd.DataFrame | None = None) -> dict:
    clean = load_clean() if clean is None else clean
    X_train, X_test, y_train, y_test = split(clean)
    name, log_target, params = best_config()
    log.info("Best model: %s (log target: %s) with %s", name, log_target, params)

    fitted = build_model().fit(X_train, y_train)
    test = test_scores(fitted, X_test, y_test)
    log.info("Refitted on the training rows: test R² %.4f, MAE %.2f", test["test_r2"], test["test_mae"])

    builtin, columns = builtin_importance(fitted)
    perm = permutation(fitted, X_test, y_test)
    importance = perm.join(builtin.rename("gain_share")).sort_values("perm_r2_drop", ascending=False)
    importance.insert(0, "gain_share", importance.pop("gain_share"))
    importance.to_csv(REPORTS_DIR / "feature_importance.csv")
    columns.head(25).to_csv(REPORTS_DIR / "feature_importance_columns.csv", index=False)
    for f, r in importance.iterrows():
        log.info("%-11s gain %.3f | permutation: R² drop %.3f ± %.3f, MAE +%.2f lakhs",
                 f, r["gain_share"], r["perm_r2_drop"], r["perm_r2_drop_std"], r["perm_mae_increase"])

    n_columns = len(columns)
    n_locations = int((columns["feature"] == "location").sum())
    selection = feature_selection(X_train, y_train, n_columns)
    selection.to_csv(REPORTS_DIR / "feature_selection.csv", index=False)
    similar = smallest_similar(selection)

    write_report(REPORTS_DIR / "features_report.md", name, log_target, len(X_train), test, importance, columns,
                 selection, similar, n_columns, n_locations)
    log.info("Saved reports/feature_importance.csv, feature_importance_columns.csv, feature_selection.csv, "
             "features_report.md")
    return {"importance": importance, "selection": selection, "similar": similar}


def write_report(path, name, log_target, n_train, test, importance, columns, selection, similar,
                 n_columns, n_locations) -> Path:
    imp = importance.copy()
    perm_rank = imp["perm_r2_drop"].rank(ascending=False).astype(int)
    gain_rank = imp["gain_share"].rank(ascending=False).astype(int)
    disagree = [f for f in imp.index if perm_rank[f] != gain_rank[f]]
    base = selection.iloc[0]
    lines = [
        "# Feature importance and selection",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `src/features.py`.",
        "",
        f"Model: the tuned **{name}** from `src/modeling.py` ({'log-log' if log_target else 'price'} target), "
        f"refitted on the same {n_train:,} training rows; held-out test R² {test['test_r2']:.3f}, "
        f"MAE {test['test_mae']:.2f} lakhs. The 5 input features become {n_columns} encoded columns "
        f"({n_locations} location dummies, {int((columns['feature'] == 'area_type').sum())} area_type dummies, "
        "3 numbers).",
        "",
        "## 1. Importance per feature",
        "",
        md_table(imp.reset_index()),
        "",
        "- **gain_share**: XGBoost's built-in importance, the share of the total loss reduction from splits on "
        "that feature, with dummies summed. It is measured on the training data and only says how much the "
        "trees *used* a feature.",
        f"- **perm_r2_drop / perm_mae_increase**: shuffle one column of the held-out test set "
        f"{PERMUTATION_REPEATS} times and measure how much worse the predictions get (MAE in lakhs). This is "
        "how much the model *needs* the feature to predict unseen listings.",
        "",
        (f"The two methods rank the features differently ({', '.join(disagree)}). " if disagree else
         "Both methods give the same ranking. ")
        + "Permutation importance is the more reliable guide: gain is computed on training data and favours "
        "features with many possible split points. Note also that total_sqft, bhk and bath are correlated: when "
        "one is shuffled the others still carry part of its information, so permutation understates each of "
        "them individually.",
        "",
        f"![importance](../outputs/plots_v2/{PLOTS['importance']})",
        "",
        "Top encoded columns by built-in gain (before summing dummies):",
        "",
        md_table(columns.head(15)),
        "",
        "## 2. Feature selection",
        "",
        "Both selectors sit *inside* the pipeline, between preprocessing and the model, so in each CV fold they "
        "choose columns from the training folds only. **SelectKBest** keeps the k columns with the strongest "
        "univariate linear relationship with (log) price (F-test). **RFE** fits a smaller XGBoost, drops the "
        "10% least important columns, and repeats until k remain, so it accounts for interactions. "
        f"k counts encoded columns (out of {n_columns}).",
        "",
        md_table(selection.drop(columns="fit_seconds")),
        "",
        (f"\"Similar accuracy\" here means CV R² no more than {R2_TOLERANCE} below and CV MAE no more than "
         f"{MAE_TOLERANCE:.0%} above the all-features model (R² {base['cv_r2_mean']:.4f}, "
         f"MAE {base['cv_mae_mean']:.2f}). "
         + ("Smallest k that qualifies: " + "; ".join(
             f"**{r['method']}: k = {r['k']}** (R² {r['cv_r2_mean']:.4f}, MAE {r['cv_mae_mean']:.2f}; keeps "
             f"{r['kept']})" for _, r in similar.iterrows()) + "."
            if len(similar) else "No tested k qualifies: every reduction costs noticeable accuracy.")),
        "",
        f"![selection](../outputs/plots_v2/{PLOTS['selection']})",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_features()


if __name__ == "__main__":
    main()
