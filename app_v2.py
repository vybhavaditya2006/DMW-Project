"""
PropIntel Bengaluru v2: data-mining & data-wrangling dashboard
==============================================================
Shows the results of run_pipeline.py, tab by tab, in the order of the pipeline.
Everything is loaded from files in outputs/ and reports/: nothing is trained or
recomputed when the page loads (the only model call is predict_price() for the
predictor, which loads outputs/best_model.joblib once).

Usage:
    python run_pipeline.py          # once, to produce the outputs
    streamlit run app_v2.py
"""

import ast
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Make `src` importable however the app is launched (streamlit run from another folder, AppTest, ...).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import FIGURES_DIR, OUTPUTS_DIR, PLOTS, RAW_CSV, REPORTS_DIR, ROOT

st.set_page_config(
    page_title="PropIntel Bengaluru v2",
    layout="wide",
    page_icon="https://img.icons8.com/ios-filled/50/6366f1/department.png",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────
# STYLING: reuse v1's CSS without importing (and so running) app.py
# ─────────────────────────────────────────────────────────────
@st.cache_data
def v1_css() -> str:
    """Read the PROFESSIONAL_CSS string literal out of app.py's source."""
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "PROFESSIONAL_CSS" for t in node.targets):
            return ast.literal_eval(node.value)
    return ""


st.markdown(v1_css(), unsafe_allow_html=True)


def section(title: str, subtitle: str = "") -> None:
    st.markdown(f'<p class="section-title">{title}</p>'
                + (f'<p class="section-subtitle">{subtitle}</p>' if subtitle else ""), unsafe_allow_html=True)


def what_it_does(text: str) -> None:
    """The short demo note at the top of each tab."""
    st.markdown(f'<div class="clean-card" style="padding:1.1rem 1.5rem; border-left:5px solid #6366f1;">'
                f'<p style="margin:0; font-size:0.95rem;"><b>What this step does.</b> {text}</p></div>',
                unsafe_allow_html=True)


def metric_row(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.markdown(f'<div class="metric-box"><h3>{label}</h3><p>{value}</p></div>', unsafe_allow_html=True)


def show_plot(key: str, caption: str = "") -> None:
    path = FIGURES_DIR / PLOTS[key]
    if path.exists():
        st.image(str(path), caption=caption or None, width="stretch")
    else:
        st.warning(f"`outputs/plots_v2/{PLOTS[key]}` is missing. Run `python run_pipeline.py` (or "
                   "`python -m src.visualize`) to create it.")


# ─────────────────────────────────────────────────────────────
# DATA LOADING (cached; files only)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def read_csv(path: Path, **kw) -> pd.DataFrame | None:
    return pd.read_csv(path, **kw) if path.exists() else None


def report(name: str, **kw) -> pd.DataFrame | None:
    return read_csv(REPORTS_DIR / name, **kw)


def output(name: str, **kw) -> pd.DataFrame | None:
    return read_csv(OUTPUTS_DIR / name, **kw)


def require(df: pd.DataFrame | None, name: str) -> bool:
    if df is None:
        st.warning(f"`{name}` is missing. Run `python run_pipeline.py` to create it.")
        return False
    return True


@st.cache_resource
def model_bundle() -> dict | None:
    import joblib
    path = OUTPUTS_DIR / "best_model.joblib"
    return joblib.load(path) if path.exists() else None


@st.cache_data
def quality_problems() -> pd.DataFrame:
    """Data-quality problems in the raw CSV (simple counts; no model involved)."""
    from src.wrangling import parse_sqft

    raw = pd.read_csv(RAW_CSV)
    how = raw["total_sqft"].map(lambda v: parse_sqft(v)[1])
    sqft = raw["total_sqft"].map(lambda v: parse_sqft(v)[0])
    bhk = raw["size"].str.extract(r"(\d+)")[0].astype(float)
    loc = raw["location"].dropna()
    rows = [
        ("Exact duplicate rows", raw.duplicated().sum(), "Dropped (step 1)"),
        ("society missing", raw["society"].isna().sum(), "Column dropped: 41% missing, ~2,700 names (step 5)"),
        ("balcony missing", raw["balcony"].isna().sum(), "Median of same-bhk listings (step 5)"),
        ("bath missing", raw["bath"].isna().sum(), "Median of same-bhk listings (step 5)"),
        ("size missing (all bare land plots)", raw["size"].isna().sum(), "Rows dropped (step 2)"),
        ("location missing", raw["location"].isna().sum(), "Mode of the same society (step 5)"),
        ("total_sqft given as a range, e.g. '2100 - 2850'", (how == "range (midpoint)").sum(), "Midpoint (step 3)"),
        ("total_sqft in other units (Sq. Meter, Acres, ...)", how.str.startswith("unit").sum(),
         "Converted to sq ft (step 3)"),
        ("Location with stray spaces or case variants", (loc != loc.str.strip()).sum(), "Normalised (step 1)"),
        ("Every area_type has a double space", raw["area_type"].str.contains("  ").sum(), "Normalised (step 1)"),
        ("Less than 300 sq ft per bedroom", (sqft / bhk < 300).sum(), "Removed as outliers (step 7)"),
        ("More than 10 bedrooms", (bhk > 10).sum(), "Kept unless another rule flags them (step 7)"),
    ]
    return pd.DataFrame(rows, columns=["problem", "rows", "how v2 handles it"])


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <h2 style="margin:0; font-weight:800; color:#f8fafc; font-size: 1.2rem; letter-spacing: 0.5px;
                   text-transform: uppercase;">Pipeline v2</h2>
    </div>""", unsafe_allow_html=True)
    st.markdown("**Data Mining & Data Wrangling** on 13,320 Bengaluru property listings.")
    st.divider()
    st.markdown("##### Pipeline")
    for i, step in enumerate(["Profile raw data", "Wrangle (8 logged steps)", "Model & tune",
                              "Cluster locations", "Mine association rules", "Feature importance"], 1):
        st.markdown(f"{i}. {step}")
    st.divider()
    bundle = model_bundle()
    if bundle:
        st.caption(f"Model trained: {bundle['created'].replace('T', ' ')}")
    st.caption("All results are precomputed by `python run_pipeline.py`; this page only reads them.")


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 30px 0 10px 0;">
    <h1 style="font-size:3.2rem; font-weight:900; margin:0;
               background: linear-gradient(135deg, #1e293b 0%, #4f46e5 100%);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
               letter-spacing: -2px; text-transform: uppercase;">PropIntel Bengaluru v2</h1>
    <p style="color:#64748b; font-size:1rem; font-weight: 500; margin-top:10px; text-transform: uppercase;
              letter-spacing: 2px;">Data Wrangling · Data Mining · Explained step by step
    <span style="display:block; width: 60px; height: 3px; background: #6366f1; margin: 20px auto;
                 border-radius: 2px;"></span></p>
</div>""", unsafe_allow_html=True)

tabs = st.tabs(["RAW DATA", "WRANGLING", "PRICE PREDICTION", "MARKET SEGMENTS", "ASSOCIATION RULES",
                "FEATURE IMPORTANCE"])


# ═══════════════════════════════════════════════════════════════
# TAB 1: RAW DATA
# ═══════════════════════════════════════════════════════════════
with tabs[0]:
    section("Raw data audit", "Bengaluru_House_Data.csv exactly as downloaded, before any cleaning")
    what_it_does("Profiles the raw CSV before touching it: types, missing values, duplicates and formatting "
                 "problems. Every cleaning decision in the next tab is a response to a problem found here.")
    raw = read_csv(RAW_CSV)
    problems = quality_problems()
    metric_row([("Listings", f"{len(raw):,}"), ("Columns", f"{raw.shape[1]}"),
                ("Missing cells", f"{int(raw.isna().sum().sum()):,}"),
                ("Duplicate rows", f"{int(raw.duplicated().sum()):,}")])
    st.write("")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown("#### Data-quality problems found")
        st.dataframe(problems, hide_index=True, width="stretch",
                     column_config={"rows": st.column_config.NumberColumn(format="%d")})
    with c2:
        st.markdown("#### Missing values per column")
        miss = raw.isna().sum().rename("missing").to_frame().assign(
            pct=lambda t: (t["missing"] / len(raw) * 100).round(1)).sort_values("missing", ascending=False)
        st.dataframe(miss, width="stretch",
                     column_config={"pct": st.column_config.ProgressColumn("% missing", format="%.1f%%",
                                                                           min_value=0, max_value=100)})
    st.markdown("#### Where the gaps are (missingno)")
    m1, m2 = st.columns([1.4, 1])
    with m1:
        show_plot("missingno_matrix", "Each row of pixels is a listing; white = missing.")
    with m2:
        show_plot("missingno_heatmap", "Nullity correlation: do two columns tend to be missing together?")
    with st.expander("Sample of raw rows"):
        st.dataframe(raw.head(20), width="stretch")
    profile = REPORTS_DIR / "raw_profile.html"
    if profile.exists():
        with st.expander("Full ydata-profiling report (large; opens inline)"):
            if st.toggle("Load the report", key="load_profile"):
                components.html(profile.read_text(encoding="utf-8"), height=900, scrolling=True)
        st.download_button("Download raw_profile.html", profile.read_bytes(), file_name="raw_profile.html",
                           mime="text/html")


# ═══════════════════════════════════════════════════════════════
# TAB 2: WRANGLING
# ═══════════════════════════════════════════════════════════════
with tabs[1]:
    section("Wrangling pipeline", "Eight small, logged steps from the raw CSV to outputs/clean_data.csv")
    what_it_does("Cleans the data one documented step at a time. Each step records the rows and nulls before "
                 "and after, plus the reason for its decision, so every removed or changed row can be explained. "
                 "Outliers are flagged by nine methods side by side before choosing which to remove.")
    wlog = report("wrangling_log.csv")
    if require(wlog, "reports/wrangling_log.csv"):
        metric_row([("Raw rows", f"{wlog['rows_before'].iloc[0]:,}"),
                    ("Clean rows", f"{wlog['rows_after'].iloc[-1]:,}"),
                    ("Removed", f"{wlog['rows_removed'].sum():,}"),
                    ("Logged steps", f"{len(wlog)}")])
        st.write("")
        st.markdown("#### Step-by-step log")
        st.dataframe(wlog[["step", "title", "rows_before", "rows_after", "rows_removed", "nulls_before",
                           "nulls_after", "columns_added", "columns_dropped"]], hide_index=True, width="stretch")
        pick = st.selectbox("Explain a step", wlog["step"], format_func=lambda s: f"Step {s}: "
                            + wlog.set_index("step").at[s, "title"])
        row = wlog.set_index("step").loc[pick]
        st.markdown(f"**Decision:** {row['decision']}")
        notes = [n for n in str(row["notes"]).split(" | ") if n and n != "nan"]
        if notes:
            st.markdown("\n".join(f"- {n}" for n in notes))
        st.markdown("#### Rows kept at each step")
        show_plot("waterfall")
        st.markdown("#### Before vs after")
        show_plot("distributions")
        st.markdown("#### Outlier detection: nine methods compared")
        show_plot("outliers")
        flags = output("outlier_flags.csv")
        if flags is not None:
            from src.wrangling import FINAL_OUTLIER_RULES, OUTLIER_METHODS
            f = flags.drop(columns="raw_row").astype(bool)
            st.dataframe(pd.DataFrame({
                "method": list(OUTLIER_METHODS), "family": [v[0] for v in OUTLIER_METHODS.values()],
                "rule": [v[1] for v in OUTLIER_METHODS.values()],
                "flagged": [int(f[m].sum()) for m in OUTLIER_METHODS],
                "used for removal": ["✓" if m in FINAL_OUTLIER_RULES else "" for m in OUTLIER_METHODS],
            }), hide_index=True, width="stretch")
        clean = output("clean_data.csv")
        if clean is not None:
            with st.expander("Sample of clean rows (outputs/clean_data.csv)"):
                st.dataframe(clean.head(20), width="stretch")


# ═══════════════════════════════════════════════════════════════
# TAB 3: PRICE PREDICTION
# ═══════════════════════════════════════════════════════════════
with tabs[2]:
    section("Price prediction", "Seven models compared with 5-fold cross-validation, the best two tuned")
    what_it_does("Trains seven regression models on 80% of the clean data and compares them with 5-fold "
                 "cross-validation, each on price and on log price. The two best are tuned with GridSearchCV; the "
                 "winner is scored once on the untouched 20% test set and compared with the v1 model.")
    comp, v1v2 = report("model_comparison.csv"), report("v1_vs_v2.csv")
    if bundle and v1v2 is not None:
        s = bundle["scores"]
        v1 = v1v2.iloc[0]
        metric_row([("Best model", f"{bundle['model_name']}"), ("Test R²", f"{s['test_r2']:.3f}"),
                    ("Test MAE", f"₹{s['test_mae']:.1f} L"),
                    ("Error vs v1", f"−{1 - s['test_mae'] / v1['test_mae']:.0%}")])
        st.write("")

    st.markdown("#### Try the model")
    if bundle is None:
        st.warning("`outputs/best_model.joblib` is missing. Run `python run_pipeline.py` to create it.")
    else:
        from src.modeling import predict_price
        locations = [l for l in bundle["locations"] if l != "other"]
        with st.form("predict"):
            c1, c2, c3, c4, c5 = st.columns([2, 1.5, 1, 1, 1])
            loc = c1.selectbox("Location", locations + ["other"],
                               index=locations.index("Whitefield") if "Whitefield" in locations else 0,
                               format_func=lambda l: "Other / not listed" if l == "other" else l)
            area = c2.selectbox("Area type", bundle["area_types"],
                                index=bundle["area_types"].index("Super built-up Area")
                                if "Super built-up Area" in bundle["area_types"] else 0)
            sqft = c3.number_input("Area (sq ft)", 300, 30000, 1200, step=50)
            bhk = c4.number_input("BHK", 1, 20, 2)
            bath = c5.number_input("Bathrooms", 1, 20, 2)
            go = st.form_submit_button("PREDICT PRICE", type="primary")
        if go:
            if sqft / bhk < 300 or bath > bhk + 2:
                st.warning("These inputs break the cleaning rules (under 300 sq ft per bedroom, or more than "
                           "bhk + 2 bathrooms). The model never saw such listings; treat the result with caution.")
            price = predict_price(loc, sqft, bath, bhk, area)
            pred = output("test_predictions.csv")
            band = ""
            if pred is not None:
                pct = (pred["actual"] - pred["predicted"]) / pred["predicted"]
                lo, hi = price * (1 + pct.quantile(0.25)), price * (1 + pct.quantile(0.75))
                band = (f"<p style='color:#475569; margin-top:14px;'>Likely range <b>₹{lo:,.1f} L – "
                        f"₹{hi:,.1f} L</b>: half of the held-out test listings had an asking price within this "
                        f"band around their prediction (the middle 50% of test errors).</p>")
            clean = output("clean_data.csv")
            context = ""
            if clean is not None and loc != "other":
                same = clean[(clean["location"] == loc) & (clean["bhk"] == bhk)]
                if len(same):
                    context = (f"<p style='color:#475569;'>{len(same)} clean listings of {bhk} BHK in {loc}: "
                               f"median ₹{same['price'].median():,.1f} L, median {same['total_sqft'].median():,.0f} "
                               "sq ft.</p>")
            st.markdown(f"""
            <div class="clean-card" style="border-left:5px solid #6366f1;">
              <p style="color:#64748b; font-size:0.75rem; font-weight:800; text-transform:uppercase;
                        letter-spacing:3px; margin-bottom:10px;">Predicted price</p>
              <h1 style="font-size:3.5rem; font-weight:900; margin:0; color:#0f172a; letter-spacing:-2px;">
                ₹ {price:,.2f} <span style="font-size:1.2rem; color:#94a3b8; letter-spacing:0;">Lakhs</span></h1>
              <p style="color:#475569;">₹ {price * 1e5 / sqft:,.0f} per sq ft · {bundle['model_name']} trained on
                 {bundle['target'].replace('_', ' ')}</p>
              {band}{context}
            </div>""", unsafe_allow_html=True)

    if require(comp, "reports/model_comparison.csv"):
        st.markdown("#### Model comparison (5-fold CV on the training split)")
        show = comp[["model", "target", "cv_r2_mean", "cv_r2_std", "cv_mae_mean", "cv_mae_std",
                     "cv_rmse_mean", "cv_rmse_std"]].copy()
        st.dataframe(show.style.format({c: "{:.3f}" for c in show.columns if "r2" in c}
                                       | {c: "{:.2f}" for c in show.columns if "mae" in c or "rmse" in c})
                     .highlight_max(subset=["cv_r2_mean"], color="#e0e7ff")
                     .highlight_min(subset=["cv_mae_mean"], color="#e0e7ff"),
                     hide_index=True, width="stretch")
        show_plot("models")
    if v1v2 is not None:
        st.markdown("#### v1 → v2, one change at a time (held-out test set)")
        st.dataframe(v1v2[["step", "data", "test_r2", "test_mae", "test_rmse"]].style.format(
            {"test_r2": "{:.3f}", "test_mae": "{:.2f}", "test_rmse": "{:.2f}"}), hide_index=True, width="stretch")
        st.caption("Step 1 also changes which rows are scored (impossible listings are removed), so part of its "
                   "gain is an easier test set. Steps 2–4 use exactly the same rows.")
    st.markdown("#### How good are the predictions?")
    show_plot("predictions")


# ═══════════════════════════════════════════════════════════════
# TAB 4: MARKET SEGMENTS
# ═══════════════════════════════════════════════════════════════
with tabs[3]:
    section("Market segments", "Clustering 214 locations by their price, size, volume and readiness")
    what_it_does("Summarises each location (mean price per sq ft, median size, bedrooms, listing count, share "
                 "ready to move) and groups similar locations with K-Means, DBSCAN and Agglomerative clustering. "
                 "The best grouping is named from its profile and shown in 2-D with PCA.")
    locs, profiles, ccomp = report("location_clusters.csv"), report("cluster_profiles.csv"), \
        report("clustering_comparison.csv")
    if require(locs, "reports/location_clusters.csv") and require(profiles, "reports/cluster_profiles.csv"):
        st.markdown("#### Find a location")
        pick = st.selectbox("Location", sorted(locs["location"]), key="seg_loc",
                            index=sorted(locs["location"]).index("Whitefield")
                            if "Whitefield" in set(locs["location"]) else 0)
        r = locs.set_index("location").loc[pick]
        seg = profiles.set_index("segment").loc[r["segment"]]
        st.markdown(f'<div class="clean-card" style="border-left:5px solid #6366f1;"><p style="margin:0;">'
                    f'<b>{pick}</b> is in the <b>{r["segment"]}</b> segment '
                    f'({int(seg["locations"])} locations, e.g. {seg["examples"]}).</p></div>',
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            pick: [r["mean_pps"], r["median_sqft"], r["mean_bhk"], r["listings"], r["pct_ready"] * 100],
            "segment average": [seg["mean_pps"], seg["median_sqft"], seg["mean_bhk"], seg["mean_listings"],
                                seg["pct_ready"] * 100],
        }, index=["Mean price per sq ft (₹)", "Median size (sq ft)", "Mean BHK", "Listings",
                  "% ready to move"]).round(1), width="stretch")
        st.markdown("#### Segment profiles")
        st.dataframe(profiles, hide_index=True, width="stretch",
                     column_config={"pct_ready": st.column_config.NumberColumn("pct_ready", format="%.2f")})
        show_plot("clusters")
        if ccomp is not None:
            st.markdown("#### Three clustering methods compared")
            st.dataframe(ccomp, hide_index=True, width="stretch")
            st.caption("Silhouette: higher is better. Davies-Bouldin: lower is better. DBSCAN finds one dense "
                       "core plus 'noise' locations, so it cannot provide segments.")
        c1, c2 = st.columns(2)
        with c1:
            show_plot("k_selection")
        with c2:
            show_plot("k_distance")
        show_plot("dendrogram")


# ═══════════════════════════════════════════════════════════════
# TAB 5: ASSOCIATION RULES
# ═══════════════════════════════════════════════════════════════
with tabs[4]:
    section("Association rules", "Which listing characteristics go together?")
    what_it_does("Turns each listing into a 'basket' of items (price band, size band, BHK, bathrooms, area type, "
                 "ready / under construction, market segment) and mines rules 'if A then C' with Apriori and "
                 "FP-Growth. Lift says how many times more likely C is when A holds.")
    rules = report("association_rules_all.csv")
    algos = report("association_algorithms.csv")
    if require(rules, "reports/association_rules_all.csv"):
        c1, c2, c3 = st.columns(3)
        min_sup = c1.slider("Minimum support (%)", float(rules["support"].min() * 100),
                            float(rules["support"].max() * 100), 2.0, 0.5)
        min_conf = c2.slider("Minimum confidence (%)", 50.0, 100.0, 60.0, 5.0)
        min_lift = c3.slider("Minimum lift", 1.2, float(rules["lift"].max()), 1.5, 0.1)
        term = st.text_input("Rules containing (e.g. `price=Premium`, `segment`, `bhk=3`)", "")
        f = rules[(rules["support"] * 100 >= min_sup) & (rules["confidence"] * 100 >= min_conf)
                  & (rules["lift"] >= min_lift)]
        if term:
            f = f[f["if"].str.contains(term, case=False, regex=False)
                  | f["then"].str.contains(term, case=False, regex=False)]
        st.markdown(f"**{len(f):,}** of {len(rules):,} rules match.")
        st.dataframe(f[["if", "then", "support", "confidence", "lift", "listings"]].sort_values("lift", ascending=False),
                     hide_index=True, width="stretch",
                     column_config={"support": st.column_config.NumberColumn(format="%.3f"),
                                    "confidence": st.column_config.NumberColumn(format="%.3f"),
                                    "lift": st.column_config.NumberColumn(format="%.2f")})
        rep = REPORTS_DIR / "association_report.md"
        if rep.exists():
            text = rep.read_text(encoding="utf-8")
            m = re.search(r"## 4\. The five best rules in plain English\n\n(.*?)\n\nCaveats", text, re.S)
            if m:
                st.markdown("#### The five best rules in plain English")
                st.markdown(m.group(1))
        show_plot("rules")
    if algos is not None:
        st.markdown("#### Apriori vs FP-Growth")
        st.dataframe(algos, hide_index=True, width="stretch")
        st.caption("Both are exact and returned identical itemsets. Here FP-Growth is slower: only 27 items and "
                   "dense baskets suit mlxtend's vectorised Apriori, while its FP-Growth is pure Python.")


# ═══════════════════════════════════════════════════════════════
# TAB 6: FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════
with tabs[5]:
    section("Feature importance", "What does the price model actually rely on?")
    what_it_does("Measures importance two ways: XGBoost's built-in gain (how much the trees used a feature while "
                 "training) and permutation importance (how much worse test predictions get when one input is "
                 "shuffled). Then tests with SelectKBest and RFE whether fewer columns keep the same accuracy.")
    imp, sel = report("feature_importance.csv"), report("feature_selection.csv")
    if require(imp, "reports/feature_importance.csv"):
        st.dataframe(imp, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f") for c in imp.columns
                                    if c != "feature"})
        top = imp.sort_values("perm_r2_drop", ascending=False).iloc[0]
        st.caption(f"Permutation importance on the test set says **{top['feature']}** matters most (R² falls by "
                   f"{top['perm_r2_drop']:.2f} when it is shuffled). Gain credits location most because its "
                   "150+ dummy columns offer many split points: gain measures use, not need.")
        show_plot("importance")
    if require(sel, "reports/feature_selection.csv"):
        st.markdown("#### Can fewer columns keep the accuracy?")
        st.dataframe(sel.drop(columns="fit_seconds", errors="ignore"), hide_index=True, width="stretch",
                     column_config={"cv_r2_mean": st.column_config.NumberColumn(format="%.4f"),
                                    "cv_mae_mean": st.column_config.NumberColumn(format="%.2f")})
        show_plot("selection")
