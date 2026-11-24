"""
Data wrangling pipeline
=======================
Turns the raw listings into outputs/clean_data.csv through eight small steps.
Every step is a function decorated with @logged_step, which records rows,
columns and nulls before and after, plus the step's own notes, tables and
decision. The log is written to:
  - reports/wrangling_log.csv   (one row per step, machine-readable)
  - reports/wrangling_report.md (the same, with explanations and tables)

Each output row keeps `raw_row`, its line in the raw CSV (0-based, header
excluded), so any cleaned or removed listing can be traced back.

Usage:
    python -m src.wrangling
"""

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from src import OUTPUTS_DIR, PLOTS, RANDOM_STATE, RAW_CSV, REPORTS_DIR
from src.profiling import load_raw

log = logging.getLogger(__name__)

TEXT_COLUMNS = ["area_type", "availability", "location", "size", "society", "total_sqft"]


# ──────────────────────────────────────────────────────────────────────────────
# STEP LOGGING
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class StepInfo:
    """What a step reports about itself, on top of the automatic before/after counts."""
    notes: list[str] = field(default_factory=list)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    decision: str = ""


@dataclass
class StepRecord:
    number: int
    title: str
    purpose: str
    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int
    nulls_before: int
    nulls_after: int
    columns_added: list[str]
    columns_dropped: list[str]
    null_changes: dict[str, tuple[int, int]]
    seconds: float
    info: StepInfo


class WranglingLog:
    """Collects one StepRecord per step and writes the CSV and Markdown reports."""

    def __init__(self) -> None:
        self.records: list[StepRecord] = []

    def add(self, number, title, purpose, before: pd.DataFrame, after: pd.DataFrame, info: StepInfo, seconds):
        nb, na = before.isna().sum(), after.isna().sum()
        cols = list(before.columns) + [c for c in after.columns if c not in before.columns]
        null_changes = {
            c: (int(nb.get(c, 0)), int(na.get(c, 0)))
            for c in cols
            if int(nb.get(c, 0)) != int(na.get(c, 0))
        }
        record = StepRecord(
            number=number,
            title=title,
            purpose=purpose,
            rows_before=len(before),
            rows_after=len(after),
            cols_before=before.shape[1],
            cols_after=after.shape[1],
            nulls_before=int(nb.sum()),
            nulls_after=int(na.sum()),
            columns_added=[c for c in after.columns if c not in before.columns],
            columns_dropped=[c for c in before.columns if c not in after.columns],
            null_changes=null_changes,
            seconds=seconds,
            info=info,
        )
        self.records.append(record)
        log.info(
            "Step %d %s: rows %d -> %d, nulls %d -> %d (%.2fs)",
            number, title, record.rows_before, record.rows_after,
            record.nulls_before, record.nulls_after, seconds,
        )
        for note in info.notes:
            log.info("    %s", note)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "step": r.number,
                "title": r.title,
                "rows_before": r.rows_before,
                "rows_after": r.rows_after,
                "rows_removed": r.rows_before - r.rows_after,
                "cols_before": r.cols_before,
                "cols_after": r.cols_after,
                "nulls_before": r.nulls_before,
                "nulls_after": r.nulls_after,
                "columns_added": ";".join(r.columns_added),
                "columns_dropped": ";".join(r.columns_dropped),
                "null_changes": ";".join(f"{c}:{b}->{a}" for c, (b, a) in r.null_changes.items()),
                "decision": r.info.decision,
                "notes": " | ".join(r.info.notes),
                "seconds": round(r.seconds, 3),
            }
            for r in self.records
        ])

    def write_csv(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(path, index=False)
        return path

    def write_markdown(self, path: Path, raw: pd.DataFrame, clean: pd.DataFrame) -> Path:
        lines = [
            "# Wrangling report",
            "",
            f"Generated {datetime.now():%Y-%m-%d %H:%M} by `src/wrangling.py` from `{RAW_CSV.name}`.",
            f"Raw data: **{len(raw):,} rows × {raw.shape[1]} columns**. "
            f"Clean data: **{len(clean):,} rows × {clean.shape[1]} columns** "
            f"({len(raw) - len(clean):,} rows removed, {(len(raw) - len(clean)) / len(raw):.1%}).",
            "",
            "Raw profile: [`raw_profile.html`](raw_profile.html). "
            f"Missing-value plots: `outputs/plots_v2/{PLOTS['missingno_matrix']}`, "
            f"`outputs/plots_v2/{PLOTS['missingno_heatmap']}`. Before/after distributions and a row-count "
            f"waterfall: `outputs/plots_v2/{PLOTS['distributions']}`, `outputs/plots_v2/{PLOTS['waterfall']}`.",
            "",
            "## Summary",
            "",
            md_table(self.to_frame()[[
                "step", "title", "rows_before", "rows_after", "rows_removed", "nulls_before", "nulls_after",
            ]]),
            "",
        ]
        for r in self.records:
            lines += [f"## Step {r.number}: {r.title}", "", f"*{r.purpose}*", ""]
            lines.append(
                f"- Rows: {r.rows_before:,} → {r.rows_after:,} ({r.rows_before - r.rows_after:,} removed). "
                f"Columns: {r.cols_before} → {r.cols_after}. Nulls: {r.nulls_before:,} → {r.nulls_after:,}."
            )
            if r.columns_added:
                lines.append(f"- Columns added: {', '.join(f'`{c}`' for c in r.columns_added)}")
            if r.columns_dropped:
                lines.append(f"- Columns dropped: {', '.join(f'`{c}`' for c in r.columns_dropped)}")
            if r.null_changes:
                lines.append("- Null counts changed: " + ", ".join(
                    f"`{c}` {b:,} → {a:,}" for c, (b, a) in r.null_changes.items()))
            lines += [f"- {n}" for n in r.info.notes]
            lines.append("")
            if r.info.decision:
                lines += [f"**Decision:** {r.info.decision}", ""]
            for name, table in r.info.tables.items():
                lines += [f"**{name}**", "", md_table(table), ""]
        lines += [
            "## Output: `outputs/clean_data.csv`",
            "",
            md_table(pd.DataFrame({
                "column": clean.columns,
                "dtype": clean.dtypes.astype(str).to_numpy(),
                "nulls": clean.isna().sum().to_numpy(),
                "example": clean.iloc[0].astype(str).to_numpy(),
            })),
            "",
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
        return path


def md_table(df: pd.DataFrame, index: bool = False) -> str:
    """Render a DataFrame as a GitHub Markdown table (no tabulate dependency)."""
    if index:
        df = df.reset_index()

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else (f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.3g}" if abs(v) < 1 else f"{v:,.2f}")
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return str(v).replace("|", "\\|")

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def logged_step(number: int, title: str):
    """Wrap a step `fn(df) -> (df, StepInfo)` so it is timed and recorded in the WranglingLog."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(df: pd.DataFrame, wlog: WranglingLog, **kwargs) -> pd.DataFrame:
            start = time.perf_counter()
            out, info = fn(df.copy(), **kwargs)
            purpose = re.sub(r"\s+", " ", (fn.__doc__ or "").strip().split("\n\n")[0])
            wlog.add(number, title, purpose, df, out, info, time.perf_counter() - start)
            return out
        return wrapper
    return decorator


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: TEXT NORMALISATION AND DUPLICATES
# ──────────────────────────────────────────────────────────────────────────────
@logged_step(1, "Normalise text and drop exact duplicates")
def clean_text_and_drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, StepInfo]:
    """Strip and collapse whitespace in every text column, unify location spellings
    that differ only by letter case, then drop rows that are exact duplicates."""
    info = StepInfo()
    raw_dups = int(df.drop(columns="raw_row").duplicated().sum())

    changed = {}
    for col in TEXT_COLUMNS:
        before = df[col]
        after = before.str.strip().str.replace(r"\s+", " ", regex=True)
        if col == "location":
            after = after.str.strip(" ,")  # e.g. "Electronic city Phase 1,"
        after = after.replace("", np.nan)
        changed[col] = int(((before != after) & before.notna()).sum())
        df[col] = after

    # "Anand Nagar" vs "Anand nagar": keep the most common spelling of each lower-cased name.
    key = df["location"].str.lower()
    canonical = df.groupby(key)["location"].agg(lambda s: s.value_counts().index[0])
    unified = key.map(canonical)
    case_fixed = int(((unified != df["location"]) & df["location"].notna()).sum())
    loc_unique_before = df["location"].nunique()
    df["location"] = unified

    dup_mask = df.drop(columns="raw_row").duplicated()
    n_dups = int(dup_mask.sum())
    df = df[~dup_mask]

    info.tables["Values changed by whitespace normalisation"] = pd.DataFrame(
        {"column": list(changed), "values_changed": list(changed.values())})
    info.notes += [
        f"Every `area_type` value had a double space (\"Super built-up  Area\"); {changed['area_type']:,} values fixed.",
        f"Location case variants unified: {case_fixed} values; unique locations {loc_unique_before:,} → {df['location'].nunique():,}.",
        f"Exact duplicate rows in the raw file: {raw_dups}. After normalisation: {n_dups} (removed, first copy kept).",
    ]
    info.decision = (
        "Normalise before de-duplicating, so rows that differ only by stray spaces count as duplicates. "
        "Duplicates are removed because they over-weight some listings in every statistic and can leak "
        "between train and test sets."
    )
    return df, info


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: SIZE -> BHK
# ──────────────────────────────────────────────────────────────────────────────
SIZE_RE = re.compile(r"^(\d+)\s*(BHK|Bedroom|RK)$", re.IGNORECASE)
SIZE_UNITS = {"bhk": "BHK", "bedroom": "Bedroom", "rk": "RK"}


@logged_step(2, "Parse size into bhk")
def parse_size(df: pd.DataFrame) -> tuple[pd.DataFrame, StepInfo]:
    """Regex-parse `size` ("2 BHK", "4 Bedroom", "1 RK") into an integer `bhk`
    and keep the suffix as `size_unit`, because it encodes the property type."""
    info = StepInfo()
    parts = df["size"].str.extract(SIZE_RE)
    unmatched = df["size"].notna() & parts[0].isna()
    if unmatched.any():
        info.notes.append(f"Unparseable size values (set to missing): {df.loc[unmatched, 'size'].unique().tolist()}")

    missing = parts[0].isna()
    gone = df[missing]
    info.notes.append(
        f"{len(gone)} rows have no size. All are {', '.join(gone['area_type'].unique())} listings "
        f"(availability: {', '.join(gone['availability'].unique())}) with bath missing in "
        f"{gone['bath'].isna().sum()} and balcony missing in {gone['balcony'].isna().sum()}: "
        "they are bare land plots, not houses."
    )
    df = df[~missing].copy()
    df["bhk"] = parts.loc[~missing, 0].astype(int)
    df["size_unit"] = parts.loc[~missing, 1].str.lower().map(SIZE_UNITS)
    df = df.drop(columns="size")

    info.tables["size_unit by area_type"] = pd.crosstab(df["area_type"], df["size_unit"]).reset_index()
    info.tables["bhk distribution"] = df["bhk"].value_counts().sort_index().rename_axis("bhk").reset_index()
    info.notes.append(f"{(df['bhk'] > 10).sum()} listings have more than 10 bedrooms (max {df['bhk'].max()}); handled in step 7.")
    info.decision = (
        "Drop the rows without a size: a house-price model cannot use a listing with no bedrooms, "
        "bathrooms or balconies, and guessing all three for empty land would invent data."
    )
    return df, info


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: TOTAL_SQFT -> NUMBER
# ──────────────────────────────────────────────────────────────────────────────
SQFT_PER_UNIT = {
    "Sq. Meter": 10.7639,
    "Sq. Yards": 9.0,
    "Acres": 43560.0,
    "Perch": 272.25,
    "Cents": 435.6,
    "Guntha": 1089.0,
    "Grounds": 2400.0,
}
RANGE_RE = re.compile(r"^([\d.]+)\s*-\s*([\d.]+)$")
UNIT_RE = re.compile(r"^([\d.]+)\s*(" + "|".join(map(re.escape, SQFT_PER_UNIT)) + r")$")


def parse_sqft(value) -> tuple[float, str]:
    """Return (square feet, how it was parsed) for one raw total_sqft value."""
    if pd.isna(value):
        return np.nan, "missing"
    text = str(value).strip()
    try:
        return float(text), "number"
    except ValueError:
        pass
    if m := RANGE_RE.match(text):
        return (float(m[1]) + float(m[2])) / 2, "range (midpoint)"
    if m := UNIT_RE.match(text):
        return float(m[1]) * SQFT_PER_UNIT[m[2]], f"unit: {m[2]}"
    return np.nan, "unparseable"


@logged_step(3, "Convert total_sqft to square feet")
def convert_total_sqft(df: pd.DataFrame) -> tuple[pd.DataFrame, StepInfo]:
    """Turn the text column `total_sqft` into a number: plain numbers as they are,
    ranges ("2100 - 2850") to their midpoint, other units to square feet."""
    info = StepInfo()
    parsed = df["total_sqft"].apply(parse_sqft)
    raw = df["total_sqft"]
    df["total_sqft"] = parsed.str[0]
    how = parsed.str[1]

    summary = (
        pd.DataFrame({"how": how, "raw": raw, "sqft": df["total_sqft"]})
        .groupby("how")
        .agg(rows=("raw", "size"), example_raw=("raw", "first"), example_sqft=("sqft", "first"))
        .sort_values("rows", ascending=False)
        .reset_index()
    )
    info.tables["How each value was parsed"] = summary
    info.tables["Conversion factors used (sq ft per unit)"] = pd.DataFrame(
        {"unit": list(SQFT_PER_UNIT), "sqft_per_unit": list(SQFT_PER_UNIT.values())})

    bad = how == "unparseable"
    if bad.any():
        info.notes.append(f"Unparseable values (rows removed): {raw[bad].tolist()}")
        log.warning("Unparseable total_sqft values: %s", raw[bad].tolist())
    else:
        info.notes.append("No unparseable values: every total_sqft was converted.")
    df = df[~bad]
    n_converted = int((~how.isin(["number"])).sum() - bad.sum())
    info.notes.append(f"{n_converted} values needed conversion; v1 kept the ranges but turned all "
                      f"{int(how.str.startswith('unit').sum())} unit values into NaN and dropped them.")
    info.decision = (
        "Ranges become their midpoint (the best single estimate of a range). Units use standard "
        "conversion factors; Guntha, Cents and Grounds are Indian land units."
    )
    return df, info


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: AVAILABILITY
# ──────────────────────────────────────────────────────────────────────────────
READY_VALUES = {"Ready To Move", "Immediate Possession"}
POSSESSION_RE = re.compile(r"^(\d{2})-([A-Za-z]{3})$")
MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}


@logged_step(4, "Parse availability")
def parse_availability(df: pd.DataFrame) -> tuple[pd.DataFrame, StepInfo]:
    """Split `availability` into `is_ready_to_move` (1/0) and, for properties not yet
    ready, `possession_year` and `possession_month` from values like "18-Dec"."""
    info = StepInfo()
    avail = df["availability"]
    ready = avail.isin(READY_VALUES)
    parts = avail.str.extract(POSSESSION_RE)
    dated = parts[0].notna()

    other = ~ready & ~dated
    if other.any():
        info.notes.append(f"Unrecognised availability values (treated as not ready, date unknown): "
                          f"{avail[other].value_counts().to_dict()}")

    years = parts.loc[dated, 0].astype(int)
    info.notes.append(
        f"Date-like values: {dated.sum():,}. The 2-digit prefix only takes values {years.min()}–{years.max()} "
        f"while all {parts.loc[dated, 1].nunique()} months occur, so it is a year (\"18-Dec\" = Dec 2018), "
        "not a day of the month."
    )
    df["is_ready_to_move"] = ready.astype(int)
    df["possession_year"] = (2000 + years).astype("Int64").reindex(df.index)
    df["possession_month"] = parts.loc[dated, 1].str.title().map(MONTHS).astype("Int64").reindex(df.index)
    df = df.drop(columns="availability")

    info.tables["Availability categories"] = pd.DataFrame({
        "category": ["Ready To Move", "Immediate Possession", "future possession date", "unrecognised"],
        "rows": [int((avail == "Ready To Move").sum()), int((avail == "Immediate Possession").sum()),
                 int(dated.sum()), int(other.sum())],
    })
    info.notes.append(
        "`possession_year`/`possession_month` are empty for ready properties by design "
        "(there is no future date), which is why the null count rises in this step."
    )
    info.decision = "\"Immediate Possession\" is treated as ready to move."
    if not (avail == "Immediate Possession").any():
        info.decision += " (All 16 such rows in the raw file were the land plots removed in step 2.)"
    return df, info


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5: MISSING VALUES
# ──────────────────────────────────────────────────────────────────────────────
IMPUTE_COLUMNS = ["bath", "balcony"]


def impute_median_by_bhk(df: pd.DataFrame, cols=IMPUTE_COLUMNS) -> pd.DataFrame:
    """Fill each column with the median of listings that have the same bhk (global median as fallback)."""
    out = df.copy()
    for c in cols:
        group_median = out.groupby("bhk")[c].transform("median")
        out[c] = out[c].fillna(group_median).fillna(out[c].median()).round()
    return out


def impute_knn(df: pd.DataFrame, cols=IMPUTE_COLUMNS, k: int = 5) -> pd.DataFrame:
    """Fill each column from the k most similar listings (bhk, log area, readiness and the other column).
    Price is deliberately not used, so the target does not leak into the features."""
    out = df.copy()
    X = pd.DataFrame({
        "bhk": out["bhk"],
        "log_sqft": np.log1p(out["total_sqft"]),
        "is_ready_to_move": out["is_ready_to_move"],
        **{c: out[c] for c in cols},
    })
    scaler = StandardScaler()  # KNN uses distances, so features must share a scale
    filled = scaler.inverse_transform(KNNImputer(n_neighbors=k).fit_transform(scaler.fit_transform(X)))
    for i, c in enumerate(X.columns):
        if c in cols:
            out[c] = np.clip(np.round(filled[:, i]), 0, None)
    return out


def compare_imputers(df: pd.DataFrame, n_repeats: int = 5, mask_frac: float = 0.10) -> pd.DataFrame:
    """Hide 10% of the known bath/balcony values, impute them with both methods and
    measure how close each gets to the truth. Repeated with different random masks."""
    complete = df.dropna(subset=IMPUTE_COLUMNS)
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for repeat in range(n_repeats):
        masked = complete.copy()
        masks = {}
        for c in IMPUTE_COLUMNS:
            masks[c] = rng.random(len(masked)) < mask_frac
            masked.loc[masks[c], c] = np.nan
        for method, fn in [("median_by_bhk", impute_median_by_bhk), ("knn_k5", impute_knn)]:
            filled = fn(masked)
            for c in IMPUTE_COLUMNS:
                truth = complete[c].to_numpy()[masks[c]]
                pred = filled[c].to_numpy()[masks[c]]
                rows.append({"repeat": repeat, "method": method, "column": c,
                             "mae": np.abs(pred - truth).mean(), "exact_match": (pred == truth).mean()})
    return (pd.DataFrame(rows).groupby(["column", "method"])[["mae", "exact_match"]].mean().reset_index())


@logged_step(5, "Handle missing values")
def handle_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, StepInfo]:
    """Drop `society`, fill the missing `location` with the mode, and impute `bath`
    and `balcony` with the better of per-bhk median and KNN imputation."""
    info = StepInfo()
    info.tables["Missing values at the start of this step"] = (
        df.isna().sum().rename("missing").to_frame()
        .assign(pct=lambda t: (t["missing"] / len(df) * 100).round(1))
        .query("missing > 0").rename_axis("column").reset_index())

    # society: too sparse and too high-cardinality to be a feature
    soc = df["society"]
    pct_missing = soc.isna().mean()
    per_society = soc.value_counts()
    soc_facts = (
        f"{pct_missing:.1%} missing and {soc.nunique():,} distinct values among "
        f"{soc.notna().sum():,} known rows (median {per_society.median():.0f} listing per society; "
        f"{(per_society == 1).sum():,} societies appear once)"
    )

    # location: mode of the same society when known, else the overall mode
    loc_missing = df["location"].isna()
    for idx in df.index[loc_missing]:
        same_soc = df.loc[(df["society"] == df.at[idx, "society"]) & df["location"].notna(), "location"]
        fill = same_soc.mode().iloc[0] if len(same_soc) else df["location"].mode().iloc[0]
        source = f"mode of society '{df.at[idx, 'society']}'" if len(same_soc) else "overall mode"
        info.notes.append(f"raw_row {df.at[idx, 'raw_row']}: location filled with '{fill}' ({source}).")
        df.at[idx, "location"] = fill
    df = df.drop(columns="society")
    info.notes.insert(0, f"`society`: {soc_facts}.")

    # bath / balcony: evaluate both imputers before choosing
    comparison = compare_imputers(df)
    info.tables["Imputer comparison (10% of known values hidden, 5 repeats)"] = comparison
    mae = comparison.groupby("method")["mae"].mean()
    median_df, knn_df = impute_median_by_bhk(df), impute_knn(df)
    for c in IMPUTE_COLUMNS:
        miss = df[c].isna()
        differ = int((median_df.loc[miss, c] != knn_df.loc[miss, c]).sum())
        info.notes.append(f"`{c}`: {miss.sum()} missing; the two methods give different values for {differ} of them.")
    imputed_counts = pd.concat({
        f"{c} ({m})": frame.loc[df[c].isna(), c].astype(int).value_counts()
        for c in IMPUTE_COLUMNS for m, frame in [("median", median_df), ("knn", knn_df)]
    }, axis=1).fillna(0).astype(int).sort_index()
    info.tables["Imputed values on the actually-missing rows"] = imputed_counts.rename_axis("value").reset_index()

    # Prefer the simpler method unless KNN is clearly (>5%) more accurate.
    knn_better = mae["knn_k5"] < 0.95 * mae["median_by_bhk"]
    chosen = "knn_k5" if knn_better else "median_by_bhk"
    df = knn_df if knn_better else median_df
    df["bath"], df["balcony"] = df["bath"].astype(int), df["balcony"].astype(int)
    gain = 1 - mae["knn_k5"] / mae["median_by_bhk"]
    info.decision = (
        f"Drop `society` ({soc_facts}): too much is missing to impute honestly, and one-hot or target "
        "encoding thousands of tiny groups would overfit. "
        f"For bath/balcony, mean MAE is {mae['median_by_bhk']:.3f} (median by bhk) vs {mae['knn_k5']:.3f} (KNN), "
        f"so KNN is {abs(gain):.1%} {'more' if gain >= 0 else 'less'} accurate. Rule: use KNN only if it is more "
        f"than 5% better, otherwise prefer the simpler, deterministic median. Chosen: **{chosen}**."
    )
    return df, info


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6: PRICE FEATURES
# ──────────────────────────────────────────────────────────────────────────────
@logged_step(6, "Engineer price_per_sqft and log_price")
def add_price_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StepInfo]:
    """Add `price_per_sqft` in rupees (price is in lakhs, 1 lakh = ₹100,000) and
    `log_price`, the natural log of price, to tame its right skew."""
    info = StepInfo()
    df["price_per_sqft"] = df["price"] * 100_000 / df["total_sqft"]
    df["log_price"] = np.log(df["price"])
    info.tables["Distribution"] = (
        df[["price", "log_price", "price_per_sqft"]].describe().T
        .assign(skew=df[["price", "log_price", "price_per_sqft"]].skew())
        .rename_axis("feature").reset_index())
    info.notes.append(f"Skewness of price {df['price'].skew():.2f} → log_price {df['log_price'].skew():.2f}.")
    info.decision = (
        "`price_per_sqft` is used for outlier detection, clustering and analysis only. It is computed "
        "from the price, so it must never be a model input (target leakage)."
    )
    return df, info


# ──────────────────────────────────────────────────────────────────────────────
# STEP 7: OUTLIERS
# ──────────────────────────────────────────────────────────────────────────────
OUTLIER_METHODS = {
    # name: (family, what it checks)
    "iqr_pps": ("statistical", "price_per_sqft outside Q1 − 1.5·IQR … Q3 + 1.5·IQR (whole city)"),
    "zscore_pps": ("statistical", "|z-score of price_per_sqft| > 3 (whole city)"),
    "iqr3_log_pps": ("statistical", "log price_per_sqft outside Q1 − 3·IQR … Q3 + 3·IQR (Tukey far-out fence)"),
    "isolation_forest": ("ML, multivariate", "IsolationForest on log sqft, bhk, bath, log price_per_sqft (5% contamination)"),
    "lof": ("ML, multivariate", "LocalOutlierFactor, 20 neighbours, same features (5% contamination)"),
    "rule_sqft_per_bhk": ("domain rule", "total_sqft / bhk < 300"),
    "rule_bath": ("domain rule", "bath > bhk + 2"),
    "rule_location_pps": ("domain rule", "price_per_sqft outside its location's mean ± 1 std"),
    "rule_bhk_pps": ("domain rule", "n-BHK cheaper per sqft than the mean (n−1)-BHK in the same location (>5 listings)"),
}
FINAL_OUTLIER_RULES = ["rule_sqft_per_bhk", "rule_bath", "iqr3_log_pps"]


def log_pps_fences(pps: pd.Series, k: float = 3.0) -> tuple[float, float]:
    """Tukey fences computed on log(price_per_sqft), returned in rupees per sq ft."""
    q1, q3 = np.log(pps).quantile([0.25, 0.75])
    return float(np.exp(q1 - k * (q3 - q1))), float(np.exp(q3 + k * (q3 - q1)))


def flag_bhk_pps_inversion(df: pd.DataFrame, min_count: int = 5) -> pd.Series:
    """Flag n-BHK listings priced per sqft below the mean of (n−1)-BHK listings in the same location."""
    stats = (df.groupby(["location", "bhk"])["price_per_sqft"].agg(prev_mean="mean", prev_count="count")
             .reset_index())
    stats["bhk"] += 1  # the (n−1)-BHK statistics, keyed by n
    merged = df[["location", "bhk", "price_per_sqft"]].merge(stats, on=["location", "bhk"], how="left")
    flag = (merged["prev_count"] > min_count) & (merged["price_per_sqft"] < merged["prev_mean"])
    return flag.set_axis(df.index)


def outlier_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Run every outlier method and return one boolean column per method."""
    pps = df["price_per_sqft"]
    q1, q3 = pps.quantile([0.25, 0.75])
    iqr = q3 - q1

    X = StandardScaler().fit_transform(np.column_stack([
        np.log1p(df["total_sqft"]), df["bhk"], df["bath"], np.log(pps)]))
    iso = IsolationForest(contamination=0.05, random_state=RANDOM_STATE).fit_predict(X)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05).fit_predict(X)

    loc = df.groupby("location")["price_per_sqft"]
    loc_mean, loc_std = loc.transform("mean"), loc.transform("std")
    low, high = log_pps_fences(pps)

    return pd.DataFrame({
        "iqr_pps": (pps < q1 - 1.5 * iqr) | (pps > q3 + 1.5 * iqr),
        "zscore_pps": ((pps - pps.mean()) / pps.std()).abs() > 3,
        "iqr3_log_pps": (pps < low) | (pps > high),
        "isolation_forest": iso == -1,
        "lof": lof == -1,
        "rule_sqft_per_bhk": df["total_sqft"] / df["bhk"] < 300,
        "rule_bath": df["bath"] > df["bhk"] + 2,
        "rule_location_pps": loc_std.notna() & ((pps < loc_mean - loc_std) | (pps > loc_mean + loc_std)),
        "rule_bhk_pps": flag_bhk_pps_inversion(df),
    }, index=df.index)


@logged_step(7, "Detect and remove outliers")
def remove_outliers(df: pd.DataFrame, flags_path: Path | None = None) -> tuple[pd.DataFrame, StepInfo]:
    """Flag outliers with nine methods side by side, compare what each one catches,
    then remove only the rows flagged by the chosen strategy."""
    info = StepInfo()
    flags = outlier_flags(df)
    if flags_path is not None:
        flags_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([df[["raw_row"]], flags], axis=1).to_csv(flags_path, index=False)
        info.notes.append(f"Per-row flags saved to `{flags_path.relative_to(flags_path.parents[1]).as_posix()}`.")

    final = flags[FINAL_OUTLIER_RULES].any(axis=1)
    info.tables["Rows flagged by each method"] = pd.DataFrame({
        "method": list(OUTLIER_METHODS),
        "family": [v[0] for v in OUTLIER_METHODS.values()],
        "rule": [v[1] for v in OUTLIER_METHODS.values()],
        "flagged": [int(flags[m].sum()) for m in OUTLIER_METHODS],
        "pct": [round(flags[m].mean() * 100, 1) for m in OUTLIER_METHODS],
        "also_in_final": [int((flags[m] & final).sum()) for m in OUTLIER_METHODS],
    })
    overlap = pd.DataFrame(
        [[int((flags[a] & flags[b]).sum()) for b in OUTLIER_METHODS] for a in OUTLIER_METHODS],
        index=list(OUTLIER_METHODS), columns=list(OUTLIER_METHODS))
    info.tables["Overlap: rows flagged by both methods"] = overlap.rename_axis("method").reset_index()
    info.tables["How many methods flag each row"] = (
        flags.sum(axis=1).value_counts().sort_index().rename_axis("methods_flagging").rename("rows").reset_index())

    # What the statistical methods would cut, by location price level
    removed_stats = {m: df.loc[flags[m], "price_per_sqft"].median() for m in ["iqr_pps", "rule_location_pps"]}
    info.notes += [
        f"Median price_per_sqft of all rows: ₹{df['price_per_sqft'].median():,.0f}; of rows flagged by "
        f"iqr_pps: ₹{removed_stats['iqr_pps']:,.0f}; by rule_location_pps: ₹{removed_stats['rule_location_pps']:,.0f}.",
        f"z-score uses the mean and std, which the extreme rows themselves inflate (max price_per_sqft "
        f"₹{df['price_per_sqft'].max():,.0f}), so it flags only {int(flags['zscore_pps'].sum())} rows.",
    ]
    examples = df.loc[flags.sum(axis=1).nlargest(5).index,
                      ["raw_row", "location", "bhk", "total_sqft", "bath", "price", "price_per_sqft"]]
    info.tables["Rows flagged by the most methods"] = examples.assign(
        methods=flags.loc[examples.index].sum(axis=1))

    low, high = log_pps_fences(df["price_per_sqft"])
    domain = flags[["rule_sqft_per_bhk", "rule_bath"]].any(axis=1)
    info.decision = OUTLIER_DECISION.format(
        n_final=int(final.sum()), pct_final=final.mean() * 100,
        n_domain=int(domain.sum()), n_far=int(flags["iqr3_log_pps"].sum()),
        n_far_extra=int((flags["iqr3_log_pps"] & ~domain).sum()), low=low, high=high,
        n_loc=int(flags["rule_location_pps"].sum()), pct_loc=flags["rule_location_pps"].mean() * 100,
        n_iqr=int(flags["iqr_pps"].sum()), n_bhk=int(flags["rule_bhk_pps"].sum()),
    )
    return df[~final], info


OUTLIER_DECISION = (
    "Remove {n_final:,} rows ({pct_final:.1f}%) flagged by either of two kinds of evidence. "
    "(1) The two physical domain rules ({n_domain:,} rows): less than 300 sq ft per bedroom, or more than "
    "bhk + 2 bathrooms. These are almost certainly data errors whatever the location. "
    "(2) Tukey's far-out fence on log price_per_sqft ({n_far:,} rows, {n_far_extra:,} not already caught by "
    "the domain rules): outside ₹{low:,.0f} … ₹{high:,.0f} per sq ft. The domain rules only catch areas that "
    "are too small; this catches the opposite errors too, such as land parcels converted from Acres/Perch "
    "(a 2-BHK on 1.3 million sq ft at ₹2/sq ft). It uses quartiles, so the extreme rows cannot inflate it the "
    "way they inflate the z-score, and the log scale makes it symmetric for a right-skewed price. "
    "The other methods are kept as diagnostic flags, not used for removal: "
    "per-location mean ± 1 std would remove {n_loc:,} rows ({pct_loc:.1f}%), because about a third of any "
    "distribution lies outside ±1 std even when nothing is wrong; the city-wide IQR ({n_iqr:,} rows) mostly "
    "removes genuinely expensive localities; the bhk-inversion rule ({n_bhk:,} rows) assumes bigger flats "
    "cannot be cheaper per sq ft, which is not true in general; IsolationForest and LOF need a contamination "
    "rate that we would have to guess (5% here), so the number they flag is our choice, not a finding."
)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 8: RARE LOCATIONS AND PRICE BANDS
# ──────────────────────────────────────────────────────────────────────────────
RARE_LOCATION_MAX = 10
PRICE_BANDS = ["Low", "Mid", "High", "Premium"]


@logged_step(8, "Group rare locations and bin price")
def group_locations_and_bin_price(df: pd.DataFrame) -> tuple[pd.DataFrame, StepInfo]:
    """Replace locations with 10 or fewer listings by "other", and bin price into
    four quartile bands: Low, Mid, High, Premium."""
    info = StepInfo()
    counts = df["location"].value_counts()
    rare = counts[counts <= RARE_LOCATION_MAX].index
    is_rare = df["location"].isin(rare)
    df["location"] = df["location"].where(~is_rare, "other")
    info.notes.append(
        f"{len(rare):,} of {len(counts):,} locations have ≤ {RARE_LOCATION_MAX} listings; "
        f"their {is_rare.sum():,} rows ({is_rare.mean():.1%}) are now 'other'. "
        f"Locations left: {df['location'].nunique():,} (including 'other').")

    df["price_band"], edges = pd.qcut(df["price"], q=4, labels=PRICE_BANDS, retbins=True)
    info.tables["Price bands (lakhs)"] = pd.DataFrame({
        "band": PRICE_BANDS,
        "from": edges[:-1],
        "to": edges[1:],
        "rows": df["price_band"].value_counts().reindex(PRICE_BANDS).to_numpy(),
    })
    info.decision = (
        "Locations with very few listings give unreliable statistics and add hundreds of near-empty "
        "one-hot columns. Quartile bands give four equal-sized groups for association rules and the dashboard."
    )
    return df, info


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def run_wrangling(raw: pd.DataFrame | None = None, save: bool = True) -> tuple[pd.DataFrame, WranglingLog]:
    raw = load_raw() if raw is None else raw
    wlog = WranglingLog()
    df = raw.reset_index(drop=True).rename_axis(None)
    df.insert(0, "raw_row", df.index)

    df = clean_text_and_drop_duplicates(df, wlog)
    df = parse_size(df, wlog)
    df = convert_total_sqft(df, wlog)
    df = parse_availability(df, wlog)
    df = handle_missing_values(df, wlog)
    df = add_price_features(df, wlog)
    df = remove_outliers(df, wlog, flags_path=OUTPUTS_DIR / "outlier_flags.csv" if save else None)
    df = group_locations_and_bin_price(df, wlog)
    df = df.reset_index(drop=True)

    if save:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUTS_DIR / "clean_data.csv", index=False)
        wlog.write_csv(REPORTS_DIR / "wrangling_log.csv")
        wlog.write_markdown(REPORTS_DIR / "wrangling_report.md", raw, df)
        log.info("Saved outputs/clean_data.csv (%d rows), reports/wrangling_log.csv, reports/wrangling_report.md", len(df))
    return df, wlog


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_wrangling()


if __name__ == "__main__":
    main()
