"""
Raw-data profiling
==================
Audits the raw CSV before any cleaning, so the wrangling decisions can be
traced back to evidence:
  - a ydata-profiling HTML report  -> reports/raw_profile.html
  - missingno matrix (where the gaps are, row by row)
  - missingno heatmap (whether gaps in one column predict gaps in another)

Usage:
    python -m src.profiling
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures without opening windows
import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd

from src import FIGURES_DIR, PLOTS, RAW_CSV, REPORTS_DIR

log = logging.getLogger(__name__)


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    """Read the raw listings exactly as stored (total_sqft stays text)."""
    df = pd.read_csv(path)
    log.info("Loaded %s: %d rows x %d columns", path.name, *df.shape)
    return df


def build_profile_report(df: pd.DataFrame, out_path: Path = REPORTS_DIR / "raw_profile.html") -> Path:
    """Write a ydata-profiling report: types, distributions, missing values, duplicates, correlations."""
    from ydata_profiling import ProfileReport  # slow import, so only when needed

    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = ProfileReport(
        df,
        title="Bengaluru House Data: raw profile (before wrangling)",
        explorative=True,
        progress_bar=False,
    )
    report.to_file(out_path)
    log.info("Saved profiling report: %s", out_path)
    return out_path


def plot_missingno(df: pd.DataFrame, fig_dir: Path = FIGURES_DIR) -> list[Path]:
    """Save the missingno matrix and nullity-correlation heatmap."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    ax = msno.matrix(df, figsize=(12, 6), fontsize=10, sparkline=False)
    ax.set_title("Missing values by row (white = missing)", fontsize=13)
    path = fig_dir / PLOTS["missingno_matrix"]
    ax.get_figure().savefig(path, bbox_inches="tight", dpi=130)
    plt.close("all")
    paths.append(path)

    # The heatmap only makes sense for columns that actually have gaps.
    gappy = df.loc[:, df.isna().any()]
    ax = msno.heatmap(gappy, figsize=(8, 6), fontsize=10)
    ax.set_title("Nullity correlation (1 = always missing together)", fontsize=13)
    path = fig_dir / PLOTS["missingno_heatmap"]
    ax.get_figure().savefig(path, bbox_inches="tight", dpi=130)
    plt.close("all")
    paths.append(path)

    for p in paths:
        log.info("Saved figure: %s", p)
    return paths


def run_profiling(df: pd.DataFrame | None = None) -> dict[str, Path | list[Path]]:
    df = load_raw() if df is None else df
    return {"profile": build_profile_report(df), "missingno": plot_missingno(df)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_profiling()


if __name__ == "__main__":
    main()
