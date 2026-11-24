"""
Run the whole v2 pipeline end to end, from the raw CSV
======================================================
  1. profiling     raw-data profile (ydata-profiling) + missingno plots
  2. wrangling     8 logged cleaning steps -> outputs/clean_data.csv
  3. modeling      7 models x 2 targets, CV, tuning, v1 comparison -> outputs/best_model.joblib
  4. clustering    location segments (K-Means / DBSCAN / Agglomerative)
  5. association   Apriori vs FP-Growth, association rules
  6. features      built-in + permutation importance, SelectKBest / RFE
  7. visualize     every figure -> outputs/plots_v2/

Every stage logs to the console and to reports/pipeline.log. Each stage reads
the previous stages' saved files, so a stage can also be run on its own with
`python -m src.<module>`.

Usage:
    python run_pipeline.py                 # everything (about 20-25 minutes; tuning is the slow part)
    python run_pipeline.py --fresh         # delete all generated outputs and reports first
    python run_pipeline.py --skip-profile  # skip the ydata-profiling HTML report
"""

import argparse
import logging
import shutil
import sys
import time

from src import OUTPUTS_DIR, REPORTS_DIR

log = logging.getLogger("pipeline")
KEEP = {".gitkeep", "v2_summary.md", "demo_script.md"}  # hand-written, not generated


def setup_logging() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(REPORTS_DIR / "pipeline.log", mode="w",
                                                                         encoding="utf-8")]
    for h in handlers:
        h.setFormatter(fmt)
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
    # Third-party loggers: matplotlib's style loader reports pyparsing deprecations as WARNINGs.
    for noisy in ("matplotlib", "PIL", "visions", "numba"):
        logging.getLogger(noisy).setLevel(logging.ERROR)


def clear_generated() -> None:
    """Delete everything the pipeline generates, so the run below starts from the raw CSV alone."""
    removed = 0
    for folder in (OUTPUTS_DIR, REPORTS_DIR):
        for path in folder.iterdir():
            if path.name in KEEP or path.name == "pipeline.log":
                continue
            shutil.rmtree(path) if path.is_dir() else path.unlink()
            removed += 1
    log.info("--fresh: removed %d generated files/folders from outputs/ and reports/", removed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fresh", action="store_true", help="delete generated outputs and reports first")
    parser.add_argument("--skip-profile", action="store_true", help="skip the ydata-profiling HTML report")
    args = parser.parse_args()

    setup_logging()
    if args.fresh:
        clear_generated()

    # Imported here so --help is instant and logging is configured before the modules log anything.
    from src.association import run_association
    from src.clustering import run_clustering
    from src.features import run_features
    from src.modeling import run_modeling
    from src.profiling import load_raw, plot_missingno, run_profiling
    from src.visualize import run_visualize
    from src.wrangling import run_wrangling

    timings = {}

    def stage(name, fn, *a, **kw):
        log.info("=" * 70)
        log.info("STAGE: %s", name)
        log.info("=" * 70)
        start = time.perf_counter()
        result = fn(*a, **kw)
        timings[name] = time.perf_counter() - start
        log.info("%s finished in %.1fs", name, timings[name])
        return result

    total_start = time.perf_counter()
    raw = load_raw()
    if args.skip_profile:
        stage("profiling (missingno only)", plot_missingno, raw)
    else:
        stage("profiling", run_profiling, raw)
    clean, wlog = stage("wrangling", run_wrangling, raw)
    model = stage("modeling", run_modeling, clean)
    clusters = stage("clustering", run_clustering, clean)
    rules = stage("association", run_association, clean)
    feats = stage("features", run_features, clean)
    figures = stage("visualize", run_visualize)

    # ── Final summary ────────────────────────────────────────────────────────
    v1v2 = model["v1_vs_v2"].set_index(model["v1_vs_v2"]["step"].str[0])
    final = model["final"]
    log.info("=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    log.info("Rows: %d raw -> %d clean (%d removed by %d logged steps)",
             len(raw), len(clean), len(raw) - len(clean), len(wlog.records))
    log.info("Best model: %s (%s target), test R² %.3f, MAE %.2f lakhs",
             final["model"], final["target"], final["scores"]["test_r2"], final["scores"]["test_mae"])
    log.info("v1 baseline: test R² %.3f, MAE %.2f lakhs -> error down %.0f%%",
             v1v2.at["0", "test_r2"], v1v2.at["0", "test_mae"],
             100 * (1 - final["scores"]["test_mae"] / v1v2.at["0", "test_mae"]))
    log.info("Market segments: %s", "; ".join(clusters["profiles"]["segment"]))
    log.info("Association rules kept: %d (top rule lift %.2f)", len(rules["rules"]), rules["rules"]["lift"].iloc[0])
    log.info("Most important feature (permutation): %s", feats["importance"].index[0])
    log.info("Figures: %d in outputs/plots_v2/", len(figures))
    log.info("Stage times: %s", ", ".join(f"{k} {v:.0f}s" for k, v in timings.items()))
    log.info("Total: %.1f minutes. Full log: reports/pipeline.log", (time.perf_counter() - total_start) / 60)


if __name__ == "__main__":
    main()
