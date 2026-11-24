"""v2 pipeline package: shared paths and settings used by every module."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "Bengaluru_House_Data.csv"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "plots_v2"
REPORTS_DIR = ROOT / "reports"

RANDOM_STATE = 42

# Every figure's file name, so the reports can link to plots that src/visualize.py draws.
PLOTS = {
    "missingno_matrix": "00_missingno_matrix.png",
    "missingno_heatmap": "00_missingno_heatmap.png",
    "distributions": "01_distributions_before_after.png",
    "waterfall": "02_wrangling_waterfall.png",
    "outliers": "03_outlier_methods.png",
    "models": "04_model_comparison.png",
    "predictions": "05_predicted_vs_actual.png",
    "k_selection": "06_kmeans_elbow_silhouette.png",
    "k_distance": "07_dbscan_k_distance.png",
    "dendrogram": "08_dendrogram.png",
    "clusters": "09_clusters_pca.png",
    "rules": "10_rules_support_confidence.png",
    "importance": "11_feature_importance.png",
    "selection": "12_feature_selection.png",
}
