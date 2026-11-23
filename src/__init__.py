"""v2 pipeline package: shared paths and settings used by every module."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "Bengaluru_House_Data.csv"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = ROOT / "reports"

RANDOM_STATE = 42
