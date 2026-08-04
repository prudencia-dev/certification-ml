"""Configuration centrale du projet."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEFAULT_PROCESSED_DATASET = PROCESSED_DATA_DIR / "dataset_clean.csv"
DEFAULT_MODEL_PATH = MODELS_DIR / "random_forest_pipeline.joblib"
TARGET_COLUMN = "risk_level"


@dataclass(frozen=True)
class TrainingConfig:
    """Paramètres reproductibles de séparation et d'entraînement."""

    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 200
    min_samples_leaf: int = 2


TRAINING_CONFIG = TrainingConfig()
