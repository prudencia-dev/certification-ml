
# Configuration du projet Machine Learning niveau 2.
from dataclasses import dataclass # permet de créer simplement une classe destinée à contenir des données.
from pathlib import Path # permet de représenter et de construire des chemins de fichiers.

# On définit les chemins de fichiers et dossiers utilisés dans le projet, ainsi que les paramètres d'entraînement du modèle.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# On définit les chemins par défaut pour le dataset brut, le dataset prétraité et le modèle entraîné, ainsi que la colonne cible pour la classification.
DEFAULT_RAW_DATASET = RAW_DATA_DIR / "ai_incidents_fr.csv"
DEFAULT_PROCESSED_DATASET = PROCESSED_DATA_DIR / "processed_dataset.csv"
DEFAULT_MODEL_PATH = MODELS_DIR / "random_forest_pipeline.joblib"

TARGET_COLUMN = "gravite"


# Paramètres reproductibles de séparation et d'entraînement du modèle
@dataclass(frozen=True)
class TrainingConfig:

    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 200
    min_samples_leaf: int = 2

# On crée une instance de la configuration d'entraînement avec les paramètres par défaut.
TRAINING_CONFIG = TrainingConfig()
