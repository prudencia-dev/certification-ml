
# Orchestrateur : point d'entrée principal du projet Machine Learning niveau 2.
import logging

# On importe les fonctions et constantes nécessaires depuis les modules du projet.
from src.config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PROCESSED_DATASET,
    DEFAULT_RAW_DATASET,
    REPORTS_DIR,
)
from src.evaluate import export_evaluation
from src.preprocessing import preprocess_file
from src.train import train_model
from src.utils import configure_logging

# On configure le logger pour capturer les messages d'information et d'erreur.
LOGGER = logging.getLogger(__name__)

# Execution du Pipeline de Machine Learning
def main() -> None:

    # Étape 1 : transformer le CSV français brut en variables exploitables.
    LOGGER.info("Étape 1/3 — Préparation du dataset français")
    preprocess_file(
        input_path=DEFAULT_RAW_DATASET,
        output_path=DEFAULT_PROCESSED_DATASET,
    )

    # Étape 2 : séparer les données, construire le pipeline et entraîner le modèle.
    LOGGER.info("Étape 2/3 — Entraînement de la forêt aléatoire")
    train_model(
        dataset_path=DEFAULT_PROCESSED_DATASET,
        model_path=DEFAULT_MODEL_PATH,
    )

    # Étape 3 : prédire l'ensemble réservé et produire les rapports français.
    LOGGER.info("Étape 3/3 — Évaluation du modèle")
    metriques = export_evaluation(
        model_path=DEFAULT_MODEL_PATH,
        reports_dir=REPORTS_DIR,
    )

    LOGGER.info("Pipeline terminé — métriques : %s", metriques)

# Point d'entrée du script lorsqu'il est exécuté directement.
if __name__ == "__main__":
    configure_logging()
    main()
