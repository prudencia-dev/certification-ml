
# Évaluation et export des performances du modèle.
import json
import logging
from pathlib import Path
from typing import Any

# Le module joblib est utilisé pour la sérialisation et la désérialisation d'objets Python, notamment pour sauvegarder et charger des modèles de machine learning.
import joblib
import matplotlib

# Le backend non interactif permet de produire le graphique sur un serveur ou
# dans une CI, sans écran ni fenêtre graphique.
matplotlib.use("Agg")

# Outils de visualisation et métriques de classification.
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.config import DEFAULT_MODEL_PATH, REPORTS_DIR
from src.utils import require_file

LOGGER = logging.getLogger(__name__)


# Évalue le pipeline sauvegardé sur le jeu de test réservé.
def evaluate_bundle(
    bundle: dict[str, Any],
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:

    pipeline = bundle["pipeline"]
    y_true = bundle["y_test"]
    predictions = pipeline.predict(bundle["x_test"])
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )
    metrics = {
        "exactitude": float(accuracy_score(y_true, predictions)),
        "precision_ponderee": float(precision),
        "rappel_pondere": float(recall),
        "f1_pondere": float(f1_score),
        "nombre_exemples_evaluation": int(len(y_true)),
    }
    report = pd.DataFrame(
        classification_report(y_true, predictions, output_dict=True, zero_division=0)
    ).transpose()
    report = report.rename(
        index={
            "accuracy": "exactitude",
            "macro avg": "moyenne_macro",
            "weighted avg": "moyenne_ponderee",
        },
        columns={
            "recall": "rappel",
            "f1-score": "score_f1",
            "support": "nombre_exemples",
        },
    )
    labels = sorted(set(y_true) | set(predictions))
    # Fixer explicitement les labels garantit le même ordre sur les axes.
    matrix = pd.DataFrame(
        confusion_matrix(y_true, predictions, labels=labels),
        index=[f"reel_{label}" for label in labels],
        columns=[f"predit_{label}" for label in labels],
    )
    return metrics, report, matrix


def export_evaluation(
    model_path: Path = DEFAULT_MODEL_PATH,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, float]:
    """Charge le bundle, évalue le modèle et écrit les rapports sur disque."""
    require_file(model_path)
    bundle = joblib.load(model_path)
    metrics, report, matrix = evaluate_bundle(bundle)
    reports_dir.mkdir(parents=True, exist_ok=True)

    (reports_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report.to_csv(reports_dir / "classification_report.csv")
    matrix.to_csv(reports_dir / "confusion_matrix.csv")

    predictions = bundle["pipeline"].predict(bundle["x_test"])
    display = ConfusionMatrixDisplay.from_predictions(bundle["y_test"], predictions)
    display.figure_.tight_layout()
    display.figure_.savefig(reports_dir / "confusion_matrix.png", dpi=150)
    plt.close(display.figure_)
    LOGGER.info("Rapports d'évaluation enregistrés dans %s", reports_dir)
    return metrics
