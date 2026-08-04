"""Évaluation et export des performances du modèle.

Plusieurs métriques sont exportées car l'accuracy seule peut masquer de mauvais
résultats sur une classe minoritaire.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib

# Le backend non interactif permet de produire le graphique sur un serveur ou
# dans une CI, sans écran ni fenêtre graphique.
matplotlib.use("Agg")

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
from src.utils import configure_logging, require_file

LOGGER = logging.getLogger(__name__)


def evaluate_bundle(
    bundle: dict[str, Any],
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Calcule les métriques globales, le rapport par classe et la confusion.

    La moyenne pondérée tient compte du nombre d'observations de chaque classe.
    Le rapport détaillé reste indispensable pour repérer une classe délaissée.
    """
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
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1_score),
        "test_samples": int(len(y_true)),
    }
    report = pd.DataFrame(
        classification_report(y_true, predictions, output_dict=True, zero_division=0)
    ).transpose()
    labels = sorted(set(y_true) | set(predictions))
    # Fixer explicitement les labels garantit le même ordre sur les axes.
    matrix = pd.DataFrame(
        confusion_matrix(y_true, predictions, labels=labels),
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
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


def parse_args() -> argparse.Namespace:
    """Lit les arguments de la commande d'évaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    return parser.parse_args()


def main() -> None:
    """Point d'entrée du script d'évaluation."""
    configure_logging()
    args = parse_args()
    export_evaluation(args.model, args.reports_dir)


if __name__ == "__main__":
    main()
