"""Prédiction du niveau de risque pour une observation JSON."""

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import DEFAULT_MODEL_PATH
from src.utils import require_file


def predict_observation(
    observation: dict[str, Any],
    model_path: Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    """Charge le modèle et prédit une observation après validation du schéma.

    Les variables manquantes sont acceptées : elles seront traitées par les
    imputeurs appris pendant l'entraînement. À l'inverse, une variable inconnue
    est probablement une faute de saisie et provoque une erreur explicite.
    """
    require_file(model_path)
    bundle = joblib.load(model_path)
    expected_columns = bundle["feature_columns"]
    unknown_columns = sorted(set(observation) - set(expected_columns))
    if unknown_columns:
        raise ValueError(f"Variables inconnues : {unknown_columns}")

    # L'ordre et la présence des colonnes reproduisent exactement le schéma
    # mémorisé lors de l'entraînement.
    row = pd.DataFrame(
        [{column: observation.get(column, pd.NA) for column in expected_columns}]
    )
    pipeline = bundle["pipeline"]
    prediction = pipeline.predict(row)[0]
    result: dict[str, Any] = {"risk_level": str(prediction)}

    # Les probabilités rendent le résultat plus interprétable qu'une classe
    # seule, sans les présenter comme une certitude juridique.
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(row)[0]
        classes = pipeline.classes_
        result["probabilities"] = {
            str(label): float(probability)
            for label, probability in zip(classes, probabilities, strict=True)
        }
    return result


def parse_args() -> argparse.Namespace:
    """Lit l'observation JSON et le chemin du modèle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True, help="Objet JSON")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


def main() -> None:
    """Point d'entrée du script de prédiction."""
    args = parse_args()
    observation = json.loads(args.input_json)
    if not isinstance(observation, dict):
        raise ValueError("L'entrée JSON doit être un objet.")
    print(json.dumps(predict_observation(observation, args.model), ensure_ascii=False))


if __name__ == "__main__":
    main()
