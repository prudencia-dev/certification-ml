
# Prédiction de la gravité d'un incident pour une observation JSON.
from pathlib import Path
from typing import Any

# Le module joblib est utilisé pour la sérialisation et la désérialisation d'objets Python, notamment pour sauvegarder et charger des modèles de machine learning.
import joblib
import pandas as pd

# Le module config contient des constantes et des paramètres de configuration pour le projet, tels que les chemins par défaut vers les fichiers de modèle et de données.
from src.config import DEFAULT_MODEL_PATH
from src.utils import require_file


# Le module logging est utilisé pour enregistrer des messages d'information, d'avertissement ou d'erreur pendant l'exécution du script.
def predict_observation(
    observation: dict[str, Any],
    model_path: Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    require_file(model_path)
    bundle = joblib.load(model_path)
    expected_columns = bundle["feature_columns"]
    unknown_columns = sorted(set(observation) - set(expected_columns))

    # On lève une erreur si l'observation contient des colonnes inconnues,
    # afin d'éviter des prédictions incorrectes ou incohérentes.
    if unknown_columns:
        raise ValueError(f"Variables inconnues : {unknown_columns}")

    # L'ordre et la présence des colonnes reproduisent exactement le schéma
    # mémorisé lors de l'entraînement.
    row = pd.DataFrame(
        [{column: observation.get(column, pd.NA) for column in expected_columns}]
    )
    pipeline = bundle["pipeline"]
    prediction = pipeline.predict(row)[0]
    result: dict[str, Any] = {"gravite": str(prediction)}

    # Les probabilités rendent le résultat plus interprétable qu'une classe
    # seule, sans les présenter comme une certitude juridique.
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(row)[0]
        classes = pipeline.classes_
        result["probabilites"] = {
            str(label): float(probability)
            for label, probability in zip(classes, probabilities, strict=True)
        }
    return result
