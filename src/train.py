"""Entraînement reproductible du pipeline Random Forest.

Le preprocessing et le modèle sont réunis dans un seul pipeline scikit-learn.
Ainsi, les transformations apprises sur le jeu d'entraînement sont réutilisées
à l'identique lors de l'évaluation et de la prédiction.
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PROCESSED_DATASET,
    TARGET_COLUMN,
    TRAINING_CONFIG,
    TrainingConfig,
)
from src.preprocessing import validate_dataset
from src.utils import configure_logging, ensure_parent_directory, require_file

LOGGER = logging.getLogger(__name__)


def build_pipeline(
    features: pd.DataFrame,
    config: TrainingConfig = TRAINING_CONFIG,
) -> Pipeline:
    """Construit le preprocessing et le classifieur selon les types de colonnes.

    Les transformations sont ajustées plus tard par ``pipeline.fit``. Cette
    séparation évite la fuite de données : le jeu de test ne participe jamais
    au calcul des valeurs d'imputation ni à la création des catégories.
    """
    # La détection automatique permet d'adapter le socle au dataset public qui
    # sera retenu, sans écrire en dur une liste de variables encore inconnue.
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.columns.difference(numeric_columns).tolist()

    if not numeric_columns and not categorical_columns:
        raise ValueError("Aucune variable explicative n'est disponible.")

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        numeric_pipeline = Pipeline(
            [
                # La médiane résiste mieux que la moyenne aux valeurs extrêmes.
                ("imputer", SimpleImputer(strategy="median")),
                # Le Random Forest n'exige pas de standardisation, mais cette
                # étape rend le pipeline réutilisable avec d'autres modèles.
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline(
            [
                # Le mode fournit une règle simple et facile à justifier.
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    # Une catégorie nouvelle en production ne doit pas faire
                    # échouer la prédiction.
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    return Pipeline(
        [
            ("preprocessor", ColumnTransformer(transformers=transformers)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=config.n_estimators,
                    min_samples_leaf=config.min_samples_leaf,
                    # Une graine fixe rend l'expérience reproductible.
                    random_state=config.random_state,
                    # La pondération limite l'effet d'une cible déséquilibrée.
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_model(
    dataset_path: Path = DEFAULT_PROCESSED_DATASET,
    model_path: Path = DEFAULT_MODEL_PATH,
    target_column: str = TARGET_COLUMN,
    config: TrainingConfig = TRAINING_CONFIG,
) -> dict[str, Any]:
    """Sépare les données, entraîne le pipeline et sauvegarde ses artefacts.

    Le bundle contient le pipeline, son schéma d'entrée et le jeu de test. Cela
    garantit que ``evaluate.py`` mesure exactement les observations mises de
    côté avant l'entraînement.
    """
    require_file(dataset_path)
    dataframe = pd.read_csv(dataset_path)
    validate_dataset(dataframe, target_column)

    features = dataframe.drop(columns=target_column)
    target = dataframe[target_column]
    class_counts = target.value_counts()
    # La stratification conserve approximativement la proportion de chaque
    # niveau de risque dans les jeux d'entraînement et de test.
    can_stratify = class_counts.min() >= 2
    if not can_stratify:
        LOGGER.warning(
            "Séparation non stratifiée : une classe contient moins de deux exemples."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=target if can_stratify else None,
    )
    pipeline = build_pipeline(x_train, config)
    pipeline.fit(x_train, y_train)

    # Sauvegarder les colonnes attendues permet de reconstruire une observation
    # dans le même ordre au moment de l'inférence.
    bundle: dict[str, Any] = {
        "pipeline": pipeline,
        "feature_columns": features.columns.tolist(),
        "target_column": target_column,
        "x_test": x_test,
        "y_test": y_test,
        "training_config": config,
    }
    ensure_parent_directory(model_path)
    joblib.dump(bundle, model_path)
    LOGGER.info(
        "Modèle entraîné sur %s lignes et sauvegardé dans %s",
        len(x_train),
        model_path,
    )
    return bundle


def parse_args() -> argparse.Namespace:
    """Lit les arguments de la commande d'entraînement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_PROCESSED_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--target", default=TARGET_COLUMN)
    return parser.parse_args()


def main() -> None:
    """Point d'entrée du script d'entraînement."""
    configure_logging()
    args = parse_args()
    train_model(args.input, args.output, args.target)


if __name__ == "__main__":
    main()
