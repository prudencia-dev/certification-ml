"""Entraînement reproductible du pipeline Random Forest.

Le preprocessing et le modèle sont réunis dans un seul pipeline scikit-learn.
Ainsi, les transformations apprises sur le jeu d'entraînement sont réutilisées
à l'identique lors de l'évaluation et de la prédiction.
"""

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
from src.utils import ensure_parent_directory, require_file

LOGGER = logging.getLogger(__name__)


# Cette fonction construit la chaîne de traitement du modèle.
def build_pipeline(
    features: pd.DataFrame,
    config: TrainingConfig = TRAINING_CONFIG,
) -> Pipeline:

    # La séparation automatique des colonnes numériques et catégorielles permet de créer un pipeline flexible, capable de s'adapter à différents jeux de données sans nécessiter de modifications manuelles.
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.columns.difference(numeric_columns).tolist()

    # On lève une erreur si aucune variable explicative n'est disponible, car un modèle ne peut pas être entraîné sans données d'entrée.
    if not numeric_columns and not categorical_columns:
        raise ValueError("Aucune variable explicative n'est disponible.")

    # Crée une liste vide qui contiendra les différents traitements à appliquer aux colonne
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


# Fonction principale qui entraîne le modèle et sauvegarde le pipeline et les données de test.
def train_model(
    dataset_path: Path = DEFAULT_PROCESSED_DATASET,
    model_path: Path = DEFAULT_MODEL_PATH,
    target_column: str = TARGET_COLUMN,
    config: TrainingConfig = TRAINING_CONFIG,
) -> dict[str, Any]:
    require_file(dataset_path)
    dataframe = pd.read_csv(dataset_path)
    validate_dataset(dataframe, target_column)

    # variables explicatives et cible sont séparées pour l'entraînement et l'évaluation.
    features = dataframe.drop(columns=target_column)

    # La stratification conserve approximativement la proportion de chaque
    # niveau de risque dans les jeux d'entraînement et de test.
    target = dataframe[target_column]

    # Le comptage vérifie que la stratification est possible.
    class_counts = target.value_counts()

    #  La stratification conserve approximativement la proportion de chaque
    # niveau de risque dans les jeux d'entraînement et de test.
    can_stratify = class_counts.min() >= 2
    if not can_stratify:
        LOGGER.warning(
            "Séparation non stratifiée : une classe contient moins de deux exemples."
        )

    # Séparation des données en ensembles d'entraînement et de test, avec une proportion définie par le paramètre test_size. Le paramètre random_state assure la reproductibilité de la séparation.
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=target if can_stratify else None,
    )

    # Construction du pipeline de traitement et d'entraînement du modèle.
    pipeline = build_pipeline(x_train, config)

    # Entraînement du modèle sur les données d'entraînement. Le pipeline applique d'abord les transformations aux données, puis ajuste le modèle de forêt aléatoire aux données transformées.
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

    # Déterminer le répertoire parent du chemin du modèle et le créer s'il n'existe pas, afin d'éviter des erreurs lors de la sauvegarde du modèle.
    ensure_parent_directory(model_path)
    joblib.dump(bundle, model_path)
    LOGGER.info(
        "Modèle entraîné sur %s lignes et sauvegardé dans %s",
        len(x_train),
        model_path,
    )
    return bundle
