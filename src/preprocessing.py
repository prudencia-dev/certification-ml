"""Nettoyage structurel du dataset brut.

Ce module conserve volontairement les transformations métier complexes hors du
code générique. Il applique seulement des règles explicables et reproductibles,
sans modifier le sens des observations.
"""

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

from src.config import DEFAULT_PROCESSED_DATASET, TARGET_COLUMN
from src.utils import configure_logging, ensure_parent_directory, require_file

LOGGER = logging.getLogger(__name__)


def normalize_column_name(name: str) -> str:
    """Convertit un nom de colonne en snake_case simple."""
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
    return normalized.strip("_")


def load_dataset(path: Path) -> pd.DataFrame:
    """Charge un dataset CSV après avoir vérifié son existence."""
    require_file(path)
    return pd.read_csv(path)


def clean_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les noms, textes vides, espaces et lignes dupliquées.

    Une copie protège le DataFrame reçu : l'appelant peut ainsi comparer les
    données avant et après nettoyage sans effet de bord.
    """
    cleaned = dataframe.copy()

    # Des noms homogènes évitent qu'une même variable soit appelée différemment
    # entre le notebook, l'entraînement et l'inférence.
    cleaned.columns = [normalize_column_name(column) for column in cleaned.columns]

    if any(not column for column in cleaned.columns):
        raise ValueError("Au moins une colonne possède un nom vide.")
    if cleaned.columns.duplicated().any():
        duplicates = cleaned.columns[cleaned.columns.duplicated()].tolist()
        raise ValueError(
            f"Noms de colonnes dupliqués après normalisation : {duplicates}"
        )

    text_columns = cleaned.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        # "health" et " health " doivent représenter la même catégorie.
        cleaned[column] = cleaned[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
        # Une chaîne vide devient une vraie valeur manquante. Le pipeline pourra
        # alors l'imputer de manière uniforme pendant l'entraînement.
        cleaned[column] = cleaned[column].replace(r"^\s*$", pd.NA, regex=True)

    # Les doublons fausseraient la distribution et pourraient surévaluer le
    # modèle si une même observation apparaissait dans train et test.
    cleaned = cleaned.drop_duplicates().dropna(how="all").reset_index(drop=True)
    return cleaned


def validate_dataset(dataframe: pd.DataFrame, target_column: str) -> None:
    """Vérifie les préconditions minimales nécessaires à une classification."""
    if dataframe.empty:
        raise ValueError("Le dataset nettoyé est vide.")
    if target_column not in dataframe.columns:
        raise ValueError(f"Colonne cible absente : {target_column}")
    if dataframe[target_column].isna().any():
        # Sans étiquette connue, une ligne ne peut pas superviser le modèle.
        raise ValueError("La colonne cible contient des valeurs manquantes.")
    if dataframe[target_column].nunique() < 2:
        raise ValueError("La cible doit contenir au moins deux classes.")


def preprocess_file(
    input_path: Path,
    output_path: Path = DEFAULT_PROCESSED_DATASET,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Charge, nettoie, valide puis sauvegarde un dataset CSV."""
    dataframe = load_dataset(input_path)
    cleaned = clean_dataset(dataframe)
    validate_dataset(cleaned, target_column)
    ensure_parent_directory(output_path)
    cleaned.to_csv(output_path, index=False)
    LOGGER.info("%s lignes nettoyées enregistrées dans %s", len(cleaned), output_path)
    return cleaned


def parse_args() -> argparse.Namespace:
    """Lit les arguments de la commande de preprocessing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV brut")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PROCESSED_DATASET,
        help="CSV nettoyé",
    )
    parser.add_argument("--target", default=TARGET_COLUMN, help="Colonne cible")
    return parser.parse_args()


def main() -> None:
    """Point d'entrée du script de preprocessing."""
    configure_logging()
    args = parse_args()
    preprocess_file(args.input, args.output, args.target)


if __name__ == "__main__":
    main()
