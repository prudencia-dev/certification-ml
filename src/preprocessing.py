
# Ce module fournit des fonctions pour le prétraitement des datasets, notamment le nettoyage, la validation et la préparation des caractéristiques pour l'entraînement d'un modèle de classification. Il inclut également des fonctions utilitaires pour normaliser les noms de colonnes et compter les éléments dans des listes JSON ou des textes séparés par des barres verticales.
import json
import logging
import re
from pathlib import Path

# Le module pandas est utilisé pour la manipulation et l'analyse de données, notamment pour le traitement des datasets CSV.
import pandas as pd

# Le module config contient des constantes et des paramètres de configuration pour le projet, tels que les chemins par défaut vers les fichiers de modèle et de données.
from src.config import (
    DEFAULT_PROCESSED_DATASET,
    TARGET_COLUMN,
)
from src.utils import ensure_parent_directory, require_file

LOGGER = logging.getLogger(__name__)


# Fonction utilitaire pour normaliser les noms de colonnes en snake_case, ce qui facilite la manipulation des données et évite les erreurs liées à des noms de colonnes incohérents.
def normalize_column_name(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
    return normalized.strip("_")

# Fonction utilitaire pour compter le nombre d'éléments dans une valeur, qu'il s'agisse d'une liste JSON ou d'un texte séparé par des barres verticales. Les valeurs vides ou nulles sont comptées comme zéro.
def compter_elements(value: object) -> int:
    # Les valeurs vides ou nulles sont comptées comme zéro. Les chaînes de caractères sont d'abord nettoyées des espaces superflus avant d'être analysées. Si la valeur est un JSON valide représentant une liste, le nombre d'éléments dans cette liste est retourné. Sinon, la chaîne est divisée par les barres verticales et le nombre d'éléments non vides est compté.
    if pd.isna(value) or not str(value).strip():
        return 0
    texte = str(value).strip()
    try:
        elements = json.loads(texte)
    except json.JSONDecodeError:
        elements = [element.strip() for element in texte.split("|")]
    if not isinstance(elements, list):
        elements = [elements]
    return len([element for element in elements if str(element).strip()])


# Fonction pour charger un dataset CSV après avoir vérifié son existence. Elle utilise la fonction require_file pour s'assurer que le fichier est présent avant de le lire avec pandas.
def load_dataset(path: Path) -> pd.DataFrame:
    require_file(path)
    return pd.read_csv(path)


# Fonction pour nettoyer un DataFrame en normalisant les noms de colonnes, en supprimant les espaces superflus dans les chaînes de caractères, en remplaçant les chaînes vides par des valeurs manquantes et en supprimant les lignes dupliquées. Une copie du DataFrame est créée pour éviter les effets de bord sur l'appelant.
def clean_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:

    cleaned = dataframe.copy()

    # Des noms homogènes évitent qu'une même variable soit appelée différemment
    # entre le prétraitement, l'entraînement et l'inférence.
    cleaned.columns = [normalize_column_name(column) for column in cleaned.columns]

    # On lève une erreur si une colonne n'a pas de nom ou si des noms de colonnes sont dupliqués après normalisation, afin d'éviter des ambiguïtés lors de l'accès aux colonnes.
    if any(not column for column in cleaned.columns):
        raise ValueError("Au moins une colonne possède un nom vide.")
    if cleaned.columns.duplicated().any():
        duplicates = cleaned.columns[cleaned.columns.duplicated()].tolist()
        raise ValueError(
            f"Noms de colonnes dupliqués après normalisation : {duplicates}"
        )

    # Les colonnes de type texte sont nettoyées pour supprimer les espaces superflus et les chaînes vides sont remplacées
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


# Fonction pour valider un DataFrame en vérifiant qu'il n'est pas vide, que la colonne cible est présente et ne contient pas de valeurs manquantes, et que la colonne cible contient au moins deux classes distinctes. Ces vérifications sont essentielles pour garantir que le dataset est approprié pour l'entraînement d'un modèle de classification.
def validate_dataset(dataframe: pd.DataFrame, target_column: str) -> None:

    # On lève une erreur si le dataset est vide, si la colonne cible est absente ou contient des valeurs manquantes, ou si la colonne cible ne contient pas au moins deux classes distinctes. Ces vérifications sont essentielles pour garantir que le dataset est approprié pour l'entraînement d'un modèle de classification.
    if dataframe.empty:
        raise ValueError("Le dataset nettoyé est vide.")
    if target_column not in dataframe.columns:
        raise ValueError(f"Colonne cible absente : {target_column}")
    if dataframe[target_column].isna().any():
        # Sans étiquette connue, une ligne ne peut pas superviser le modèle.
        raise ValueError("La colonne cible contient des valeurs manquantes.")
    if dataframe[target_column].nunique() < 2:
        raise ValueError("La cible doit contenir au moins deux classes.")


# Fonction pour préparer les caractéristiques d'un dataset d'incidents AI en conservant uniquement les variables disponibles avant de connaître la gravité. Elle vérifie que toutes les colonnes requises sont présentes, normalise certaines colonnes, extrait l'année des dates et compte le nombre d'éléments dans certaines colonnes. Le DataFrame résultant contient uniquement les caractéristiques pertinentes pour l'entraînement du modèle.
def prepare_ai_incident_features(dataframe: pd.DataFrame) -> pd.DataFrame:

    required = {
        "gravite",
        "categorie",
        "titre",
        "date_survenue",
        "date_publication",
        "nom_source",
        "developpeurs",
        "deployeurs",
        "parties_lesees",
        "etiquettes",
    }
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(f"Colonnes AI Incidents absentes : {missing}")

    features = pd.DataFrame(index=dataframe.index)
    for column in ("categorie", "nom_source"):
        # Une colonne catégorielle totalement vide (notamment ``location``
        # dans certaines versions de la source) doit rester catégorielle après
        # l'aller-retour CSV. Sinon pandas la relit comme un nombre et le
        # pipeline tente à tort une imputation par médiane.
        features[column] = dataframe[column].fillna("non renseigné").astype(str)
    for source, output in (
        ("date_survenue", "annee_survenue"),
        ("date_publication", "annee_publication"),
    ):
        features[output] = pd.to_datetime(dataframe[source], errors="coerce").dt.year
    features["longueur_titre"] = dataframe["titre"].fillna("").str.len()
    for source, output in (
        ("developpeurs", "nombre_developpeurs"),
        ("deployeurs", "nombre_deployeurs"),
        ("parties_lesees", "nombre_parties_lesees"),
        ("etiquettes", "nombre_etiquettes"),
    ):
        features[output] = dataframe[source].map(compter_elements)
    features["gravite"] = dataframe["gravite"]
    return features


# Fonction pour prétraiter un fichier CSV en le chargeant, le nettoyant, le validant et en sauvegardant le dataset résultant. Elle utilise les fonctions précédemment définies pour effectuer chaque étape du prétraitement et retourne le DataFrame nettoyé.
def preprocess_file(
    input_path: Path,
    output_path: Path = DEFAULT_PROCESSED_DATASET,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:

    # Charge, nettoie, valide puis sauvegarde un dataset CSV.
    dataframe = load_dataset(input_path)
    cleaned = clean_dataset(dataframe)

    validate_dataset(cleaned, target_column)
    cleaned = prepare_ai_incident_features(cleaned)

    ensure_parent_directory(output_path)
    cleaned.to_csv(output_path, index=False)

    LOGGER.info("%s lignes nettoyées enregistrées dans %s", len(cleaned), output_path)

    return cleaned
