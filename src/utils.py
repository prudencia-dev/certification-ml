
import logging
from pathlib import Path


# On configure un format de journalisation concis pour les scripts.
def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# On crée un logger pour le module utils.py, qui peut être utilisé pour enregistrer des messages d'information, d'avertissement ou d'erreur.
def ensure_parent_directory(path: Path) -> None:
    """Crée le dossier parent d'un fichier lorsqu'il n'existe pas."""
    path.parent.mkdir(parents=True, exist_ok=True)


# On crée un logger pour le module utils.py, qui peut être utilisé pour enregistrer des messages d'information, d'avertissement ou d'erreur.
def require_file(path: Path) -> None:
    """Lève une erreur explicite lorsque le fichier demandé est absent."""
    if not path.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
