"""Fonctions utilitaires partagées par les scripts."""

import logging
from pathlib import Path


def configure_logging() -> None:
    """Configure un format de journalisation concis pour les scripts."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_parent_directory(path: Path) -> None:
    """Crée le dossier parent d'un fichier lorsqu'il n'existe pas."""
    path.parent.mkdir(parents=True, exist_ok=True)


def require_file(path: Path) -> None:
    """Lève une erreur explicite lorsque le fichier demandé est absent."""
    if not path.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

