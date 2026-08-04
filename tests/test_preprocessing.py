"""Tests du nettoyage des données."""

import pandas as pd
import pytest

from src.preprocessing import clean_dataset, validate_dataset


def test_clean_dataset_normalizes_text_columns_and_duplicates() -> None:
    """Le nettoyage harmonise les colonnes, espaces et doublons."""
    raw = pd.DataFrame(
        {
            "Risk Level": [" high ", " high ", "low"],
            "AI Sector": [" health ", " health ", "  "],
        }
    )

    cleaned = clean_dataset(raw)

    assert cleaned.columns.tolist() == ["risk_level", "ai_sector"]
    assert len(cleaned) == 2
    assert cleaned.loc[0, "ai_sector"] == "health"
    assert pd.isna(cleaned.loc[1, "ai_sector"])


def test_validate_dataset_requires_target() -> None:
    """Une cible absente produit un message explicite."""
    with pytest.raises(ValueError, match="Colonne cible absente"):
        validate_dataset(pd.DataFrame({"feature": [1, 2]}), "risk_level")

