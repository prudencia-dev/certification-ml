"""Test de bout en bout du pipeline ML."""

import json
from pathlib import Path

import pandas as pd

from src.config import TrainingConfig
from src.evaluate import export_evaluation
from src.predict import predict_observation
from src.preprocessing import preprocess_file
from src.train import train_model


def test_end_to_end_pipeline(tmp_path: Path) -> None:
    """Le pipeline nettoie, entraîne, évalue et prédit."""
    rows = []
    for index in range(60):
        rows.append(
            {
                "Sector": "health" if index % 2 else "education",
                "Personal Data": "yes" if index % 3 else "no",
                "People Count": index * 10,
                "Risk Level": "high" if index % 2 else "limited",
            }
        )
    raw_path = tmp_path / "raw.csv"
    clean_path = tmp_path / "clean.csv"
    model_path = tmp_path / "model.joblib"
    reports_path = tmp_path / "reports"
    pd.DataFrame(rows).to_csv(raw_path, index=False)

    preprocess_file(raw_path, clean_path)
    train_model(
        clean_path,
        model_path,
        config=TrainingConfig(test_size=0.25, random_state=7, n_estimators=20),
    )
    metrics = export_evaluation(model_path, reports_path)
    prediction = predict_observation(
        {"sector": "health", "personal_data": "yes", "people_count": 100},
        model_path,
    )

    assert 0 <= metrics["accuracy"] <= 1
    assert prediction["risk_level"] in {"high", "limited"}
    assert (reports_path / "confusion_matrix.png").is_file()
    assert json.loads((reports_path / "metrics.json").read_text())["test_samples"] == 15

