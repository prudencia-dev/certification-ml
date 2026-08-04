# PRUDENCIA ML CERTIFICATION

Démonstrateur pédagogique et autonome d'un pipeline de Machine Learning qui
classe le niveau de risque d'un projet d'intelligence artificielle au regard de
l'AI Act.

> Ce projet illustre une démarche de Machine Learning. Il ne fournit pas de
> conseil juridique et ne remplace pas une analyse de conformité.

## État du projet

Le socle logiciel est opérationnel : nettoyage, entraînement, évaluation et
prédiction sont disponibles en ligne de commande. Le choix d'un dataset public,
sa licence et sa provenance doivent encore être validés avant l'entraînement
final.

## Installation

Python 3.12 ou supérieur est requis.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebooks]"
```

## Utilisation

1. Déposer un CSV public dans `data/raw/`.
2. Documenter sa provenance dans `data/SOURCE.md`.
3. Vérifier qu'il contient la cible `risk_level`.
4. Exécuter le pipeline :

```bash
python -m src.preprocessing --input data/raw/dataset.csv
python -m src.train
python -m src.evaluate
python -m src.predict --input-json '{"sector":"health","country":"FR"}'
```

Les variables numériques et catégorielles sont détectées automatiquement. Les
valeurs manquantes sont imputées dans le pipeline scikit-learn, les catégories
sont encodées, puis un `RandomForestClassifier` est entraîné.

## Sorties

- `data/processed/dataset_clean.csv` : données nettoyées ;
- `models/random_forest_pipeline.joblib` : modèle et métadonnées ;
- `reports/metrics.json` : métriques ;
- `reports/classification_report.csv` : rapport par classe ;
- `reports/confusion_matrix.csv` et `.png` : matrice de confusion.

## Qualité

```bash
pytest
ruff check .
```

## Organisation

- `src/preprocessing.py` : nettoyage structurel du CSV ;
- `src/train.py` : séparation des données, pipeline et entraînement ;
- `src/evaluate.py` : calcul et export des résultats ;
- `src/predict.py` : inférence sur une observation JSON ;
- `src/config.py` : chemins et paramètres ;
- `tests/` : tests unitaires et test de bout en bout ;
- `notebooks/` : notebooks pédagogiques à compléter avec le dataset retenu.

