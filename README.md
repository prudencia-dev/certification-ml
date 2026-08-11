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

## Documentation niveau 2

Le niveau 2 cible une architecture Python professionnelle pilotée par un
`main.py` unique, sans FastAPI ni Streamlit. Les documents suivants décrivent
la cible et son chemin de mise en œuvre ; le code actuel reste le socle niveau 1
tant que la migration n'est pas réalisée.

- [Architecture cible](docs/ARCHITECTURE_LEVEL_2.md) : composants,
  responsabilités, arborescence et contrats ;
- [Pipeline ML](docs/ML_PIPELINE.md) : flux des données, entraînement,
  évaluation, inférence et prévention des fuites ;
- [Guide pas à pas](docs/STEP_BY_STEP_GUIDE.md) : préparation, commandes,
  contrôles et checklist de livraison ;
- [Rapport d'entraînement](docs/TRAINING_REPORT.md) : gabarit traçable, sans
  métrique inventée ;
- [Standards de code et de documentation](docs/CODE_DOCUMENTATION_STANDARDS.md) :
  docstrings, commentaires pédagogiques, tests et journalisation.

## Organisation

- `src/preprocessing.py` : nettoyage structurel du CSV ;
- `src/train.py` : séparation des données, pipeline et entraînement ;
- `src/evaluate.py` : calcul et export des résultats ;
- `src/predict.py` : inférence sur une observation JSON ;
- `src/config.py` : chemins et paramètres ;
- `tests/` : tests unitaires et test de bout en bout ;
- `notebooks/` : notebooks pédagogiques à compléter avec le dataset retenu.
