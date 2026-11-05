# POMENOVA — Machine Learning Pipeline

## Objective

Explore the predicted inhibitory activity of ten pomegranate-derived compounds against human acetylcholinesterase (AChE), a therapeutic target investigated in Alzheimer's disease research.

All molecular activity predictions are computational and have not been experimentally validated.

## Data and methodology

- Data source: ChEMBL 37
- Target: Human acetylcholinesterase (CHEMBL220)
- Activity endpoint: IC50, converted to pIC50
- Curated dataset: 4,859 distinct compounds
- Molecular representation: Morgan fingerprints, radius 2, 2,048 bits, with chirality
- Data splitting: GroupShuffleSplit, 80% training / 20% testing, random state 42
- Models evaluated: Random Forest and XGBoost

## Model performance

| Model | MAE (pIC50 units) | R² |
|---|---:|---:|
| Random Forest | 0.5400 | 0.7350 |
| XGBoost | 0.6851 | 0.6291 |

Random Forest was selected for generating the exploratory predictions.

## Molecular validation

Ten pomegranate-related compounds were checked against PubChem compound identifiers and chemical structures, including a stereochemistry correction for rutin.

## Outputs

- `data/predictions.csv`: predicted pIC50 and IC50 values, tree prediction dispersion, and similarity information.
- `data/pubchem_verification.csv`: PubChem structure verification.
- `data/pomegranate_descriptors_verified.csv`: verified molecular structures.
- `models/random_forest_v1.joblib`: selected trained Random Forest model.
- `figures/`: model evaluation and compound ranking plots.

## Scientific limitations

The predictions do not establish experimental AChE inhibition, bioavailability, clinical efficacy, or Alzheimer's disease treatment potential. Several investigated compounds have low structural similarity to the training data. Prediction dispersion across Random Forest trees is not a calibrated confidence interval. Additional scaffold-based evaluation and experimental validation are recommended.
## Additional Validation — Bemis-Murcko Scaffold Split

To evaluate generalization to unseen molecular scaffolds,
we performed an additional scaffold-based validation.

Dataset:
- 4,859 molecules
- 2,103 distinct scaffold groups
- 3,901 training molecules
- 958 testing molecules
- 0 overlapping scaffold groups

### Results

| Model | MAE | R² |
|---|---:|---:|
| Random Forest | 0.5935 | 0.6777 |
| XGBoost | 0.7013 | 0.5924 |

Random Forest remained the best-performing model.

The reduction in performance compared with the initial
split highlights the difficulty of generalizing to
unseen molecular scaffolds.

These results are based on one scaffold split and do
not establish experimental activity or clinical efficacy.
