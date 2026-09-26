# Model Card — Primary Diagnosis Classification

## Model
- Model: Random Forest + SMOTE + Feature Engineering
- Model type: RandomForestClassifier
- Task: Multiclass primary-diagnosis classification

## Intended use
Demonstrate primary-diagnosis prediction from structured patient information within the HealthAI portfolio application.

## Inputs / preprocessing
The finalized workflow uses feature engineering, one-hot encoding, scaling and SMOTE as recorded in the finalized model metadata.

## Evaluation
| Metric | Result |
|---|---:|
| Accuracy | 43.46% |
| Macro Precision | 0.2283 |
| Macro Recall | 0.2278 |
| Macro F1 | 0.2255 |
| Weighted F1 | 0.4181 |

## Model-selection history
An earlier XGBoost + SMOTE experiment recorded Macro F1 = 0.2315. The finalized application artifact and metadata identify Random Forest + SMOTE + feature engineering as the integrated model. The project documentation therefore does not claim that the finalized Random Forest had the highest Macro F1 across every experiment.

## Data provenance
Indian Healthcare Patient Records — Kaggle (Arun's Workspace), as documented in the project README.

## Limitations
- Multiclass class imbalance affects performance.
- The evaluation dataset is not a substitute for prospective clinical validation.
- Output is a machine-learning prediction, not a confirmed diagnosis.
