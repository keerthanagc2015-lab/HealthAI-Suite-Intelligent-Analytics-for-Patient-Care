# Model Card — Diabetes Risk Classification

## Model
- Model: Diabetes_RandomForest
- Framework: scikit-learn
- Task: Binary disease-risk classification
- Final application model: Random Forest

## Intended use
Estimate diabetes risk from structured patient attributes in the HealthAI portfolio application.

## Inputs / preprocessing
Structured patient health features are prepared through the project's modelling pipeline. The model is used through the application adapter/API rather than as a standalone clinical device.

## Evaluation
| Metric | Result |
|---|---:|
| Accuracy | 92.76% |
| Precision | 86.61% |
| Recall | 87.95% |
| F1-score | 87.28% |
| ROC-AUC | 96.62% |

## Data provenance
The main tabular healthcare workflows use the documented Indian Healthcare Patient Records dataset from Kaggle (Arun's Workspace). The project README records the source URL and notes that external data are kept outside Git history.

## Limitations
- Evaluation reflects the recorded project development/evaluation data.
- Performance may not generalize to real hospital populations.
- This is a risk estimate, not a medical diagnosis.

## Responsible use
Do not use this output as a substitute for clinical assessment, diagnosis, or treatment.
