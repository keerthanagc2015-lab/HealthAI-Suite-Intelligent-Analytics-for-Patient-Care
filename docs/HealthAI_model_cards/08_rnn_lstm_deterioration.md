# Model Card — RNN / LSTM Patient Deterioration Workflow

## Models
- Primary sequential model: RNN
- Comparison model: LSTM
- Task: Sequential patient deterioration modelling

## RNN evaluation
| Metric | Result |
|---|---:|
| Accuracy | 84.53% |
| Precision | 32.70% |
| Recall | 61.50% |
| F1-score | 42.70% |
| ROC-AUC | 81.99% |

## LSTM evaluation
| Metric | Result |
|---|---:|
| Accuracy | 84.75% |
| Precision | 31.72% |
| Recall | 54.47% |
| F1-score | 40.09% |
| ROC-AUC | 79.36% |

## Interpretation
The recorded results show different strengths across metrics. The project retains the RNN as its primary sequential model and reports the LSTM comparison without claiming a universal winner.

## Data provenance
Hospital Deterioration — Simulated Early Warning dataset from Hugging Face (tarekmasryo/hospital-deterioration-dataset), recorded in the project README as CC BY 4.0.

## Limitations
- Requires further validation before any real clinical use.
- Sequence-model performance depends on the simulated/development data distribution.
