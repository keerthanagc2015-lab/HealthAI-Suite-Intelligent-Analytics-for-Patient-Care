# Model Card — ANN / MLP Length-of-Stay Workflow

## Model
- Model: Final Tuned ANN
- Framework: TensorFlow / Keras
- Task: Regression

## Intended use
Demonstrate a neural-network regression workflow for hospital length-of-stay prediction.

## Final saved evaluation
| Metric | Result |
|---|---:|
| MAE | 0.4424 |
| RMSE | 0.6371 |
| R² | 0.9260 |

## Tuning context
A separate hyperparameter-tuning table recorded a lowest test MAE of 0.4057 for a configuration using [128, 64, 32], learning rate 0.001, batch size 128 and dropout 0.1. This tuning result is not treated as the same evaluation as the separately saved final tuned ANN artifact.

## Limitations
- Performance reflects the recorded development/evaluation data.
- ANN results are not clinical validation evidence.
- Inputs and preprocessing must remain consistent with the training pipeline.
