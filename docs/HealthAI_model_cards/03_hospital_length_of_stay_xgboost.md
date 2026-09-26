# Model Card — Hospital Length of Stay

## Model
- Model: XGBoost
- Task: Regression
- Selection rule: Lowest recorded test MAE

## Intended use
Estimate expected hospital length of stay within the HealthAI portfolio application.

## Evaluation
| Metric | Result |
|---|---:|
| MAE | 0.3201 |
| RMSE | 0.4329 |
| R² | 0.9659 |

## Comparison context
The documented model comparison recorded XGBoost as the selected configuration by lowest test MAE. Standard SVR was not retained because of excessive runtime on the 80k-row dataset.

## Data provenance
LengthOfStay.csv — Microsoft R Server Hospital Length of Stay. The README records the source and MIT license and notes that Microsoft describes the data as synthetic, modeled after real-world hospital inpatient records.

## Limitations
- The target is learned from a synthetic/public development dataset.
- Predictions can differ from real hospital length-of-stay distributions.
- Output is decision-support information, not a guaranteed admission duration.
