# Model Card — Chest X-ray CNN

## Model
- Model: Chest_XRay_CNN
- Framework: TensorFlow / Keras
- Task: Image classification
- Classes: Normal / Pneumonia

## Intended use
Portfolio/educational demonstration of CNN-based chest X-ray classification using a synthetic chest X-ray dataset.

## Evaluation
| Metric | Result |
|---|---:|
| Accuracy | 99.17% |
| Precision | 99.17% |
| Recall | 99.17% |
| F1-score | 99.17% |

## Explainability artifacts
The project contains Grad-CAM-related explanation artifacts, including xray_gradcam_explanation.png and xray_gradcam_heatmap.png.

## Data provenance
Synthetic Chest X-Ray Pneumonia Dataset — Hugging Face (chimbiwide/synthetic-chest-xray-pneumonia), recorded in the project README as CC BY 4.0.

## Critical limitation
The dataset is synthetic and the system is not clinically validated. The application explicitly presents this as a portfolio demonstration and not as a clinical diagnosis or medical decision-making tool.

## Responsible use
Do not use this model for real patient diagnosis or treatment decisions.
