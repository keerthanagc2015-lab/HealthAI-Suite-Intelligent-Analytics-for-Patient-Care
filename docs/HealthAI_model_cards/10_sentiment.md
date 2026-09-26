# Model Card — Patient Sentiment

## Evaluated workflow
- Recorded evaluation: TF-IDF + Logistic Regression
- Additional project workflow: DistilBERT-based sentiment workflow
- Task: Positive / negative patient-feedback classification

## Recorded TF-IDF + Logistic Regression evaluation
| Metric | Result |
|---|---:|
| Accuracy | 50.00% |
| Precision | 50.00% |
| Recall | 100.00% |
| F1-score | 66.67% |

## Important evaluation note
The project contains a DistilBERT sentiment workflow, but the current recorded artifacts do not provide directly comparable final DistilBERT metrics alongside the TF-IDF results. Therefore, no performance ranking between the two approaches is claimed.

## Intended use
Demonstrate patient-feedback sentiment classification and support exploratory service-quality analysis.

## Limitations
- Evaluation uses a controlled dataset with limited unique feedback texts.
- Sentiment labels are not clinical diagnoses.
- Results may not generalize to broader patient populations.
