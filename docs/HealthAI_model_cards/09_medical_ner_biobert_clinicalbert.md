# Model Card — Medical Named-Entity Recognition

## Models evaluated
- BioBERT: dmis-lab/biobert-v1.1
- ClinicalBERT: emilyalsentzer/Bio_ClinicalBERT

## Intended use
Extract medical entities from clinical-style text for portfolio/educational NLP demonstrations.

## Evaluation
| Model | Token Accuracy | Entity Precision | Entity Recall | Entity F1 |
|---|---:|---:|---:|---:|
| BioBERT | 78.16% | 41.01% | 46.02% | 43.37% |
| ClinicalBERT | 77.14% | 36.87% | 44.08% | 40.16% |

## Model selection
BioBERT was retained because its recorded metrics were higher across the listed evaluation measures.

## Data provenance
Multilingual Synthetic Medical Notes for NER — Hugging Face (E3-JSI/synthetic-multi-med-notes-ner-v1), recorded in the project README as MIT.

## Limitations
- Entity-level performance is dataset-dependent.
- Extracted entities are not clinical facts guaranteed to be correct.
- Human review would be required for real clinical workflows.
