# HealthAI Suite — Model Evaluation and Selection

## 1. Evaluation Strategy

HealthAI uses multiple AI and machine-learning approaches across
classification, regression, clustering, association mining,
deep learning, medical NLP, RAG and agentic orchestration.

Models and configurations were evaluated using metrics appropriate
to the corresponding task rather than relying on accuracy alone.

The general selection principles were:

- Classification: Accuracy, Precision, Recall, F1-score and ROC-AUC,
  with Macro F1 receiving greater importance for imbalanced
  multiclass tasks.
- Regression: MAE, RMSE and R².
- Clustering: Silhouette Score and interpretability of the resulting
  patient groups.
- Association mining: Support, Confidence and Lift, followed by
  domain-specific rule filtering.
- Deep learning: Validation and test metrics, together with training
  behaviour and saved evaluation artifacts.
- Medical NER: Entity-level Precision, Recall and F1-score.
- RAG: Retrieval accuracy, topic consistency, source acceptance,
  out-of-domain rejection and safety behaviour.
- Agentic AI: Routing accuracy, execution validity and end-to-end
  integration success.

The final model or configuration used by the application was based on
the recorded project artifacts and finalized model metadata.

---

# 2. Classical Machine Learning

## 2.1 Diabetes Risk Classification

### Final Model

**Diabetes_RandomForest**

Framework: Scikit-learn

### Final Evaluation

| Metric | Result |
|---|---:|
| Accuracy | 92.76% |
| Precision | 86.61% |
| Recall | 87.95% |
| F1-score | 87.28% |
| ROC-AUC | 96.62% |

### Model Selection

The finalized diabetes-risk model is the Random Forest model recorded
in the project's MLflow tracking summary and model artifact metadata.

The model was retained with a strong overall metric profile,
including an ROC-AUC of 0.9662 and an F1-score of 0.8728.

The final application therefore uses the saved Random Forest artifact
for the diabetes-risk workflow.

---

## 2.2 Primary Diagnosis Classification

### Final Integrated Model

**Random Forest + SMOTE + Feature Engineering**

Model type: `RandomForestClassifier`

### Final Evaluation

| Metric | Result |
|---|---:|
| Accuracy | 43.46% |
| Macro Precision | 0.2283 |
| Macro Recall | 0.2278 |
| Macro F1 | 0.2255 |
| Weighted F1 | 0.4181 |

### Model Selection History

The primary-diagnosis problem is a multiclass and imbalanced
classification task. Macro F1 was therefore used as the primary
evaluation metric, with Accuracy, Macro Precision, Macro Recall and
Weighted F1 used as supporting metrics.

During the earlier advanced model-comparison stage, multiple
classification configurations were evaluated.

An earlier XGBoost + SMOTE configuration recorded:

- Accuracy: 42.81%
- Macro F1: 0.2315
- Weighted F1: 0.4229

This configuration had the strongest recorded Macro F1 among the
earlier evaluated configurations and was selected as the initial
classification candidate.

A later XGBoost + SMOTE + Feature Engineering configuration recorded:

- Accuracy: 45.43%
- Macro F1: 0.2099
- Weighted F1: 0.4101

The finalized application workflow was subsequently implemented using
**Random Forest + SMOTE + Feature Engineering**, with the saved model
artifact and metadata identifying Random Forest as the model used by
the final primary-diagnosis workflow.

Final integrated-model evaluation:

- Accuracy: 43.46%
- Macro Precision: 0.2283
- Macro Recall: 0.2278
- Macro F1: 0.2255
- Weighted F1: 0.4181

This documentation distinguishes the earlier candidate-selection stage
from the later integrated model so that different development stages
are not presented as the same selection event.

### Limitation

The recorded results show limited performance for minority diagnosis
classes. The model is therefore intended for research and educational
demonstration rather than clinical diagnosis.

---

## 2.3 Hospital Length-of-Stay Regression

### Final Model

**XGBoost**

### Selection Criterion

**Lowest test MAE**

### Final Evaluation

| Metric | Result |
|---|---:|
| MAE | 0.3201 |
| RMSE | 0.4329 |
| R² | 0.9659 |

### Model Selection

Multiple regression models were evaluated, including XGBoost,
Decision Tree, Random Forest, Ridge Regression, Linear Regression,
ElasticNet and Lasso.

XGBoost was selected because it achieved the lowest recorded test
MAE of 0.3201. It also achieved an R² of 0.9659 and the lowest RMSE
among the evaluated models.

The final XGBoost model was saved as the Hospital Length-of-Stay
inference artifact.

A standard SVR experiment was skipped because it scaled poorly for the
80,000-row training set and resulted in excessive runtime.

---

## 2.4 Patient Clustering

### Final Configuration

**K = 2 clusters**

### Selection Criterion

**Highest Silhouette Score**

### Silhouette Evaluation

| K | Silhouette Score |
|---:|---:|
| 2 | 0.1544 |
| 3 | 0.1013 |
| 4 | 0.0924 |
| 5 | 0.0886 |
| 6 | 0.0794 |
| 7 | 0.0765 |
| 8 | 0.0709 |
| 9 | 0.0701 |
| 10 | 0.0671 |

K = 2 achieved the highest recorded Silhouette Score and was used for
the final clustering workflow.

### Cluster Profiles

| Cluster | Patients | Percentage | Avg Glucose | Avg HbA1c | Type 2 Diabetes |
|---|---:|---:|---:|---:|---:|
| 0 | 25,546 | 25.55% | 186.03 | 8.50 | 91.53% |
| 1 | 74,454 | 74.45% | 107.68 | 5.68 | 6.49% |

The cluster profiles describe observed differences within the dataset.
Clustering does not establish causality or provide an independent
clinical diagnosis.

---

## 2.5 Association Rule Mining

Association analysis does not have a single predictive model winner.
The evaluation instead focuses on frequent itemsets, association-rule
generation and progressive filtering.

### Initial Mining Configuration

| Measure | Result |
|---|---:|
| Transactions | 100,000 |
| Unique Items | 39 |
| Minimum Support | 0.01 |
| Minimum Patient Count | 1,000 |
| Frequent Itemsets | 858 |

### Rule Generation and Filtering

| Stage | Number of Rules |
|---|---:|
| Rules Generated | 2,479 |
| Rules with Lift > 1 | 1,669 |
| After Support / Confidence / Lift Filtering | 451 |
| After Imaging Filter | 61 |
| After Same-Feature Filter | 61 |
| Final Meaningful Rules | **61** |

The final association workflow therefore uses the filtered set of
61 meaningful rules rather than selecting a single machine-learning
model.

---

# 3. Deep Learning

## 3.1 ANN / MLP — Hospital Length-of-Stay

### Baseline Model

| Metric | Baseline |
|---|---:|
| MAE | 0.4818 |
| RMSE | 0.6796 |
| R² | 0.9158 |

### Hyperparameter Tuning

Seven ANN configurations were evaluated by varying architecture,
learning rate, batch size and dropout.

The recorded tuning experiments included:

- Smaller batch configuration
- Dropout 0.1
- Larger network
- Lower learning rate
- Dropout 0.2
- Larger network with lower learning rate
- Baseline configuration

The lowest recorded test MAE in the tuning table was **0.4057** for
the Smaller_Batch configuration.

### Final Saved ANN Model

The project also contains a finalized tuned ANN artifact with the
following recorded evaluation:

| Metric | Final Tuned ANN |
|---|---:|
| MAE | 0.4424 |
| RMSE | 0.6371 |
| R² | 0.9260 |

The 0.4057 MAE corresponds to the best recorded hyperparameter
experiment in the tuning table, whereas 0.4424 corresponds to the
separately saved final ANN evaluation artifact. These results are
therefore reported separately and are not treated as the same model
evaluation.

The baseline and final tuned metrics are reported separately because
they come from distinct saved evaluation artifacts.

The tuned ANN was retained as the final ANN implementation for the
deep-learning length-of-stay workflow.

---

## 3.2 CNN — Chest X-ray Classification

### Final Model

**Chest_XRay_CNN**

Framework: TensorFlow / Keras

### Final Evaluation

| Metric | Result |
|---|---:|
| Accuracy | 99.17% |
| Precision | 99.17% |
| Recall | 99.17% |
| F1-score | 99.17% |

The saved CNN model is used for the chest X-ray workflow.

The project explicitly presents this CNN as a portfolio/educational
demonstration using the project's synthetic chest X-ray data. It is
not a clinically validated diagnostic model.

No architecture-level winner is claimed because the recorded project
evidence provides the final CNN evaluation rather than a complete
architecture comparison benchmark.

---

## 3.3 RNN / LSTM — Patient Deterioration

Two sequential-model approaches were evaluated.

### RNN

| Metric | Result |
|---|---:|
| Accuracy | 84.53% |
| Precision | 32.70% |
| Recall | 61.50% |
| F1-score | 42.70% |
| ROC-AUC | 81.99% |

### LSTM

| Metric | Result |
|---|---:|
| Accuracy | 84.75% |
| Precision | 31.72% |
| Recall | 54.47% |
| F1-score | 40.09% |
| ROC-AUC | 79.36% |

### Selection

The finalized project artifact identifies the RNN as the primary
sequential deterioration model.

The comparison shows that the LSTM achieved slightly higher accuracy,
while the RNN achieved higher precision, recall, F1-score and ROC-AUC.

Therefore, the documentation presents both results and does not reduce
the comparison to accuracy alone.

The deterioration workflow remains a demonstration model and requires
further validation before any real clinical use.

---

# 4. Medical NLP

## 4.1 Medical Named Entity Recognition

Two pretrained biomedical language-model approaches were evaluated.

### BioBERT

| Metric | Result |
|---|---:|
| Token Accuracy | 78.16% |
| Entity Precision | 41.01% |
| Entity Recall | 46.02% |
| Entity F1 | 43.37% |

### ClinicalBERT

| Metric | Result |
|---|---:|
| Token Accuracy | 77.14% |
| Entity Precision | 36.87% |
| Entity Recall | 44.08% |
| Entity F1 | 40.16% |

### Model Selection

**BioBERT (`dmis-lab/biobert-v1.1`)** was retained as the final
medical NER model.

BioBERT achieved higher token accuracy, entity precision, entity
recall and entity F1-score than the evaluated ClinicalBERT model.

Entity-level F1 was particularly useful for assessing the quality of
medical entity extraction.

---

## 4.2 Sentiment Analysis

### Recorded Evaluation

**TF-IDF + Logistic Regression**

| Metric | Result |
|---|---:|
| Test Accuracy | 50.00% |
| Test Precision | 50.00% |
| Test Recall | 100.00% |
| Test F1 | 66.67% |

The project also contains a DistilBERT-based sentiment workflow.

However, the current recorded evaluation artifacts do not provide a
directly comparable DistilBERT metric record alongside the TF-IDF
results.

Therefore, no performance ranking between TF-IDF Logistic Regression
and DistilBERT is claimed in this document.

---

# 5. Medical RAG Evaluation

The medical RAG system is evaluated through retrieval quality,
source grounding and safety behaviour rather than by assigning a
traditional classification accuracy to generated medical answers.

### Retrieval Infrastructure

- 55 stored medical vectors
- 384-dimensional embedding space
- 55 metadata records
- 0 duplicate chunk IDs
- 6 retrieval tests
- 6 successful retrieval queries
- Top-3 topic hit rate: 100%

The vector-store validation confirms that the RAG index is available
and ready for generation. The indexed sources include curated
WHO and MedlinePlus medical information.

### Final RAG Evaluation

| Measure | Result |
|---|---:|
| Top-1 Topic Accuracy | 1.0000 |
| Top-3 Topic Hit Rate | 1.0000 |
| Mean Topic Consistency | 0.8333 |
| In-domain Acceptance | 1.0000 |
| Out-of-domain Rejection | 1.0000 |
| Safety Score | 0.9333 |
| Context Acceptance / Generation / Extraction | 0.8571 |

The RAG pipeline uses retrieval, reranking and an evidence-safety
threshold. When sufficient evidence is unavailable, the system can
abstain rather than producing an unsupported medical-information
response.

These results measure the behaviour of the retrieval and safety
pipeline for the recorded evaluation set and should not be interpreted
as clinical accuracy.

---

# 6. Agentic AI Evaluation

The agentic layer contains a router and executor that direct
natural-language requests to the appropriate healthcare capability.

## 6.1 Agent Router Evaluation

The router evaluation contained:

- 14 test cases
- 14 correct routes
- Routing accuracy: **100%**

The tests covered healthcare capabilities including diabetes
prediction, hospital LOS, clustering, association analysis,
chest X-ray, deterioration prediction, medical NER, sentiment,
medical RAG and safe fallback.

The router also correctly classified unsupported and empty requests
into the safe-fallback pathway.

## 6.2 Final Agentic Integration Evaluation

The final integration test contained 10 scenarios.

| Metric | Result |
|---|---:|
| Total Tests | 10 |
| Correct Routes | 10 |
| Routing Accuracy | **100%** |
| Valid Executions | 9 |
| Execution Success Rate | **90%** |
| Passed Tests | 9 |
| Overall Pass Rate | **90%** |

The final integration evidence shows that all 10 requests were routed
to the expected tool. One chest X-ray scenario was routed correctly
but produced an execution error, resulting in 9 successful executions
out of 10.

The failed scenario therefore represents an execution-level issue
rather than a routing error.

---

# 7. Final Model Selection Summary

| Module | Final Model / Configuration | Selection / Evaluation Basis |
|---|---|---|
| Diabetes Risk | Random Forest | Finalized model artifact and recorded evaluation |
| Primary Diagnosis | Random Forest + SMOTE + Feature Engineering | Final integrated model; earlier XGBoost configuration was selected as an initial candidate |
| Hospital LOS | XGBoost | Lowest test MAE |
| Patient Clustering | K = 2 | Highest Silhouette Score |
| Association Mining | Filtered Apriori Rules | Support, confidence, lift and domain-specific filtering |
| ANN / MLP | Final Tuned ANN | Hyperparameter tuning and final saved evaluation |
| Chest X-ray CNN | CNN | Final saved image-classification evaluation |
| Deterioration | RNN | Finalized sequential model with comparative RNN/LSTM evaluation |
| Medical NER | BioBERT | Higher recorded metrics than ClinicalBERT |
| Sentiment | TF-IDF + Logistic Regression evaluation; DistilBERT workflow | No directly comparable final metric record for both approaches |
| Medical RAG | Retrieval + Reranking + Safety Gate | Retrieval quality, topic consistency and safety evaluation |
| Agentic AI | Router + Executor | Routing accuracy, execution validity and integration testing |

---

# 8. Responsible Interpretation of Results

The reported metrics describe performance on the project's development
and evaluation datasets. They should not be interpreted as evidence of
clinical effectiveness or safety in real-world populations.

Several modules use public or synthetic datasets. Model performance may
therefore differ substantially on real hospital data.

In particular:

- Primary diagnosis classification remains affected by class imbalance.
- RNN/LSTM deterioration performance requires further validation.
- The ANN and regression results depend on the characteristics of the
  development dataset.
- Association rules describe relationships in the analysed dataset and
  do not establish causation.
- The CNN chest X-ray workflow is a portfolio demonstration and is not
  intended for clinical diagnosis.
- RAG performance depends on the coverage and quality of the curated
  medical sources.
- Agentic AI integration testing measures routing and execution
  behaviour for the recorded scenarios, not clinical reliability.

The HealthAI platform is therefore positioned as a portfolio and
educational healthcare-AI demonstration rather than a clinically
validated medical system.