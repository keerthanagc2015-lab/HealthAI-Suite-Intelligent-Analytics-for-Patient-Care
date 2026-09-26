# HealthAI Suite - Classification Results

## Objective

Predict the patient's primary diagnosis using demographic,
symptom, and clinical health indicators.

## Dataset

**Indian Healthcare Patient Records**

Source:

Kaggle - https://www.kaggle.com/datasets/arunsworkspace/indian-healthcare-patient-records

Dataset size:

100,000 records

License:

MIT

## Selected Features

- Age
- Gender
- Region
- Socioeconomic_Status
- Symptoms
- Blood_Glucose_mg_dL
- HbA1c_%
- Total_Cholesterol_mg_dL
- BMI
- BMI_Category
- Age_Group
- High_Glucose
- High_Cholesterol
- HbA1c_Category

## Target

`Primary_Diagnosis`

## Preprocessing

- Train-test split: 80:20
- Numerical feature standardization
- Categorical feature one-hot encoding
- Target label encoding
- SMOTE for class balancing

## Models Evaluated

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- KNN
- Gaussian Naive Bayes
- LinearSVC

## Evaluation Metrics

Because the dataset is imbalanced and multiclass,
the primary evaluation metric was **Macro F1**.

Additional metrics:

- Accuracy
- Macro Precision
- Macro Recall
- Weighted F1

## Advanced Model Evaluation

### Random Forest + SMOTE + Feature Engineering

- Accuracy: 43.46%
- Macro Precision: 0.2283
- Macro Recall: 0.2278
- Macro F1: 0.2255
- Weighted F1: 0.4181

### XGBoost + SMOTE + Feature Engineering

- Accuracy: 45.43%
- Macro F1: 0.2099
- Weighted F1: 0.4101

### Earlier XGBoost + SMOTE Configuration

- Accuracy: 42.81%
- Macro F1: 0.2315
- Weighted F1: 0.4229

## Model Selection

Multiple model configurations were evaluated for the
multiclass primary-diagnosis task.

Because the target is imbalanced and multiclass, Macro F1
was treated as the primary evaluation metric rather than
relying on accuracy alone.

### Initial Candidate Selection

During the earlier advanced model-comparison stage, the
XGBoost + SMOTE configuration achieved a Macro F1 of
**0.2315**, which was the strongest recorded Macro F1 among
the evaluated configurations at that stage.

Therefore, **XGBoost + SMOTE was selected as the initial
classification candidate**.

### Final Integrated Model

During the subsequent integration stage, the finalized
primary-diagnosis workflow was implemented using:

**Random Forest + SMOTE + Feature Engineering**

The saved application model and its metadata identify
Random Forest as the model used by the final
primary-diagnosis workflow.

Final integrated-model evaluation:

- Accuracy: 43.46%
- Macro Precision: 0.2283
- Macro Recall: 0.2278
- Macro F1: 0.2255
- Weighted F1: 0.4181

This documentation distinguishes the earlier candidate
selection experiment from the later integrated model so that
the two development stages are not presented as the same
model-selection event.

## Explainability

Random Forest feature importance and XGBoost SHAP analysis
were performed.

The most influential features included:

- Blood Glucose
- Total Cholesterol
- HbA1c
- BMI
- Age

These results indicate which features the models relied on
most heavily. They do not establish causal relationships.

## Limitations

- Class imbalance remains challenging.
- Minority disease classes have substantially lower
  predictive performance.
- The dataset is not a clinically validated diagnostic
  dataset.
- The recorded model performance may not generalize to
  real-world clinical populations.
- The models require further validation before any
  real-world clinical use.
- The primary-diagnosis module should be considered a
  research and educational prediction system rather than
  a clinically validated diagnostic system.

## Data Attribution

**Indian Healthcare Patient Records**

Kaggle - Arun's Workspace

Dataset URL:

https://www.kaggle.com/datasets/arunsworkspace/indian-healthcare-patient-records