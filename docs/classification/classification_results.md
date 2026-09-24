# HealthAI Suite - Classification Results

## Objective

Predict the patient's primary diagnosis using demographic,
symptom, and clinical health indicators.

## Dataset

Indian Healthcare Patient Records

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

Primary_Diagnosis

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
the primary evaluation metric was Macro F1.

Additional metrics:

- Accuracy
- Macro Precision
- Macro Recall
- Weighted F1

## Advanced Models

Random Forest + SMOTE + Feature Engineering:

- Accuracy: 43.46%
- Macro F1: 0.2255
- Weighted F1: 0.4181

XGBoost + SMOTE + Feature Engineering:

- Accuracy: 45.43%
- Macro F1: 0.2099
- Weighted F1: 0.4101

Earlier XGBoost + SMOTE:

- Accuracy: 42.81%
- Macro F1: 0.2315
- Weighted F1: 0.4229

## Model Selection

XGBoost + SMOTE was selected as the classification
candidate based on its stronger Macro F1 among the
evaluated configurations.

The model still has limitations, particularly for
minority disease classes. Therefore, it should be
considered a research/educational prediction system
rather than a clinically validated diagnostic system.

## Explainability

Random Forest feature importance and XGBoost SHAP
analysis were performed.

The most influential features included:

- Blood Glucose
- Total Cholesterol
- HbA1c
- BMI
- Age

These results indicate which features the models relied
on most heavily. They do not establish causal relationships.

## Limitations

- Class imbalance remains challenging.
- Minority disease classes have substantially lower
  predictive performance.
- The dataset is not a clinically validated diagnostic
  dataset.
- The models require further validation before any
  real-world clinical use.

## Data Attribution

Indian Healthcare Patient Records,
Kaggle, Arun's Workspace.

Dataset URL:
https://www.kaggle.com/datasets/arunsworkspace/indian-healthcare-patient-records