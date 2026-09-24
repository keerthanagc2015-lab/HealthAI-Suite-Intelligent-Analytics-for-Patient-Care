"""
======================================================================
HealthAI Suite - Advanced Model Training
Module : Preprocessing
======================================================================

Author : Keerthana

Description:
Loads the advanced healthcare dataset,
performs feature selection,
train-test split,
preprocessing,
label encoding,
and SMOTE.

Production improvement:
The fitted preprocessing pipeline is returned so that the exact
same preprocessing can be reused during model inference.

======================================================================
"""

# ==========================================================
# Import Libraries
# ==========================================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import SMOTE


# ==========================================================
# Load Dataset
# ==========================================================

def load_dataset(file_path):

    df = pd.read_csv(file_path)

    print("=" * 70)
    print("HealthAI Suite - Advanced Model Training")
    print("=" * 70)

    print("\nDataset Loaded Successfully")
    print(f"Dataset Shape : {df.shape}")

    return df


# ==========================================================
# Feature Selection
# ==========================================================

def select_features(df):

    selected_features = [

        # --------------------------------------------------
        # Original Features
        # --------------------------------------------------

        "Age",
        "Gender",
        "Region",
        "Socioeconomic_Status",
        "Symptoms",
        "Blood_Glucose_mg_dL",
        "HbA1c_%",
        "Total_Cholesterol_mg_dL",
        "BMI",

        # --------------------------------------------------
        # Engineered Features
        # --------------------------------------------------

        "BMI_Category",
        "Age_Group",
        "High_Glucose",
        "High_Cholesterol",
        "HbA1c_Category"

    ]

    target = "Primary_Diagnosis"

    X = df[selected_features]

    y = df[target]

    print("\nSelected Features")
    print("-" * 50)

    for feature in selected_features:
        print(feature)

    print("\nTarget")
    print(target)

    print("\nInput Shape :", X.shape)
    print("Target Shape:", y.shape)

    return X, y


# ==========================================================
# Train Test Split
# ==========================================================

def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )

    print("\n" + "=" * 70)
    print("Train Test Split")
    print("=" * 70)

    print("\nX_train :", X_train.shape)
    print("X_test  :", X_test.shape)

    print("\ny_train :", y_train.shape)
    print("y_test  :", y_test.shape)

    return X_train, X_test, y_train, y_test


# ==========================================================
# Feature Preprocessing
# ==========================================================

def preprocess_features(X_train, X_test):

    # ------------------------------------------------------
    # Categorical Features
    # ------------------------------------------------------

    categorical_features = [

        "Gender",
        "Region",
        "Socioeconomic_Status",
        "Symptoms",

        "BMI_Category",
        "Age_Group",
        "High_Glucose",
        "High_Cholesterol",
        "HbA1c_Category"

    ]

    # ------------------------------------------------------
    # Numerical Features
    # ------------------------------------------------------

    numerical_features = [

        "Age",
        "Blood_Glucose_mg_dL",
        "HbA1c_%",
        "Total_Cholesterol_mg_dL",
        "BMI"

    ]

    # ------------------------------------------------------
    # Column Transformer
    # ------------------------------------------------------

    preprocessor = ColumnTransformer(

        transformers=[

            (

                "num",

                StandardScaler(),

                numerical_features

            ),

            (

                "cat",

                OneHotEncoder(handle_unknown="ignore"),

                categorical_features

            )

        ]

    )

    # ------------------------------------------------------
    # Fit ONLY on Training Data
    # ------------------------------------------------------

    X_train_processed = preprocessor.fit_transform(

        X_train

    )

    # ------------------------------------------------------
    # Transform Test Data Using Same Fitted Preprocessor
    # ------------------------------------------------------

    X_test_processed = preprocessor.transform(

        X_test

    )

    print("\n" + "=" * 70)
    print("Feature Preprocessing")
    print("=" * 70)

    print("\nCategorical Features")

    print(categorical_features)

    print("\nNumerical Features")

    print(numerical_features)

    print("\nProcessed Shapes")

    print("X_train :", X_train_processed.shape)

    print("X_test  :", X_test_processed.shape)

    # ------------------------------------------------------
    # Return fitted preprocessor
    #
    # This is important for production inference because
    # the same fitted scaler and encoder must be reused.
    # ------------------------------------------------------

    return (

        X_train_processed,

        X_test_processed,

        preprocessor

    )


# ==========================================================
# Encode Target
# ==========================================================

def encode_target(y_train, y_test):

    label_encoder = LabelEncoder()

    y_train_encoded = label_encoder.fit_transform(

        y_train

    )

    y_test_encoded = label_encoder.transform(

        y_test

    )

    print("\n" + "=" * 70)
    print("Target Encoding")
    print("=" * 70)

    print("\nEncoded Classes\n")

    for index, disease in enumerate(

        label_encoder.classes_

    ):

        print(

            f"{index} --> {disease}"

        )

    return (

        y_train_encoded,

        y_test_encoded,

        label_encoder

    )


# ==========================================================
# Apply SMOTE
# ==========================================================

def apply_smote(X_train, y_train):

    print("\n" + "=" * 70)
    print("Applying SMOTE")
    print("=" * 70)

    print("\nClass Distribution Before SMOTE")

    print(

        pd.Series(y_train).value_counts()

    )

    smote = SMOTE(

        random_state=42

    )

    X_train_smote, y_train_smote = smote.fit_resample(

        X_train,

        y_train

    )

    print("\nClass Distribution After SMOTE")

    print(

        pd.Series(y_train_smote).value_counts()

    )

    print("\nTraining Shape Before SMOTE")

    print(

        X_train.shape

    )

    print("\nTraining Shape After SMOTE")

    print(

        X_train_smote.shape

    )

    return (

        X_train_smote,

        y_train_smote

    )


# ==========================================================
# Prepare Data
# ==========================================================

def prepare_data(file_path):

    # ------------------------------------------------------
    # Load Dataset
    # ------------------------------------------------------

    df = load_dataset(

        file_path

    )

    # ------------------------------------------------------
    # Feature Selection
    # ------------------------------------------------------

    X, y = select_features(

        df

    )

    # ------------------------------------------------------
    # Train-Test Split
    # ------------------------------------------------------

    (

        X_train,

        X_test,

        y_train,

        y_test

    ) = split_data(

        X,

        y

    )

    # ------------------------------------------------------
    # Feature Preprocessing
    #
    # IMPORTANT:
    # The fitted preprocessor is captured here so it can
    # later be saved and reused during inference.
    # ------------------------------------------------------

    (

        X_train_processed,

        X_test_processed,

        preprocessor

    ) = preprocess_features(

        X_train,

        X_test

    )

    # ------------------------------------------------------
    # Target Encoding
    # ------------------------------------------------------

    (

        y_train_encoded,

        y_test_encoded,

        label_encoder

    ) = encode_target(

        y_train,

        y_test

    )

    # ------------------------------------------------------
    # Apply SMOTE
    #
    # SMOTE is applied ONLY to training data.
    # Test data remains untouched.
    # ------------------------------------------------------

    (

        X_train_smote,

        y_train_smote

    ) = apply_smote(

        X_train_processed,

        y_train_encoded

    )

    # ------------------------------------------------------
    # Return Prepared Data
    #
    # preprocessor is intentionally returned so the exact
    # fitted preprocessing pipeline can be persisted for
    # production inference.
    # ------------------------------------------------------

    return (

        X_train_smote,

        X_test_processed,

        y_train_smote,

        y_test_encoded,

        label_encoder,

        preprocessor

    )


# ==========================================================
# End of Module
# ==========================================================