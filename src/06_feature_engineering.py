from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from data_loader import load_data
from data_cleaning import clean_data


# ============================================================
# 1. FEATURE PREPARATION
# ============================================================

def prepare_features(df):

    print("=" * 70)
    print("HealthAI Suite - Feature Engineering")
    print("=" * 70)

    # Candidate features available before diagnosis
    features = [
        "Age",
        "Gender",
        "Region",
        "Socioeconomic_Status",
        "Symptoms",
        "Blood_Glucose_mg_dL",
        "HbA1c_%",
        "Total_Cholesterol_mg_dL",
        "BMI"
    ]

    target = "Primary_Diagnosis"

    # X = input features
    X = df[features].copy()

    # y = target to predict
    y = df[target].copy()

    print("\nInput Feature Shape:")
    print(X.shape)

    print("\nTarget Shape:")
    print(y.shape)

    print("\nSelected Features:")
    print(X.columns.tolist())

    print("\nTarget:")
    print(target)

    return X, y


# ============================================================
# 2. TRAIN-TEST SPLIT
# ============================================================

def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\n" + "=" * 70)
    print("Train-Test Split")
    print("=" * 70)

    print("\nX_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    print("\ny_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    return X_train, X_test, y_train, y_test


# ============================================================
# 3. PREPROCESSING
# ============================================================

def preprocess_features(X_train, X_test):

    # Categorical features
    categorical_features = [
        "Gender",
        "Region",
        "Socioeconomic_Status",
        "Symptoms"
    ]

    # Numerical features
    numerical_features = [
        "Age",
        "Blood_Glucose_mg_dL",
        "HbA1c_%",
        "Total_Cholesterol_mg_dL",
        "BMI"
    ]

    # Apply different preprocessing to different column types
    preprocessor = ColumnTransformer(
        transformers=[

            # One-Hot Encode categorical features
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_features
            ),

            # Standardize numerical features
            (
                "numerical",
                StandardScaler(),
                numerical_features
            )
        ]
    )

    # Learn preprocessing rules ONLY from training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Apply the same learned rules to test data
    X_test_processed = preprocessor.transform(X_test)

    print("\n" + "=" * 70)
    print("Feature Preprocessing")
    print("=" * 70)

    print("\nOriginal Shapes:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)

    print("\nProcessed Shapes:")
    print("X_train_processed:", X_train_processed.shape)
    print("X_test_processed:", X_test_processed.shape)

    print("\nCategorical Features:")
    print(categorical_features)

    print("\nNumerical Features Standardized:")
    print(numerical_features)

    return X_train_processed, X_test_processed, preprocessor


# ============================================================
# 4. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    # Load dataset
    dataframe = load_data()

    # Clean dataset
    dataframe = clean_data(dataframe)

    # Create X and y
    X, y = prepare_features(dataframe)

    # Split BEFORE preprocessing
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Encode categorical features and standardize numerical features
    X_train_processed, X_test_processed, preprocessor = (
        preprocess_features(
            X_train,
            X_test
        )
    )