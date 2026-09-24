# ==========================================================
# Import Libraries
# ==========================================================

import sys
import os

# Add Module 10 to Python Path
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "10_advanced_model_training"
        )
    )
)

from preprocessing import prepare_data
from evaluation import evaluate_model, compare_models

from random_forest_tuning import tune_random_forest
from xgboost_tuning import tune_xgboost


# ==========================================================
# Main Function
# ==========================================================

def main():

    file_path = "data/processed/advanced_healthcare.csv"

    (
        X_train,
        X_test,
        y_train,
        y_test,
        label_encoder

    ) = prepare_data(file_path)

    results = []

    # ======================================================
    # Random Forest Hyperparameter Tuning
    # ======================================================

    rf_model, rf_predictions = tune_random_forest(

        X_train,
        y_train,
        X_test

    )

    rf_results = evaluate_model(

        "Random Forest - Hyperparameter Tuned",

        y_test,

        rf_predictions

    )

    results.append(rf_results)

    # ======================================================
    # XGBoost Hyperparameter Tuning
    # ======================================================

    xgb_model, xgb_predictions = tune_xgboost(

        X_train,
        y_train,
        X_test

    )

    xgb_results = evaluate_model(

        "XGBoost - Hyperparameter Tuned",

        y_test,

        xgb_predictions

    )

    results.append(xgb_results)

    # ======================================================
    # Final Comparison
    # ======================================================

    compare_models(results)


# ==========================================================
# Driver Code
# ==========================================================

if __name__ == "__main__":

    main()