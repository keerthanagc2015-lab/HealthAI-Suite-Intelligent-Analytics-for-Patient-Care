# ==========================================================
# Import Libraries
# ==========================================================

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


# ==========================================================
# Tune XGBoost
# ==========================================================

def tune_xgboost(
        X_train,
        y_train,
        X_test
):

    print("\n" + "=" * 70)
    print("XGBoost Hyperparameter Tuning")
    print("=" * 70)

    # ------------------------------------------------------
    # Base Model
    # ------------------------------------------------------

    xgb = XGBClassifier(

        objective="multi:softmax",

        num_class=len(set(y_train)),

        random_state=42,

        eval_metric="mlogloss"

    )

    # ------------------------------------------------------
    # Parameter Grid
    # ------------------------------------------------------

    param_grid = {

        "n_estimators": [100, 200],

        "max_depth": [5, 10],

        "learning_rate": [0.05, 0.1],

        "subsample": [0.8, 1.0]

    }

    # ------------------------------------------------------
    # Grid Search
    # ------------------------------------------------------

    grid_search = GridSearchCV(

        estimator=xgb,

        param_grid=param_grid,

        scoring="f1_macro",

        cv=3,

        verbose=2,

        n_jobs=-1

    )

    grid_search.fit(

        X_train,

        y_train

    )

    print("\nBest Parameters")

    print(grid_search.best_params_)

    print("\nBest Cross Validation Score")

    print(grid_search.best_score_)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(

        X_test

    )

    return best_model, y_pred