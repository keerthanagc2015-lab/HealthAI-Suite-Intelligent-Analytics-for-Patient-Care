# ==========================================================
# Import Libraries
# ==========================================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# ==========================================================
# Tune Random Forest
# ==========================================================

def tune_random_forest(
        X_train,
        y_train,
        X_test
):

    print("\n" + "=" * 70)
    print("Random Forest Hyperparameter Tuning")
    print("=" * 70)

    # ------------------------------------------------------
    # Base Model
    # ------------------------------------------------------

    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )

    # ------------------------------------------------------
    # Parameter Grid
    # ------------------------------------------------------

    param_grid = {

        "n_estimators": [100, 200],

        "max_depth": [10, 20],

        "min_samples_split": [2, 5],

        "min_samples_leaf": [1, 2]

    }

    # ------------------------------------------------------
    # Grid Search
    # ------------------------------------------------------

    grid_search = GridSearchCV(

        estimator=rf,

        param_grid=param_grid,

        cv=3,

        scoring="f1_macro",

        n_jobs=-1,

        verbose=2

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