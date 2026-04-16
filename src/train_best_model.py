import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

RANDOM_SEED = 42
TARGET_COL = "traffic_volume"

def split_data(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)


def train_baseline_models(X_train, y_train):
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_SEED
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=100,
            random_state=RANDOM_SEED
        )
    }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models


def tune_random_forest(X_train, y_train):
    param_grid = {
        "n_estimators": [100, 150],
        "max_depth": [10, None],
        "min_samples_split": [2, 5]
    }

    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_SEED),
        param_grid=param_grid,
        cv=3,
        scoring="r2",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_