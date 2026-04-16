import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model, X_train, y_train, X_test, y_test) -> dict:
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    return {
        "Train MAE": mean_absolute_error(y_train, y_train_pred),
        "Train RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "Train R2": r2_score(y_train, y_train_pred),
        "Test MAE": mean_absolute_error(y_test, y_test_pred),
        "Test RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "Test R2": r2_score(y_test, y_test_pred)
    }


def compare_models(trained_models, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    results = []

    for name, model in trained_models.items():
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
        metrics["Model"] = name
        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Test R2", ascending=False).reset_index(drop=True)
    return results_df