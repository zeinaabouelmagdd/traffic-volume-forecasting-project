import pandas as pd

from cleaning import handle_missing_values, remove_duplicates, cap_outliers
from feature_engineering import (
    create_time_features,
    add_rush_hour_feature,
    add_part_of_day_feature,
    add_season_feature,
    encode_categorical_features,
    drop_unused_columns
)
from train_best_model import split_data, train_baseline_models, tune_random_forest
from evaluate import compare_models

RAW_DATA_PATH = "data/raw/Metro_Interstate_Traffic_Volume.csv"

def main():
    # 1. Load data
    df = pd.read_csv(RAW_DATA_PATH)
    print("Raw data loaded:", df.shape)

    # 2. Clean data
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = cap_outliers(df, ["temp", "rain_1h", "snow_1h", "clouds_all"])
    print("Cleaning completed:", df.shape)

    # 3. Feature engineering
    df = create_time_features(df)
    df = add_rush_hour_feature(df)
    df = add_part_of_day_feature(df)
    df = add_season_feature(df)
    df = encode_categorical_features(df)
    df = drop_unused_columns(df)
    print("Feature engineering completed:", df.shape)

    # Save processed data
    PROCESSED_DATA_PATH = "data/processed/traffic_volume_prepared.csv"
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print("Processed data saved to:", PROCESSED_DATA_PATH)

    # 4. Split
    X_train, X_test, y_train, y_test = split_data(df)
    print("Train-test split completed.")

    print("DEBUG: reached training stage")

    # 5. Train baseline models
    trained_models = train_baseline_models(X_train, y_train)
    print("Baseline models trained.")

    # 6. Evaluate baseline models
    baseline_results = compare_models(trained_models, X_train, y_train, X_test, y_test)
    print("\nBaseline model comparison:")
    print(baseline_results)

    # 7. Tune random forest
    best_rf_model, best_params, best_cv_score = tune_random_forest(X_train, y_train)
    trained_models["Tuned Random Forest Regressor"] = best_rf_model

    print("\nBest tuned parameters:", best_params)
    print("Best CV score:", best_cv_score)

    # 8. Evaluate all models including tuned model
    final_results = compare_models(trained_models, X_train, y_train, X_test, y_test)
    print("\nFinal model comparison:")
    print(final_results)

if __name__ == "__main__":
    main()