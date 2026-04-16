import pandas as pd

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["date_time"] = pd.to_datetime(df["date_time"])

    df["year"] = df["date_time"].dt.year
    df["month"] = df["date_time"].dt.month
    df["day"] = df["date_time"].dt.day
    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df


def add_rush_hour_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)
    return df


def add_part_of_day_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def get_part_of_day(hour: int) -> str:
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        return "night"

    df["part_of_day"] = df["hour"].apply(get_part_of_day)
    return df


def add_season_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["season"] = df["month"] % 12 // 3 + 1
    season_map = {1: "winter", 2: "spring", 3: "summer", 4: "autumn"}
    df["season"] = df["season"].map(season_map)
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    categorical_cols = [
        "holiday",
        "weather_main",
        "weather_description",
        "part_of_day",
        "season"
    ]

    existing_cols = [col for col in categorical_cols if col in df.columns]
    df = pd.get_dummies(df, columns=existing_cols, drop_first=True)

    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "date_time" in df.columns:
        df = df.drop(columns=["date_time"])

    return df