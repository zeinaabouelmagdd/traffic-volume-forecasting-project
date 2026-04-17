from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(page_title="Traffic Volume Prediction App", layout="centered")

st.title("Traffic Volume Prediction App")
st.write("Enter traffic, weather, and date/time information to generate a prediction.")

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "tuned_random_forest_model.pkl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "deployment_predictions.csv"

@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(MODEL_PATH)


try:
    model = load_model(MODEL_PATH)
    st.success("Model loaded successfully ✅")
except Exception as e:
    st.error(f"Model failed to load ❌: {e}")
    st.stop()

st.subheader("Input Features")

holiday = st.selectbox(
    "Holiday",
    [
        "None",
        "Columbus Day",
        "Veterans Day",
        "Thanksgiving Day",
        "Christmas Day",
        "New Years Day",
        "Washingtons Birthday",
        "Memorial Day",
        "Independence Day",
        "State Fair",
        "Labor Day",
        "Martin Luther King Jr Day"
    ]
)

temp = st.number_input(
    "Temperature (Kelvin)",
    min_value=200.0,
    max_value=330.0,
    value=290.0,
    step=0.1
)

rain_1h = st.number_input(
    "Rain in last 1 hour (mm)",
    min_value=0.0,
    value=0.0,
    step=0.1
)

snow_1h = st.number_input(
    "Snow in last 1 hour (mm)",
    min_value=0.0,
    value=0.0,
    step=0.1
)

clouds_all = st.slider(
    "Cloud coverage (%)",
    min_value=0,
    max_value=100,
    value=40
)

weather_main = st.selectbox(
    "Weather Main",
    ["Clouds", "Clear", "Rain", "Snow", "Mist", "Haze", "Fog", "Thunderstorm", "Drizzle", "Smoke", "Squall"]
)

weather_description = st.selectbox(
    "Weather Description",
    [
        "sky is clear",
        "few clouds",
        "scattered clouds",
        "broken clouds",
        "overcast clouds",
        "light rain",
        "moderate rain",
        "heavy intensity rain",
        "proximity shower rain",
        "light snow",
        "heavy snow",
        "mist",
        "haze",
        "fog",
        "proximity thunderstorm",
        "thunderstorm with heavy rain",
        "drizzle",
        "smoke",
        "SQUALLS"
    ]
)

date_value = st.date_input("Date")
time_value = st.time_input("Time")


if st.button("Predict Traffic Volume"):
    try:
        dt = pd.to_datetime(f"{date_value} {time_value}")

        # Raw user input
        input_df = pd.DataFrame({
            "holiday": [holiday],
            "temp": [temp],
            "rain_1h": [rain_1h],
            "snow_1h": [snow_1h],
            "clouds_all": [clouds_all],
            "weather_main": [weather_main],
            "weather_description": [weather_description],
            "date_time": [dt]
        })

        # Datetime features
        input_df["year"] = input_df["date_time"].dt.year
        input_df["month"] = input_df["date_time"].dt.month
        input_df["day"] = input_df["date_time"].dt.day
        input_df["hour"] = input_df["date_time"].dt.hour
        input_df["dayofweek"] = input_df["date_time"].dt.dayofweek
        input_df["is_weekend"] = (input_df["dayofweek"] >= 5).astype(int)

        # Keep a clean copy for CSV / display
        clean_output_row = pd.DataFrame({
            "holiday": [holiday],
            "temp": [temp],
            "rain_1h": [rain_1h],
            "snow_1h": [snow_1h],
            "clouds_all": [clouds_all],
            "weather_main": [weather_main],
            "weather_description": [weather_description],
            "year": [dt.year],
            "month": [dt.month],
            "day": [dt.day],
            "hour": [dt.hour],
            "dayofweek": [dt.dayofweek],
            "is_weekend": [1 if dt.dayofweek >= 5 else 0]
        })

        # Model input
        model_input = input_df.drop(columns=["date_time"])

        model_input_encoded = pd.get_dummies(
            model_input,
            columns=["holiday", "weather_main", "weather_description"],
            drop_first=False
        )

        if hasattr(model, "feature_names_in_"):
            expected_cols = list(model.feature_names_in_)
            model_input_encoded = model_input_encoded.reindex(columns=expected_cols, fill_value=0)
        else:
            st.error("This saved model does not expose feature names, so automatic alignment is not available.")
            st.stop()

        prediction = model.predict(model_input_encoded)[0]

        st.subheader("Prediction Result")
        st.success(f"Predicted Traffic Volume: {prediction:.0f} vehicles")

        # Save a clean readable CSV
        clean_output_row["predicted_traffic_volume"] = round(float(prediction), 2)

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

        if OUTPUT_PATH.exists():
            existing = pd.read_csv(OUTPUT_PATH)
            updated = pd.concat([existing, clean_output_row], ignore_index=True)
        else:
            updated = clean_output_row.copy()

        updated.to_csv(OUTPUT_PATH, index=False)

        # Recent predictions table
        st.subheader("Recent Predictions")
        st.dataframe(updated.tail(10), width="stretch")

        # Recent predictions chart
       
        st.subheader("Recent Prediction Trend")

        recent_preds = updated["predicted_traffic_volume"].tail(10).reset_index(drop=True)

        if len(recent_preds) == 1:
            st.info("Only one prediction saved so far.")
            st.metric("Predicted Traffic Volume", f"{recent_preds.iloc[0]:.0f} vehicles")
        else:
            st.write("X-axis: Prediction Number")
            st.write("Y-axis: Predicted Traffic Volume")

            chart_df = pd.DataFrame({
                "Prediction Number": range(1, len(recent_preds) + 1),
                "Predicted Traffic Volume": recent_preds.values
            }).set_index("Prediction Number")

            st.line_chart(chart_df, width="stretch")

        # Optional technical view
        with st.expander("Show processed model input"):
            st.dataframe(model_input_encoded, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed ❌: {e}")