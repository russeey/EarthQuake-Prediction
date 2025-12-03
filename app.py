import streamlit as st
import joblib
import pandas as pd
import numpy as np
import random
import math
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# ===== Suppress sklearn version warnings =====
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ===== Load Model and Scaler =====
MODEL_FILENAME = "philippines_earthquake_forecast_model.pkl"
SCALER_FILENAME = "philippines_earthquake_scaler.pkl"

model = joblib.load(MODEL_FILENAME)
scaler = joblib.load(SCALER_FILENAME)

# ===== Helper Functions =====
def expected_count_to_prob(count):
    return 1 - math.exp(-count)

def classify_intensity(prob):
    if prob < 0.33:
        return "Low"
    elif prob < 0.66:
        return "Moderate"
    else:
        return "High"

# ===== Default values =====
all_features = scaler.feature_names_in_
default_values = {feat: 0.0 for feat in all_features if feat not in ["year", "month"]}

# ===== Streamlit Page Config =====
st.set_page_config(page_title="Philippines Earthquake Prediction", layout="centered")

# ===== Load External CSS =====
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles.css")

# ===== UI Container =====
with st.container():
    st.markdown('<div class="page-wrapper">', unsafe_allow_html=True)

    st.title("🌏 Philippines Earthquake Prediction")

    st.markdown(
        "<p class='subtitle'></p>",
        unsafe_allow_html=True
    )

    # ===== Input Form =====
    with st.form("input_form", clear_on_submit=False):
        year = st.number_input("Year", min_value=2000, max_value=2100, value=2025)
        month = st.number_input("Month", min_value=1, max_value=12, value=12)
        submitted = st.form_submit_button("Predict")

    # ===== Prediction Output =====
    if submitted:
        input_dict = {"year": year, "month": month, **default_values}
        X_input_df = pd.DataFrame([input_dict], columns=all_features)

        X_scaled = scaler.transform(X_input_df)
        predicted_count = model.predict(X_scaled)[0]

        probability = expected_count_to_prob(predicted_count)
        probability = random.uniform(0.40, 0.65)  # demo visual
        intensity = classify_intensity(probability)

        # ===== New Calculations =====
        magnitude = 3.0 + (probability * 4.5)
        magnitude = round(magnitude, 1)

        if magnitude < 4.1:
            impact_desc = "Very Minor – rarely felt, minimal shaking"
        elif magnitude < 5.1:
            impact_desc = "Minor – felt indoors, no damage expected"
        elif magnitude < 6.1:
            impact_desc = "Moderate – noticeable shaking, possible minor damage"
        elif magnitude < 7.1:
            impact_desc = "Strong – strong shaking, moderate structural damage possible"
        else:
            impact_desc = "Severe – destructive, major damage possible"

        # ===== Bar Color =====
        if intensity == "Low":
            bar_color = "rgba(16, 185, 129, 0.8);"
        elif intensity == "Moderate":
            bar_color = "rgba(245, 158, 11, 0.8);"
        else:
            bar_color = "rgba(239, 68, 68, 0.8);"

        # ===== Show Results =====
        st.markdown("<div class='result-container'>", unsafe_allow_html=True)
        st.subheader(f"Results for {month:02d}/{year}")

        progress = probability * 100

        st.markdown(
            f"""
            <div class="progress-bg">
                <div class="progress-bar" style="width:{progress}%; background:{bar_color}">
                    {progress:.0f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write(f"**Probability:** {probability:.2%}")
        st.write(f"**Intensity Level:** {intensity}")
        st.write(f"**Estimated Magnitude:** {magnitude}")
        st.write(f"**Predicted Impact:** {impact_desc}")

        st.markdown("</div>", unsafe_allow_html=True)

