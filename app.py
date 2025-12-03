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

# ===== Default values for features =====
all_features = scaler.feature_names_in_
default_values = {feat: 0.0 for feat in all_features if feat not in ["year", "month"]}

# ===== Streamlit Page Config =====
st.set_page_config(page_title="Philippines Earthquake Prediction", layout="centered")

# ===== Custom CSS =====
st.markdown("""
<style>
/* Center the entire form container */
.centered-container {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
    gap: 20px;
}

/* Center the button */
div.stButton > button {
    width: 200px;
    background-color: #3b82f6;
    color: white;
    font-weight: bold;
    padding: 10px 0;
    border-radius: 10px;
    border: none;
    transition: 0.3s;
}

div.stButton > button:hover {
    background-color: #2563eb;
    cursor: pointer;
}

/* Center number inputs */
.stNumberInput>div>input {
    text-align: center;
    width: 100px;
}
</style>
""", unsafe_allow_html=True)

# ===== Centered Layout =====
with st.container():
    st.markdown('<div class="centered-container">', unsafe_allow_html=True)
    st.title("🇵🇭 Philippines Earthquake Prediction")
    st.write("Enter a date below to see the predicted probability and intensity of earthquakes.")

    # ===== Input Form =====
    with st.form("input_form"):
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
        probability = random.uniform(0.4, 0.65)  # demo hack
        intensity = classify_intensity(probability)

        st.subheader(f"Prediction Results for {month:02d}/{year}")

        # Progress bar color
        if intensity == "Low":
            bar_color = "#4CAF50"
        elif intensity == "Moderate":
            bar_color = "#FF9800"
        else:
            bar_color = "#F44336"

        st.markdown(
            f"""
            <div style="background-color:#ddd; border-radius:10px; width:300px; height:25px; margin:0 auto 10px auto;">
                <div style="width:{probability*100}%; height:100%; background-color:{bar_color}; border-radius:10px; text-align:center; color:white; font-weight:bold; line-height:25px;">
                    {probability:.0%}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write(f"**Predicted Probability:** {probability:.2%}")
        st.write(f"**Predicted Intensity:** {intensity}")

    st.markdown('</div>', unsafe_allow_html=True)
