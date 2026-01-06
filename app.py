import streamlit as st
import numpy as np
import joblib
import os

# ------------------------------
# App Config
# ------------------------------
st.set_page_config(page_title="Customer Transaction Prediction")
st.title("Customer Transaction Prediction")

# ------------------------------
# Load Model & Scaler
# ------------------------------
if not os.path.exists("mlp_classifier_model.pkl"):
    st.error("❌ Model file not found")
    st.stop()

if not os.path.exists("scaler.pkl"):
    st.error("❌ Scaler file not found")
    st.stop()

model = joblib.load("mlp_classifier_model.pkl")
scaler = joblib.load("scaler.pkl")

# ------------------------------
# FORCE FEATURE COUNT = 200
# ------------------------------
EXPECTED_FEATURES = 200
st.info("Model expects 200 input features")

# ------------------------------
# Input Section
# ------------------------------
st.subheader("Enter Feature Values")

input_values = []
for i in range(EXPECTED_FEATURES):
    val = st.number_input(
        f"Feature {i+1}",
        value=0.0,
        step=0.01
    )
    input_values.append(val)

# Convert to numpy array
input_array = np.array(input_values).reshape(1, EXPECTED_FEATURES)

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict"):
    try:
        scaled_input = scaler.transform(input_array)
        prediction = model.predict(scaled_input)

        st.success(f"Prediction Result: {prediction[0]}")

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(scaled_input)
            st.write(f"Prediction Confidence: {np.max(prob):.2f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
