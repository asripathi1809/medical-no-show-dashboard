import joblib
import requests
import os
import streamlit as st

MODEL_URL = "https://drive.google.com/uc?id=1P1Vr53XafaGaZVqxnXCqNvZRiZo_qrGU"

@st.cache_resource
def load_model():
    """Download and cache the trained pipeline"""
    if not os.path.exists("mo_pipeline.pkl"):
        response = requests.get(MODEL_URL)
        with open("mo_pipeline.pkl", "wb") as f:
            f.write(response.content)
    return joblib.load("mo_pipeline.pkl")

def make_prediction(model, data):
    """Ensure features match the model and return probability of NoShow"""
    expected_features = model.feature_names_in_

    # Add missing columns
    for col in expected_features:
        if col not in data.columns:
            data[col] = 0

    # Reorder columns
    data = data[expected_features]

    # Predict probability
    prob = model.predict_proba(data)[:,1][0]
    return prob
