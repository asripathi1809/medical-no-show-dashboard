import joblib
import requests
import os
import streamlit as st

MODEL_URL = "https://drive.google.com/uc?id=1P1Vr53XafaGaZVqxnXCqNvZRiZo_qrGU"

@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl"):
        response = requests.get(MODEL_URL)
        with open("model.pkl", "wb") as f:
            f.write(response.content)
    return joblib.load("model.pkl")
