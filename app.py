# app.py
import streamlit as st
import pandas as pd
import sys
import os

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))

from preprocess import clean_data
from predict import load_model, make_prediction

st.title("Hospital Appointment No-Show Prediction App")

# Load & clean data
uploaded_file = st.file_uploader("Upload CSV file", type="csv")
if uploaded_file:
    df = clean_data(uploaded_file)
    st.write("Cleaned Data Preview:")
    st.dataframe(df.head())

    # Load trained model
    model = load_model()

    st.subheader("Predict No-Show Risk for a Patient")
    patient_idx = st.number_input("Select patient row index", min_value=0, max_value=len(df)-1, value=0)
    patient_data = df.iloc[[patient_idx]]  # keep as DataFrame
    
    prob = make_prediction(model, patient_data)
    st.write(f"Probability of No-Show: {prob:.2f}")

    if prob > 0.7:
        st.warning("⚠️ High risk of No-Show! Recommended actions: Reduce waiting time, send SMS reminder, avoid Fridays, call older patients.")
    else:
        st.success("✅ Low risk of No-Show")
