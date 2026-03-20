# app.py
import streamlit as st
import pandas as pd
from utils.preprocess import clean_data
from utils.predict import load_model, make_prediction

st.title("Hospital Appointment No-Show Prediction App")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV file", type="csv")
if uploaded_file is not None:
    df = clean_data(uploaded_file)
    st.write("Cleaned Data Preview:")
    st.dataframe(df.head())

    # Load model with caching
    with st.spinner("Loading model..."):
        model = load_model()

    st.subheader("Predict No-Show Risk for a Patient")

    # Select patient row index
    patient_idx = st.number_input(
        "Select patient row index",
        min_value=0,
        max_value=len(df)-1,
        value=0
    )
    patient_data = df.iloc[[patient_idx]]  # keep as DataFrame

    if st.button("Predict"):
        prob = make_prediction(model, patient_data)
        st.write(f"Probability of No-Show: {prob:.2f}")

        if prob > 0.7:
            st.warning("⚠️ High risk of No-Show! Recommended actions: Reduce waiting time, send SMS reminder, avoid Fridays, call older patients.")
        else:
            st.success("✅ Low risk of No-Show")
