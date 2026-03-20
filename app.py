# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.preprocess import clean_data
from utils.predict import load_model, make_prediction
from lime.lime_tabular import LimeTabularExplainer

st.set_page_config(page_title="Hospital No-Show Dashboard", layout="wide")
st.title("🏥 Hospital Appointment No-Show Prediction App & Dashboard")

# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    df = clean_data(uploaded_file)
    st.success("✅ Data loaded and cleaned successfully!")

    # -----------------------------
    # Dataset Overview
    # -----------------------------
    st.subheader("Dataset Overview")
    st.write(df.describe())
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(x='NoShow', data=df, ax=ax[0])
    ax[0].set_title("No-Show Distribution")
    sns.histplot(df['Age'], bins=20, ax=ax[1])
    ax[1].set_title("Age Distribution")
    st.pyplot(fig)

    # -----------------------------
    # Load Model
    # -----------------------------
    with st.spinner("Loading trained model..."):
        model = load_model()
    st.success("Model loaded successfully!")

    # -----------------------------
    # LIME Explainer
    # -----------------------------
    explainer = LimeTabularExplainer(
        df.values,
        feature_names=df.columns,
        class_names=['Show', 'NoShow'],
        mode='classification'
    )

    # -----------------------------
    # Sidebar for patient selection & filters
    # -----------------------------
    st.sidebar.title("Patient & Filters")
    patient_idx = st.sidebar.number_input(
        "Select patient row index",
        min_value=0,
        max_value=len(df)-1,
        value=0
    )
    age_filter = st.sidebar.slider(
        "Filter by Age",
        int(df['Age'].min()),
        int(df['Age'].max()),
        (0, 100)
    )
    gender_cols = [c for c in df.columns if c.startswith("Gender_")]
    selected_gender = st.sidebar.multiselect(
        "Filter by Gender",
        gender_cols,
        default=gender_cols
    )

    # -----------------------------
    # Wrapper for LIME to avoid column mismatch
    # -----------------------------
    def model_predict_proba(x):
        df_temp = pd.DataFrame(x, columns=df.columns)
        if hasattr(model, "feature_names_in_"):
            for col in model.feature_names_in_:
                if col not in df_temp.columns:
                    df_temp[col] = 0
            df_temp = df_temp[model.feature_names_in_]
        return model.predict_proba(df_temp)

    # -----------------------------
    # Patient Prediction Section
    # -----------------------------
    st.subheader("Predict No-Show for a Patient")
    patient_data = df.iloc[[patient_idx]]
    if st.button("Predict for Selected Patient"):
        prob = make_prediction(model, patient_data)
        st.write(f"**Probability of No-Show:** {prob:.2f}")
        if prob > 0.7:
            st.warning(
                "⚠️ High risk of No-Show! Recommended actions: Reduce waiting time, "
                "send SMS reminder, avoid Fridays, call older patients."
            )
        else:
            st.success("✅ Low risk of No-Show")

        # LIME explanation
        st.subheader("Why this prediction?")
        exp = explainer.explain_instance(
            patient_data.values[0],
            model_predict_proba,
            num_features=5
        )
        exp_df = pd.DataFrame(exp.as_list(), columns=['Feature', 'Contribution'])
        st.dataframe(exp_df)
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.barplot(x='Contribution', y='Feature', data=exp_df, palette='coolwarm', ax=ax)
        ax.set_title("Top 5 Features Contributing to Prediction")
        st.pyplot(fig)

    # -----------------------------
    # Top Risk Patients Dashboard
    # -----------------------------
    st.subheader("📊 Top High-Risk Patients Dashboard")

    # Filter dataset
    filtered_df = df[(df['Age'] >= age_filter[0]) & (df['Age'] <= age_filter[1])]
    if selected_gender:
        filtered_df = filtered_df[filtered_df[selected_gender].sum(axis=1) > 0]

    # Predict probabilities for filtered patients
    filtered_df['NoShow_Prob'] = filtered_df.apply(
        lambda row: make_prediction(model, row.to_frame().T), axis=1
    )

    # Top 10 high-risk
    top_risk = filtered_df.sort_values(by='NoShow_Prob', ascending=False).head(10)
    st.write(top_risk[['Age', 'NoShow_Prob'] + selected_gender])

    # Plot top 10
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='NoShow_Prob', y=top_risk.index, data=top_risk, palette='Reds', ax=ax)
    ax.set_xlabel("Predicted Probability of No-Show")
    ax.set_ylabel("Patient Row Index")
    ax.set_title("Top 10 High-Risk Patients")
    st.pyplot(fig)

    # Download top-risk CSV
    csv = top_risk.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Top Risk Patients CSV",
        data=csv,
        file_name="top_risk_patients.csv",
        mime='text/csv'
    )
