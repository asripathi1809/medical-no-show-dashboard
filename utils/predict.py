# utils/predict.py
import joblib
import pandas as pd

def load_model(model_path='models/mo_pipeline.pkl'):
    return joblib.load(model_path)

def make_prediction(model, patient_data: pd.DataFrame):
    """
    patient_data: single row DataFrame with same columns as training data
    returns probability of NoShow
    """
    prob = model.predict_proba(patient_data)[:, 1][0]
    return prob
