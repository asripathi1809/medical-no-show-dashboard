# utils/preprocess.py

import pandas as pd

def clean_data(uploaded_file):
    """
    Reads CSV and performs basic cleaning.
    Extend this as needed.
    """
    df = pd.read_csv(uploaded_file)
    
    # Example cleaning: fill NAs
    df.fillna(0, inplace=True)
    
    # Example: remove unnecessary columns if they exist
    drop_cols = [col for col in ["PatientId", "AppointmentID"] if col in df.columns]
    df = df.drop(columns=drop_cols)
    
    return df
