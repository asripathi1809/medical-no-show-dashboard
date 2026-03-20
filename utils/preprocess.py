# utils/preprocess.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_data(uploaded_file):
    df = pd.read_csv(uploaded_file)

    # Convert dates
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay']).dt.date.astype('datetime64[ns]')
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay']).dt.date.astype('datetime64[ns]')
    df['Age'] = df['Age'].astype('int64')

    # Rename columns
    df = df.rename(columns={
        'Hipertension': 'Hypertension',
        'Handcap': 'Handicap',
        'SMS_received': 'SMSReceived',
        'No-show': 'NoShow'
    })

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Extract day names
    df['ScheduledDayOfTheWeek'] = df['ScheduledDay'].dt.day_name()
    df['AppointmentDayOfTheWeek'] = df['AppointmentDay'].dt.day_name()

    # Waiting time in days
    df['WaitingTimeDays'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days

    # Label encode binary columns
    label_cols = ['SMSReceived', 'NoShow', 'Hypertension', 'Diabetes', 'Alcoholism', 'Handicap']
    for col in label_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    # One-hot encode categorical
    df = pd.get_dummies(df, columns=['Gender', 'ScheduledDayOfTheWeek', 'AppointmentDayOfTheWeek', 'Scholarship'], drop_first=True)

    # Frequency encode neighborhood
    freq_encoding = df['Neighbourhood'].value_counts().to_dict()
    df['Neighbourhood'] = df['Neighbourhood'].map(freq_encoding)
    df.drop('Neighbourhood', axis=1, inplace=True)

    # Drop irrelevant columns
    for col in ['PatientId', 'AppointmentID', 'ScheduledDay', 'AppointmentDay']:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    return df
