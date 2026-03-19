# utils/preprocess.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_data(file_path):
    df = pd.read_csv(file_path)
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay']).dt.date.astype('datetime64[ns]')
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay']).dt.date.astype('datetime64[ns]')
    df['Age'] = df['Age'].astype('int64')
    df = df.rename(columns={'Hipertension': 'Hypertension', 'Handcap': 'Handicap',
                            'SMS_received': 'SMSReceived', 'No-show': 'NoShow'})
    df.drop_duplicates(inplace=True)
    
    # Feature engineering
    df['ScheduledDayOfTheWeek'] = df['ScheduledDay'].dt.day_name()
    df['AppointmentDayOfTheWeek'] = df['AppointmentDay'].dt.day_name()
    df['WaitingTimeDays'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
    
    # Encode categorical
    label_cols = ['SMSReceived', 'NoShow', 'Hypertension', 'Diabetes', 'Alcoholism', 'Handicap']
    for col in label_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    
    df = pd.get_dummies(df, columns=['Gender', 'ScheduledDayOfTheWeek',
                                     'AppointmentDayOfTheWeek', 'Scholarship'], drop_first=True)
    
    freq_encoding = df['Neighbourhood'].value_counts().to_dict()
    df['Neighbourhood'] = df['Neighbourhood'].map(freq_encoding)
    df.drop('Neighbourhood', axis=1, inplace=True)
    
    df.drop(columns=['PatientId', 'AppointmentID', 'ScheduledDay', 'AppointmentDay'], inplace=True)
    
    return df
