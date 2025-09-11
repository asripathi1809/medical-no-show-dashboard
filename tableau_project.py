# -*- coding: utf-8 -*-
"""Tableau Project.ipynb

#What this code does: This Python script cleans and preprocesses a hospital appointment dataset, handles missing values, corrects data types, and engineers new features such as scheduled and appointment hour, weekday, and month. These processed features were then used to build an interactive Tableau dashboard.
#Why I chose Python: I chose Python for data cleaning and feature engineering due to its readability, efficiency, and the fact that I'm very comfortable with the programming language, Additionally, it integrates well with Tableau, which helps when preparing datasets for visualization.
#How the Tableau Dashboard helps: The dashboard provides the audience visual insights into the real-world problem of hospital appointment no shows in Brazil. It allows them to explore factors relating to no-shows such as a patient's gender, age, neighborhood, and appointment timing and hhow they relate to patients not showing up to their appointments. This dashboard aims to help healthcare staff identify patterns adds a visual sight to the real world problem of hospital no-shows, collecting insights on people who miss their appointments, as well as their age, demographics, and other factors indicating to why they may have missed their appointments. This can help healthcare staff identify patterns and make data-driven decisions on how to reduce no-shows.

import pandas as pd
import numpy as np
from google.colab import files
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import xgboost as xgb
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from imblearn.over_sampling import SMOTE
#I manually uploaded my file onto google colab so I could do the data analysis
uploaded = files.upload()
df = pd.read_csv('KaggleV2-May-2016.csv')
print('The shape of this dataset is {}'.format(df.shape)) #This gets us the rows and columns that are in the dataset

#Here, we are going to start cleaning the data that we will then convert into another csv to put into Tableau and make a dashboard from it!
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay']).dt.date.astype('datetime64[ns]') #changes date to YYYY-MM-DD
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay']).dt.date.astype('datetime64[ns]') #changes date to YYYY-MM-DD
df['Age'] = df['Age'].astype('int64') #converts it to a whole number integer

#I renamed the columns because I wanted it to be more organized for the analysis
df = df.rename(columns={'Hipertension': 'Hypertension', 'Handcap': 'Handicap', 'SMS_received': 'SMSReceived', 'No-show': 'NoShow'})

print(df.isnull().sum())
df.drop_duplicates(inplace=True) #drops any duplicates in the dataset

print(df.columns) #prints out the different columns in the dataset
df.info() #prints out the information about the columns (such as their data types)
df.head() #prints out the first 5 rows of the dataset

#extract day names instead of scheduled and appointment days
df['ScheduledDayOfTheWeek'] = df['ScheduledDay'].dt.day_name()
df['AppointmentDayOfTheWeek'] = df['AppointmentDay'].dt.day_name()
#calculate waiting time in days
df['WaitingTimeDays'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days

label_cols = ['SMSReceived', 'NoShow', 'Hypertension', 'Diabetes', 'Alcoholism', 'Handicap']
for col in label_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

df = pd.get_dummies(df, columns=['Gender', 'ScheduledDayOfTheWeek', 'AppointmentDayOfTheWeek', 'Scholarship'], drop_first=True)

freq_encoding = df['Neighbourhood'].value_counts().to_dict()
df['Neighbourhood'] = df['Neighbourhood'].map(freq_encoding)
df.drop('Neighbourhood', axis=1, inplace=True)

#I label encoded and one hot encoded the categorical variables in order to prepare them for machine learning models that need numerical input to train and test properly

df.drop(columns=['PatientId', 'AppointmentID', 'ScheduledDay', 'AppointmentDay'], inplace=True) #dropped this column since it was not relevant

X = df.drop(columns=['NoShow'])
y = df['NoShow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#I wanted to compare 4 different models and see which ones fit the model the best. For this,
#I made classification reports, confusion matrices, and roc auc curves for each model for deeper analysis.
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, solver='liblinear'),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results = []

plt.figure(figsize=(10, 6))
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    clas = (classification_report(y_test, y_pred))
    cm = (confusion_matrix(y_test, y_pred))
    rec = (recall_score(y_test, y_pred))
    prec = (precision_score(y_test, y_pred))
    f1 = (f1_score(y_test, y_pred))

    #roc curve
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    print(f"\nModel: {name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n{clas}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Recall: {rec:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Plot confusion matrix for each model
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Recall': rec,
        'Precision': prec,
        'F1 Score': f1
})

plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('(ROC) Curve for all models')
plt.legend(loc="lower right")
plt.show()

results_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)
print(results_df)

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', data=results_df)
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.show()

df.to_csv('cleaned_mednoshows_data.csv', index=False)
files.download('cleaned_mednoshows_data.csv')

from math import exp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from lime.lime_tabular import LimeTabularExplainer
from google.colab import files
import joblib

# ================================
# Load Data
# ================================
uploaded = files.upload()
df = pd.read_csv('cleaned_mednoshows_data.csv')

# Separate target and features
target_col = 'NoShow'
X = df.drop(target_col, axis=1)
y = df[target_col]

# Identify categorical vs numeric
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()

# Preprocessor: OneHotEncode categoricals, pass-through numerics
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

# Build pipeline with preprocessing + model
rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# ================================
# Train/Test Split & Model Training
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
rf_model.fit(X_train, y_train)

print(f"Test set shape: {X_test.shape}")

# LIME: Local Prescription
# Build explainer using transformed training data

explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns,
    class_names=['Show', 'NoShow'],
    mode='classification'
)

"""Explain and recommend actions for one patient"""
def model_predict_proba(x):
    df = pd.DataFrame(x, columns=X_train.columns)
    return rf_model.predict_proba(df)

def generate_prescription(patient_idx, top_n=3):
    patient_data = X_test.iloc[patient_idx].values
    exp = explainer.explain_instance(patient_data, model_predict_proba, num_features=top_n)

    actions = {
        'WaitingTimeDays': "✔️ Reduce waiting time to < 3 days",
        'SMSReceived': "✔️ Ensure SMS reminder is sent",
        'AppointmentDayOfTheWeek_Friday': "✔️ Avoid scheduling on Fridays",
        'Age': "✔️ For older patients, consider reminder call",
    }

    print(f"\n🔴 PATIENT {patient_idx} NO-SHOW RISK: HIGH\n")
    print("📋 TOP REASONS CONTRIBUTING TO NO-SHOW:")
    for feature, weight in exp.as_list():
        direction = "increases" if weight > 0 else "decreases"
        print(f"- {feature} ({direction} risk, weight: {weight:.3f})")

    print("\n✅ RECOMMENDED ACTIONS:")
    for feature, _ in exp.as_list():
        feat_name = feature.split(' ')[0].split('=')[0].strip()
        if feat_name in actions:
            print(actions[feat_name])
    print("\n" + "="*50 + "\n")

# Predict no-show probabilities on test set
probs = rf_model.predict_proba(X_test)[:, 1]  # probability of NoShow=1

# Find high-risk patients (threshold = 0.7)
high_risk_patients = [i for i, p in enumerate(probs) if p > 0.7]

print(f"Found {len(high_risk_patients)} high-risk patients (p > 0.7).")

# Generate prescriptions only for high-risk patients
for idx in high_risk_patients[:5]:  # limit to first 5 for readability
    generate_prescription(idx)

# ================================
# Save Model
# ================================
joblib.dump(rf_model, 'mo_pipeline.pkl')
print("Model pipeline saved.")
files.download('mo_pipeline.pkl')

import matplotlib.pyplot as plt
import pandas as pd

rf = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

rf.fit(X, y)

# Get feature importances
importances = rf.named_steps['classifier'].feature_importances_
feature_names = X.columns
feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Sort and get top 10
top_features = feature_importance.sort_values(by='Importance', ascending=False).head(10)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 10 Feature Importances')
plt.gca().invert_yaxis()  # Most important feature at the top
plt.show()
