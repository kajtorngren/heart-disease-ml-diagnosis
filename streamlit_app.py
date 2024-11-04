import streamlit as st
import pandas as pd
import numpy as np

st.title('🫀 Heart Disease ML Diagnosis 🫀')

st.write('This app build a machine learning application')


st.sidebar.header('User Input Features')


# Upload ECG signals CSV file
uploaded_file = st.sidebar.file_uploader("Upload your ECG signals CSV file", type=["csv"])
if uploaded_file is not None:
    ecg_df = pd.read_csv(uploaded_file)
else:
    st.sidebar.warning("Please upload an ECG signals CSV file.")


# Extract ECG signal starting from row 10 in the "Name" column
ecg_values = ecg_df['Name'][10:].astype(float).reset_index(drop=True)

# Define sampling rate from the metadata (499.01 Hz)
sampling_rate = 499.01
time_axis = [i / sampling_rate for i in range(len(ecg_values))]


# Collect other user input features
def user_input_features():
    age = st.sidebar.slider('Age', 0, 100, 52)
    sex = st.sidebar.selectbox('Sex', ('male', 'female'))
    chest_pain_type = st.sidebar.selectbox('Chest pain type', ('typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'))
    resting_bp_s = st.sidebar.slider('Resting blood pressure (mm Hg)', 100, 200, 153)
    serum_cholesterol = st.sidebar.slider('Cholesterol (mg/dl)', 172.0, 231.0, 201.0)
    fasting_blood_sugar = st.sidebar.selectbox('Fasting blood sugar (mg/dl)', ('> 120 mg/dl', '< 120 mg/dl'))
    max_heart_rate = st.sidebar.slider('Max heart rate', 71, 202, 172)
    
    # Combine inputs into a DataFrame
    data = {
        'age': age,
        'sex': sex,
        'chest_pain_type': chest_pain_type,
        'resting_bp_s': resting_bp_s,
        'serum_cholesterol': serum_cholesterol,
        'fasting_blood_sugar': fasting_blood_sugar,
        'max_heart_rate': max_heart_rate
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input features
input_df = user_input_features()

with st.expander('ECG data visualization'):
    st.line_chart(data=ecg_df, x='time', y='ecg_values', color='Signal')
