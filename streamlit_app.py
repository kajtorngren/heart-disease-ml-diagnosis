import streamlit as st
import pandas as pd
import numpy as np

st.title('🫀 Heart Disease ML Diagnosis 🫀')

st.write('This app builds a machine learning application for heart disease diagnosis.')

st.sidebar.header('User Input Features')

# Upload ECG signals CSV file
uploaded_file = st.sidebar.file_uploader("Upload your ECG signals CSV file", type=["csv"])
if uploaded_file is not None:
    ecg_df = pd.read_csv(uploaded_file, header=None)
    
    # Extract sampling rate from row 8, second column
    sampling_rate_str = ecg_df.iloc[8, 1]  # e.g., "499.348 Hz"
    sampling_rate = float(sampling_rate_str.split()[0])
    
    # Extract ECG signal values starting from row 11 in the first column
    ecg_values = ecg_df[0][14:].astype(float).reset_index(drop=True)
    
    # Create time axis based on the sampling rate
    time_axis = [i / sampling_rate for i in range(len(ecg_values))]
    
    # Combine into a DataFrame for visualization
    ecg_data = pd.DataFrame({'Time (s)': time_axis, 'ECG Signal': ecg_values})
else:
    st.sidebar.warning("Please upload an ECG signals CSV file.")
    ecg_data = pd.DataFrame(columns=['Time (s)', 'ECG Signal'])  # Empty DataFrame

# Collect other user input features
def user_input_features():
    age = st.sidebar.slider('Age', 0, 100, 52)
    sex = st.sidebar.selectbox('Sex', ('male', 'female'))
    chest_pain_type = st.sidebar.selectbox('Chest pain type', ('typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'))
    resting_bp_s = st.sidebar.slider('Resting blood pressure (mm Hg)', 100, 200, 153)
    serum_cholesterol = st.sidebar.slider('Cholesterol (mg/dl)', 172.0, 231.0, 201.0)
    fasting_blood_sugar = st.sidebar.selectbox('Fasting blood sugar', ('> 120 mg/dl', '< 120 mg/dl'))
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

# ECG Data Visualization
with st.expander('ECG Data Visualization'):
    if not ecg_data.empty:
        st.line_chart(ecg_data.rename(columns={'Time (s)': 'index'}).set_index('index'), x='index', y='ECG Signal')
    else:
        st.write("No ECG data available to visualize.")
