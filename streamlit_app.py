import streamlit as st
import pandas as pd
import numpy as np

st.title('🫀 Heart Disease Machine Learning Diagnosis 🫀')

st.write('This app build a machine learning application')


st.sidebar.header('User Input Features')


# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        age = st.sidebar.slider('Age', 0, 52, 100)
        sex = st.sidebar.selectbox('Sex',('male','female'))
        chest_pain_type = st.sidebar.selectbox('Chest pain type',('typical angina','atypical angina', 'non-anginal pain', 'asymptomatic'))
        resting_bp_s = st.sidebar.slider('Resting blood pressure (mm Hg)', 100, 153, 200)
        serum_cholesterol = st.sidebar.slider('Cholesterol (mg/dl)', 172.0, 201.0 ,231.0)
        fasting_blood_sugar = st.sidebar.selectbox('Fasting blood sugar (mg/dl)',('> 120 mg/dl','< 120 mg/dl'))
        max_heart_rate = st.sidebar.slider('Max heart rate', 71, 182, 202)
        
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()
