# URL: https://heart-disease-ml-diagnosis.streamlit.app/
# Streamlit APIs: https://docs.streamlit.io/develop/api-reference

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

st.legacy_caching.clear_cache()  # Clears the Streamlit cache

# Load the saved model
model = joblib.load('trained_model.pkl')

st.set_page_config(
        page_title="Heart Failure Detection",
        page_icon="❤",
        layout="wide"
    )

# Add the banner image
st.image("cardiology.jpg", use_column_width=True)

st.title('🧠 Machine Learning For Diagnosing & Monitoring Heart Disease 🫀')
st.info('This app builds a machine learning application for heart disease diagnosis. A prediction is made based on the ECG signal and input features.')

st.sidebar.header('📝 User Input Features')

# Initialize ecg_df
ecg_df = pd.DataFrame()

# Upload ECG signals CSV file
uploaded_file = st.sidebar.file_uploader(
    "Upload your ECG signal CSV file.  \nDesigned for Samsung Health Monitor App with Samsung Galaxy Watch 3.",
    type=["csv"]
)

if uploaded_file is not None:
    ecg_df = pd.read_csv(uploaded_file, header=None)
    
    # Extract sampling rate from row 8, second column
    sampling_rate_str = ecg_df.iloc[8, 1]  # e.g., "499.348 Hz"
    sampling_rate = float(sampling_rate_str.split()[0])
    
    # Extract ECG signal values starting from row 15 (index 14) in the first column
    ecg_values = ecg_df[0][14:].astype(float).reset_index(drop=True)
    
    # Create time axis based on the sampling rate
    time_axis = [i / sampling_rate for i in range(len(ecg_values))]
    
    # Combine into a DataFrame for visualization
    ecg_data = pd.DataFrame({'Time (s)': time_axis, 'ECG Signal (mV)': ecg_values})
    
    # Set time axis limit based on the last time value
    time_limit = time_axis[-1]
else:
    st.sidebar.warning("Please upload an ECG signal CSV file.")
    ecg_data = pd.DataFrame(columns=['Time (s)', 'ECG Signal (mV)'])  # Empty DataFrame
    time_limit = None

# Collect other user input features
def user_input_features():
    age = st.sidebar.slider('Age', 18, 100, 50)
    sex = st.sidebar.radio('Sex', ('male', 'female'))
    chest_pain_type = st.sidebar.selectbox('Chest pain type', ('typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'))
    exercise_induced_angina = st.sidebar.selectbox('Chest pain from exercise', ('Yes', 'No'))
    resting_bp_s = st.sidebar.slider('Resting blood pressure (mm Hg)', 90, 200, 120)
    cholesterol = st.sidebar.number_input('Cholesterol (mg/dl)', value=None, placeholder='Type a number...')
    max_heart_rate = st.sidebar.slider('Max heart rate (bps)', 70, 220, 150)
    
    chest_pain_type_mapping = {
        'typical angina': 1,
        'atypical angina': 2,
        'non-anginal pain': 3,
        'asymptomatic': 4
    }

    chest_pain_type_encoded = chest_pain_type_mapping[chest_pain_type]
    
    # Combine inputs into a DataFrame
    data = {
        'age': age,
        'sex': 1 if sex == 'male' else 0,
        'chest_pain_type': chest_pain_type_encoded,
        'exercise_induced_angina': 1 if exercise_induced_angina == 'Yes' else 0,
        'resting_bp_s': resting_bp_s,
        'cholesterol': cholesterol,
        'max_heart_rate': max_heart_rate
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input features
input_df = user_input_features()

st.write("Input DataFrame:")
st.write(input_df)

# Define a mapping between input column names and model's feature names
column_mapping = {
    'chest_pain_type': 'cp',                 # Map chest pain type
    'exercise_induced_angina': 'exang',     # Map exercise-induced angina
    'cholesterol': 'chol',                  # Map cholesterol
    'resting_bp_s': 'trestbps',             # Map resting blood pressure
    'max_heart_rate': 'thalach'             # Map max heart rate
}

# Rename columns in input_df to match the model's feature names
input_df.rename(columns=column_mapping, inplace=True)


# Encoding categorical variables in the same way as during training
input_df = pd.get_dummies(input_df, drop_first=True)

# Check for missing columns
#missing_cols = set(model.feature_names_in_) - set(input_df.columns)
#if missing_cols:
#    st.warning(f"Missing columns detected in input data: {missing_cols}")
#    for col in missing_cols:
#        input_df[col] = 0  # Add missing columns with default value 0
#else:
#    st.success("No missing columns detected. Input data is aligned with the model.")

# Check for extra columns
#extra_cols = set(input_df.columns) - set(model.feature_names_in_)
#if extra_cols:
#    st.warning(f"Extra columns detected in input data: {extra_cols}")
#else:
#    st.success("No extra columns detected. Input data matches the model's expected structure.")

# Ensure column order matches the model's expected feature order
input_df = input_df[model.feature_names_in_]



# Make predictions
if st.button('Predict'):

    # Make predictions
    #prediction = model.predict(input_df)
    #prediction_proba = model.predict_proba(input_df)

    #st.success(f"The predicted class is: {prediction[0]}")
    #st.success(f"With a probability of {prediction_proba[0][1]*100:.1f}%.")

    #------------------

    # Make predictions
    prediction = model.predict(input_df)  # Invert the predictions if they are inverted
    prediction_proba = model.predict_proba(input_df)

    # Display the prediction and probability
    if prediction[0] == 0:
        st.success(f"The model predicts that the patient is at risk of heart disease with a probability of {prediction_proba[0][0]*100:.1f}%.")
    else:
        st.success(f"The model predicts that the patient is not at risk of heart disease with a probability of {prediction_proba[0][1]*100:.1f}%.")

    #------------------

    # Display the results
    #if prediction[0] == 1:
    #    st.success(f"The model predicts that the patient is at risk of heart disease with a probability of {prediction_proba[0][1]*100:.1f}%.")
    #else:
    #    st.success(f"The model predicts that the patient is not at risk of heart disease with a probability of {prediction_proba[0][0]*100:.1f}%.")
    


# Display the ECG data and visualization side by side
col1, col2 = st.columns(2)

with col1:
    with st.expander('📑 ECG Signal Data'):
        if not ecg_df.empty:
            st.write(ecg_df)  # Display the entire raw data file as a table
        else:
            st.write("No data to display.")

with col2:
    with st.expander('📉 ECG Signal Data Visualization'):
        if not ecg_data.empty:
            # Create Altair line chart with labeled axes and red line color
            chart = alt.Chart(ecg_data).mark_line(color='#F63366').encode(
                x=alt.X('Time (s)', title='Time (s)', scale=alt.Scale(domain=[0, time_limit])),
                y=alt.Y('ECG Signal (mV)', title='ECG Signal (mV)')
            ).properties(
                width=350,
                height=400
            ).interactive()
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("No ECG data available to visualize.")
