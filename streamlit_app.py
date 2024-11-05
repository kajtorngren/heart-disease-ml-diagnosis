import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Add the banner image
st.image("cardiology-banner.jpg", use_column_width=True)  # Ensure the path to your image is correct

st.title('🫀 Heart Disease ML Diagnosis 🫀')

st.info('This app builds a machine learning application for heart disease diagnosis.')

st.sidebar.header('User Input Features')

# Initialize ecg_df
ecg_df = pd.DataFrame()

# Upload ECG signals CSV file
uploaded_file = st.sidebar.file_uploader("Upload your ECG signal CSV file. Designed for Samsung Health Monitor App with Samsung Galaxy Watch 3.", type=["csv"])
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
    age = st.sidebar.slider('Age', 0, 100, 52)
    sex = st.sidebar.radio('Sex', ('male', 'female'))
    chest_pain_type = st.sidebar.selectbox('Chest pain type', ('typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'))
    resting_bp_s = st.sidebar.slider('Resting blood pressure (mm Hg)', 100, 200, 153)
    serum_cholesterol = st.sidebar.number_input('Cholesterol (mg/dl)', value=None, placeholder='Type a number...')
    fasting_blood_sugar = st.sidebar.selectbox('Fasting blood sugar', ('> 120 mg/dl', '< 120 mg/dl'))
    max_heart_rate = st.sidebar.slider('Max heart rate (bps)', 71, 202, 172)
    
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

# Display raw ECG data in a table
with st.expander('ECG Signal Data'):
    if not ecg_df.empty:
        st.write(ecg_df)  # Display the entire raw data file as a table
    else:
        st.write("No data to display.")

# ECG Data Visualization with axis labels and red line color
with st.expander('ECG Signal Data Visualization'):
    if not ecg_data.empty:
        # Create Altair line chart with labeled axes and red line color
        chart = alt.Chart(ecg_data).mark_line(color='red').encode(
            x=alt.X('Time (s)', title='Time (s)', scale=alt.Scale(domain=[0, time_limit])),
            y=alt.Y('ECG Signal (mV)', title='ECG Signal (mV)')
        ).properties(
            width=700,
            height=400
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.write("No ECG data available to visualize.")
