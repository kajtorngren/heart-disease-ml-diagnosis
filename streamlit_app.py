# URL: https://heart-disease-ml-diagnosis.streamlit.app/
# Streamlit APIs: https://docs.streamlit.io/develop/api-reference

# Modules
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
from scipy.interpolate import interp1d
import neurokit2 as nk
from sklearn.preprocessing import MinMaxScaler
from Ensemble import run_ensemble

# Internetflik visning
st.set_page_config(
        page_title="Heart Failure Detection",
        page_icon="❤",
        layout="wide"
    )

# Add the banner image
st.image("cardiology.jpg", use_container_width=True)

st.title('🧠 Machine Learning For Diagnosing & Monitoring Heart Disease 🫀')
st.info('This app builds a machine learning application for heart disease diagnosis. A prediction is made based on the ECG signal and input features.')

####################################################################################
import pyrebase
from datetime import datetime


# Configuration Key
firebaseConfig = {
  'apiKey': st.secrets["API_KEY"],
  'authDomain': "streamlit-heart-disease-ml.firebaseapp.com",
  'projectId': "streamlit-heart-disease-ml",
  'databaseURL': "https://streamlit-heart-disease-ml-default-rtdb.europe-west1.firebasedatabase.app/",
  'storageBucket': "streamlit-heart-disease-ml.firebasestorage.app",
  'messagingSenderId': "1040645849945",
  'appId': "1:1040645849945:web:a19cc6518fdb3da11d4248",
  'measurementId': "G-NWG4YZ6VSX"
}


# Firebase Authentication
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

# Database
db = firebase.database()
storage = firebase.storage()

# Authentication
choice = st.sidebar.selectbox('Login/Signup', ['Login', 'Sign up'])

# Obtain User Input for email and password
email = st.sidebar.text_input('Please enter your email address')
password = st.sidebar.text_input('Please enter your password',type = 'password')

# App 

# Sign up Block
if choice == 'Sign up':
    handle = st.sidebar.text_input(
        'Please input your app handle name', value='Default')
    submit = st.sidebar.button('Create my account')

    if submit:
        user = auth.create_user_with_email_and_password(email, password)
        st.success('Your account is created successfully!')
        st.balloons()
        # Sign in
        user = auth.sign_in_with_email_and_password(email, password)
        db.child(user['localId']).child("Handle").set(handle)
        db.child(user['localId']).child("ID").set(user['localId'])
        st.title(f'Welcome {handle}')
        st.info('Login via login drop down selection to start you diagnosing!')

# Login Block
if choice == 'Login':
    login = st.sidebar.button('Login')
    if login:
        user = auth.sign_in_with_email_and_password(email,password)
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

#######################################################################################


        # Load the saved model
        modelBPCh = joblib.load('BPCh_model.pkl')
        modelECG = joblib.load('ECG_model.pkl')
        scaler = joblib.load('scaler.pkl')


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





        # Function to resample ECG signal to 187 data points
        def resample_signal(signal, target_length=187):
            x_original = np.linspace(0, 1, len(signal))
            x_resampled = np.linspace(0, 1, target_length)
            interpolator = interp1d(x_original, signal, kind='linear')
            return interpolator(x_resampled)

        # Function to preprocess all RR intervals (between consecutive R-peaks)
        def preprocess_ecg_for_prediction(ecg_values, rpeaks, target_length=187):
            resized_segments = []
        
            # Iterate through all consecutive R-peaks
            for i in range(len(rpeaks['ECG_R_Peaks']) - 1):
                start = rpeaks['ECG_R_Peaks'][i]
                end = rpeaks['ECG_R_Peaks'][i + 1]
            
                # Extract segment between R-peaks
                segment = ecg_values[start:end]
            
                # Resize the segment to 187 data points
                resized_segment = resample_signal(segment, target_length)
            
                # Append the resized segment to the list
                resized_segments.append(resized_segment)
        
            return np.array(resized_segments)


        # Collect other user input features
        def user_input_features():
            age = st.slider('Age', 18, 100, 50)
            sex = st.radio('Sex', ('male', 'female'))
            chest_pain_type = st.selectbox('Chest pain type', ('typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'))
            exercise_induced_angina = st.selectbox('Chest pain from exercise', ('Yes', 'No'))
            resting_bp_s = st.slider('Resting blood pressure (mm Hg)', 90, 200, 120)
            cholesterol = st.slider('Cholesterol (mg/dl)', 150, 300, 200)
            max_heart_rate = st.slider('Max heart rate (bps)', 70, 220, 150)
        
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

        # Ensure column order matches the model's expected feature order
        input_df = input_df[modelBPCh.feature_names_in_]



        # Button for ECG predictions
        if st.button('Predict ECG'):

            # Extract sampling rate from row 8, second column (e.g., "499.348 Hz")
            sampling_rate_str = ecg_df.iloc[8, 1]  # Adjust if sampling rate is stored differently
            sampling_rate = float(sampling_rate_str.split()[0])

            # Detect R-peaks using NeuroKit2
            _, rpeaks = nk.ecg_peaks(ecg_values, sampling_rate=sampling_rate)



            # Preprocess ECG signal for prediction
            X_input = preprocess_ecg_for_prediction(ecg_values, rpeaks)

            X_input_reshaped = X_input.reshape(len(X_input), -1)  # Omforma till 2D (n_samples, 187)
            X_input_normalized = scaler.transform(X_input_reshaped)  # Normalisera
            X_input_normalized = np.clip(X_input_normalized, 0, 1)  # Begränsar värden till intervallet [0, 1]

            X_input = X_input.reshape(len(X_input_normalized), 187, 1)

            # Make predictions for ECG data
            y_pred = modelECG.predict(X_input)
            predicted_classes = np.argmax(y_pred, axis=1)

            st.success(predicted_classes)  # Output will be an array of class labels (0 or 1)
            percentage_ones = (np.sum(predicted_classes == 1) / len(predicted_classes)) * 100

            st.success(f"Risk-percentage of abnormality: {percentage_ones:.2f}%")

        # Button for user input predictions
        if st.button('Predict User Input'):
            # Make predictions for user input features
            prediction = modelBPCh.predict(input_df)  # Invert the predictions if they are inverted
            prediction_proba = modelBPCh.predict_proba(input_df)

            # Display the prediction and probability
            if prediction[0] == 0:
                st.success(f"The model predicts that the patient is at risk of heart disease with a probability of {prediction_proba[0][0]*100:.1f}%.")
            else:
                st.success(f"The model predicts that the patient is not at risk of heart disease with a probability of {prediction_proba[0][1]*100:.1f}%.")

        if st.button('Total predict'):
            # Extract sampling rate from row 8, second column (e.g., "499.348 Hz")
            sampling_rate_str = ecg_df.iloc[8, 1]  # Adjust if sampling rate is stored differently
            sampling_rate = float(sampling_rate_str.split()[0])

            # Detect R-peaks using NeuroKit2
            _, rpeaks = nk.ecg_peaks(ecg_values, sampling_rate=sampling_rate)



            # Preprocess ECG signal for prediction
            X_input = preprocess_ecg_for_prediction(ecg_values, rpeaks)

            X_input_reshaped = X_input.reshape(len(X_input), -1)  # Omforma till 2D (n_samples, 187)
            X_input_normalized = scaler.transform(X_input_reshaped)  # Normalisera
            X_input_normalized = np.clip(X_input_normalized, 0, 1)  # Begränsar värden till intervallet [0, 1]

            X_input = X_input.reshape(len(X_input_normalized), 187, 1)

            # Make predictions for ECG data
            y_pred = modelECG.predict(X_input)
            predicted_classes = np.argmax(y_pred, axis=1)

            percentage_ones = (np.sum(predicted_classes == 1) / len(predicted_classes)) * 100
        
            prediction = modelBPCh.predict(input_df)  # Invert the predictions if they are inverted
            prediction_proba = modelBPCh.predict_proba(input_df)


        
            if prediction[0] == 0:
                BPCh_pred_prob = prediction_proba[0][0]

            else: 
                BPCh_pred_prob = prediction_proba[0][1]
        

            res = run_ensemble(percentage_ones,prediction[0],BPCh_pred_prob)

            if len(res) > 1:
                st.success(f'{res[0]} {res[1]}')
            else:
                st.success(res[0])






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
