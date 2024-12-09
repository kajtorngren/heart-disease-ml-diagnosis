# URL: https://heart-disease-ml-diagnosis.streamlit.app/
# Streamlit APIs: https://docs.streamlit.io/develop/api-reference

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
from scipy.interpolate import interp1d
import neurokit2 as nk
from sklearn.preprocessing import MinMaxScaler
from Ensemble import run_ensemble
import streamlit_authenticator as stauth
import json
from pathlib import Path

# Fil för att lagra användaruppgifter
CREDENTIALS_FILE = "credentials.json"

# Ladda eller initiera användaruppgifter
if Path(CREDENTIALS_FILE).exists():
    with open(CREDENTIALS_FILE, "r") as f:
        credentials = json.load(f)
else:
    credentials = {
        "usernames": {
            "testuser": {"password": "$2b$12$hashedpassword1"}
        }
    }


def register_user():
    st.sidebar.title("Registrera nytt konto")
    username = st.sidebar.text_input("Välj användarnamn")
    password = st.sidebar.text_input("Välj lösenord", type="password")
    confirm_password = st.sidebar.text_input("Bekräfta lösenord", type="password")

    if st.sidebar.button("Registrera"):
        if username in credentials["usernames"]:
            st.sidebar.error("Användarnamnet finns redan. Välj ett annat.")
        elif password != confirm_password:
            st.sidebar.error("Lösenorden matchar inte.")
        else:
            # Hasha lösenordet
            hashed_password = stauth.Hasher([password]).generate()[0]
            # Spara ny användare
            credentials["usernames"][username] = {"password": hashed_password}
            with open(CREDENTIALS_FILE, "w") as f:
                json.dump(credentials, f)
            st.sidebar.success("Konto skapat! Du kan nu logga in.")


# Sätt upp autentisering
authenticator = stauth.Authenticate(
    {"usernames": credentials["usernames"]},
    "app_name",
    "unique_signature_key",  # Byt ut mot en egen unik sträng
    cookie_expiry_days=30
)

# Välj mellan inloggning och registrering
mode = st.sidebar.radio("Välj åtgärd", ["Logga in", "Registrera"])

if mode == "Logga in":
    try:
        name, authentication_status, username = authenticator.login('Logga in', 'main')

        if authentication_status:
            st.success(f"Välkommen {name}!")
            authenticator.logout("Logga ut", 'sidebar')
            # Här lägger du huvudlogiken för din app
            st.title("Din app är nu tillgänglig")
        elif authentication_status == False:
            st.error("Ogiltigt användarnamn eller lösenord")
        elif authentication_status == None:
            st.warning("Ange ditt användarnamn och lösenord")
    except Exception as e:
        st.error(f"Fel vid inloggning: {e}")

elif mode == "Registrera":
    register_user()


# Ladda sparade modeller
modelBPCh = joblib.load('BPCh_model.pkl')
modelECG = joblib.load('ECG_model.pkl')
scaler = joblib.load('scaler.pkl')

# Appinställningar
st.set_page_config(
    page_title="Heart Failure Detection",
    page_icon="❤",
    layout="wide"
)

# Bannerbild
st.image("cardiology.jpg", use_column_width=True)

st.title('🧠 Machine Learning For Diagnosing & Monitoring Heart Disease 🫀')
st.info('This app builds a machine learning application for heart disease diagnosis. A prediction is made based on the ECG signal and input features.')

st.sidebar.header('📝 User Input Features')

# Initialize ecg_df
ecg_df = pd.DataFrame()

# Ladda upp ECG-signaler CSV-fil
uploaded_file = st.sidebar.file_uploader(
    "Upload your ECG signal CSV file.  \nDesigned for Samsung Health Monitor App with Samsung Galaxy Watch 3.",
    type=["csv"]
)

if uploaded_file is not None:
    ecg_df = pd.read_csv(uploaded_file, header=None)
    
    # Extrahera samplingsfrekvens
    sampling_rate_str = ecg_df.iloc[8, 1]  # e.g., "499.348 Hz"
    sampling_rate = float(sampling_rate_str.split()[0])
    
    # Extrahera ECG-signalvärden från rad 15 (index 14)
    ecg_values = ecg_df[0][14:].astype(float).reset_index(drop=True)
    
    # Skapa tidsaxel baserat på samplingsfrekvens
    time_axis = [i / sampling_rate for i in range(len(ecg_values))]
    
    # Kombinera till en DataFrame för visualisering
    ecg_data = pd.DataFrame({'Time (s)': time_axis, 'ECG Signal (mV)': ecg_values})
    
    # Sätt tidsgräns
    time_limit = time_axis[-1]
else:
    st.sidebar.warning("Please upload an ECG signal CSV file.")
    ecg_data = pd.DataFrame(columns=['Time (s)', 'ECG Signal (mV)'])  # Tom DataFrame
    time_limit = None


# Funktion för att interpolera ECG-signal
def resample_signal(signal, target_length=187):
    x_original = np.linspace(0, 1, len(signal))
    x_resampled = np.linspace(0, 1, target_length)
    interpolator = interp1d(x_original, signal, kind='linear')
    return interpolator(x_resampled)

# Funktion för att bearbeta RR-intervall
def preprocess_ecg_for_prediction(ecg_values, rpeaks, target_length=187):
    resized_segments = []
    
    # Iterera genom alla R-toppar
    for i in range(len(rpeaks['ECG_R_Peaks']) - 1):
        start = rpeaks['ECG_R_Peaks'][i]
        end = rpeaks['ECG_R_Peaks'][i + 1]
        
        # Extrahera segment mellan R-toppar
        segment = ecg_values[start:end]
        
        # Interpolera segment till 187 datapunkter
        resized_segment = resample_signal(segment, target_length)
        
        # Lägg till i lista
        resized_segments.append(resized_segment)
    
    return np.array(resized_segments)

# Funktion för att samla användarinput
def user_input_features():
    age = st.sidebar.slider('Age', 18, 100, 50)
    sex = st.sidebar.radio('Sex', ('male', 'female'))
    chest_pain_type = st.sidebar.selectbox('Chest pain type', ('typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'))
    exercise_induced_angina = st.sidebar.selectbox('Chest pain from exercise', ('Yes', 'No'))
    resting_bp_s = st.sidebar.slider('Resting blood pressure (mm Hg)', 90, 200, 120)
    cholesterol = st.sidebar.number_input('Cholesterol (mg/dl)', value=200)
    max_heart_rate = st.sidebar.slider('Max heart rate (bps)', 70, 220, 150)
    
    chest_pain_type_mapping = {
        'typical angina': 1,
        'atypical angina': 2,
        'non-anginal pain': 3,
        'asymptomatic': 4
    }

    chest_pain_type_encoded = chest_pain_type_mapping[chest_pain_type]
    
    # Kombinera inputs till en DataFrame
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

# Få användarinput
input_df = user_input_features()

# Fler funktioner från din ursprungliga kod här...
