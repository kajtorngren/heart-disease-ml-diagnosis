import pickle
from pathlib import Path

import streamlit_authenticator as stauth

names = ['Erik Holm', 'Adina Stark', 'Kaj Törngren Sato']
usernames = ['E_holm', 'AdinaLovisa', 'K_Sato']
passwords = ['abc123', 'def456', 'ghi789']

hashed_passwords = stauth.Hasher(passwords).generate()

file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("wb") as file:
    pickle.dump(hashed_passwords, file)



