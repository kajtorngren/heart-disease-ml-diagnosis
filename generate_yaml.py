import yaml
import streamlit_authenticator as stauth

# Skapa en dictionary med användarnamn och plaintext-lösenord
credentials = {
    "usernames": {
        "E_Holm": {
            "name": "Erik Holm",
            "password": "abc123",  # Ursprungligt lösenord
        },
        "A_Stark": {
            "name": "Adina Stark",
            "password": "def456",  # Ursprungligt lösenord
        },
    }
}

# Extrahera plaintext-lösenord
passwords = [user["password"] for user in credentials["usernames"].values()]

# Generera hashade lösenord
hashed_passwords = stauth.Hasher(passwords).generate()

# Uppdatera credentials med hashade lösenord
for i, username in enumerate(credentials["usernames"]):
    credentials["usernames"][username]["password"] = hashed_passwords[i]

# Lägg till cookie-inställningar
config = {
    "credentials": credentials,
    "cookie": {
        "expiry_days": 30,
        "key": "auth_key",
    },
    "preauthorized": {
        "emails": []
    },
}

# Spara till YAML-fil
with open("config.yaml", "w") as file:
    yaml.dump(config, file, default_flow_style=False)

print("YAML-filen har uppdaterats med hashade lösenord.")