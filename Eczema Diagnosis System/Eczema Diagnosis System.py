import streamlit as st
import hashlib
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
import time
from datetime import datetime, timedelta

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'email_verified' not in st.session_state:
    st.session_state.email_verified = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None

# Email configuration
GMAIL_USERNAME = "muneerinuse@gmail.com"  # Replace with your Gmail
GMAIL_APP_PASSWORD = "ectd ealv zyrl jchy"  # Replace with your App Password

# Your original symptoms data
universal_symptoms = {'rash', 'itchy_skin', 'redness', 'scaling'}
types = {
    'craked skin', 'burning sensation', 'Small, itchy blisters on hands or feet',
    'Round, coin-shaped spots', 'Yellowish, greasy scales', 'Swelling in lower legs', 'Intensely itchy'
}
symptoms = [
    'rash', 'dry skin', 'cracked skin', 'inflamed_skin', 'blisters',
    'thickened_skin', 'weeping_skin', 'itchy_skin', 'redness', 'scaling',
    'burning sensation', 'Small, itchy blisters on hands or feet',
    'Round, coin-shaped spots', 'Yellowish, greasy scales', 'Swelling in lower legs', 'Intensely itchy'
]


def send_verification_email(to_email):
    code = str(random.randint(100000, 999999))

    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = GMAIL_USERNAME
        msg['To'] = to_email
        msg['Subject'] = "Your Verification Code"

        body = f"Your verification code is: {code}"
        msg.attach(MIMEText(body, 'plain'))

        # Create server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(GMAIL_USERNAME, GMAIL_APP_PASSWORD)

        # Send email
        text = msg.as_string()
        server.sendmail(GMAIL_USERNAME, to_email, text)
        server.quit()

        return code
    except Exception as e:
        st.error(f"Error sending email: {str(e)}")
        return None


def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            email TEXT NOT NULL,
            security_question TEXT NOT NULL,
            security_answer TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(username, password, email, security_question, security_answer):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        hashed_password = hash_password(password)
        hashed_answer = hash_password(security_answer.lower())
        c.execute(
            "INSERT INTO users (username, password, email, security_question, security_answer) VALUES (?, ?, ?, ?, ?)",
            (username, hashed_password, email, security_question, hashed_answer)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def verify_login(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT password, email FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()

    if result and result[0] == hash_password(password):
        return True, result[1]
    return False, None


def verify_security_answer(username, answer):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT security_answer FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()

    return result and result[0] == hash_password(answer.lower())


def check_eczema(symptoms_input):
    # Your original diagnosis logic
    if (len(universal_symptoms.intersection(symptoms_input)) == 4) and (len(types.intersection(symptoms_input)) == 0):
        return "Eczema. Consult a doctor!"
    if len(universal_symptoms.intersection(symptoms_input)) <= 3:
        return "Not Eczema. Consult a doctor!"
    if universal_symptoms and 'cracked skin' in symptoms_input:
        return "Eczema(Atopic Dermatitis). Consult a doctor!"
    if universal_symptoms and 'burning sensation' in symptoms_input:
        return "Eczema(Contact Dermatitis). Consult a doctor!"
    if universal_symptoms and 'Small, itchy blisters on hands or feet' in symptoms_input:
        return "Eczema(Dyshidrotic Eczema). Consult a doctor!"
    if universal_symptoms and 'Round, coin-shaped spots' in symptoms_input:
        return "Eczema(Nummular Eczema). Consult a doctor!"
    if universal_symptoms and 'Yellowish, greasy scales' in symptoms_input:
        return "Eczema(Seborrheic Dermatitis). Consult a doctor!"
    if universal_symptoms and 'Swelling in lower legs' in symptoms_input:
        return "Eczema(Stasis Dermatitis). Consult a doctor!"
    if universal_symptoms and 'Intensely itchy' in symptoms_input:
        return "Eczema(Neurodermatitis). Consult a doctor!"


def main():
    init_db()
    st.title("Secure Eczema Diagnosis System")

    # Sidebar for authentication
    with st.sidebar:
        st.header("Authentication")
        if not st.session_state.authenticated:
            tab1, tab2 = st.tabs(["Login", "Register"])

            with tab1:
                with st.form("login_form"):
                    username = st.text_input("Username")
                    password = st.text_input("Password", type="password")
                    submit_button = st.form_submit_button("Login")

                    if submit_button:
                        success, email = verify_login(username, password)
                        if success:
                            # Send verification code
                            verification_code = send_verification_email(email)
                            if verification_code:
                                st.session_state.verification_code = verification_code
                                st.session_state.verifying_username = username
                                st.success("Verification code sent to your email!")
                            else:
                                st.error("Failed to send verification code")
                        else:
                            st.error("Invalid username or password")

                # Verification code input
                if 'verification_code' in st.session_state:
                    code = st.text_input("Enter verification code")
                    if st.button("Verify Code"):
                        if code == st.session_state.verification_code:
                            st.session_state.email_verified = True
                            # Proceed to security question
                            st.success("Email verified! Please answer your security question.")
                        else:
                            st.error("Invalid verification code")

                # Security question verification
                if st.session_state.email_verified:
                    security_answer = st.text_input("What is your mother's maiden name?")
                    if st.button("Verify Answer"):
                        if verify_security_answer(st.session_state.verifying_username, security_answer):
                            st.session_state.authenticated = True
                            st.session_state.current_user = st.session_state.verifying_username
                            st.success("Successfully authenticated!")
                            st.rerun()
                        else:
                            st.error("Incorrect security answer")

            with tab2:
                with st.form("register_form"):
                    new_username = st.text_input("Choose Username")
                    new_password = st.text_input("Choose Password", type="password")
                    confirm_password = st.text_input("Confirm Password", type="password")
                    email = st.text_input("Email Address")
                    security_question = "What is your mother's maiden name?"
                    st.write(security_question)
                    security_answer = st.text_input("Security Answer")
                    register_button = st.form_submit_button("Register")

                    if register_button:
                        if new_password != confirm_password:
                            st.error("Passwords do not match!")
                        elif register_user(new_username, new_password, email, security_question, security_answer):
                            st.success("Registration successful! Please login.")
                        else:
                            st.error("Username already exists!")
        else:
            st.write(f"Welcome, {st.session_state.current_user}!")
            if st.button("Logout"):
                st.session_state.authenticated = False
                st.session_state.email_verified = False
                st.session_state.current_user = None
                st.rerun()

    # Main content area - Eczema Diagnosis System
    if st.session_state.authenticated:
        st.header("Eczema Diagnosis")
        symptoms_input = st.multiselect(
            "Select symptoms you are experiencing:",
            options=symptoms,
            default=[]
        )

        if st.button("Diagnose"):
            if symptoms_input:
                diagnosis = check_eczema(symptoms_input)
                st.write(f"Diagnosis: {diagnosis}")
            else:
                st.write("Please select at least one symptom for diagnosis.")
    else:
        st.info("Please login to access the diagnosis system.")


if __name__ == "__main__":
    main()
