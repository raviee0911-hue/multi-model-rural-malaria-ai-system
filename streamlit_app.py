import streamlit as st
import sqlite3
import hashlib
import numpy as np
from PIL import Image
import keras
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import os
import random

# -----------------------
# Database Setup
# -----------------------

conn = sqlite3.connect("patients.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS patients(
    username TEXT PRIMARY KEY,
    password TEXT,
    name TEXT,
    age INTEGER,
    gender TEXT
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS records(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    image_result TEXT,
    risk_level TEXT,
    latitude REAL,
    longitude REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# -----------------------------
# Load Model
# -----------------------------

model = keras.models.load_model("malaria_model.h5")

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Malaria AI System", layout="wide")

menu = st.sidebar.selectbox("Menu", ["Login", "Sign Up"])

# -----------------------------
# SIGN UP
# -----------------------------

if menu == "Sign Up":

    st.title("📝 Patient Registration")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    name = st.text_input("Full Name")
    age = st.number_input("Age", 1, 100)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    if st.button("Register"):
        try:
            c.execute("INSERT INTO patients VALUES (?,?,?,?,?)",
                      (username, hash_password(password), name, age, gender))
            conn.commit()
            st.success("Registration Successful! Please Login.")
        except:
            st.error("Username already exists!")

# -----------------------------
# LOGIN
# -----------------------------

elif menu == "Login":

    st.title("🔐 Patient Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        c.execute("SELECT * FROM patients WHERE username=? AND password=?",
                  (username, hash_password(password)))
        data = c.fetchone()

        if data:
            st.success("Login Successful!")
            st.session_state["user"] = data
        else:
            st.error("Invalid Credentials")

    # If logged in
    if "user" in st.session_state:

        user = st.session_state["user"]

        st.header(f"Welcome {user[2]} 👋")

        # -----------------------------
        # IMAGE UPLOAD
        # -----------------------------

        st.subheader("🧬 Upload Blood Smear Image")

        uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

        image_prediction = None
        image_result = ""

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            image = image.resize((128,128))
            st.image(image)

            img_array = np.array(image)/255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)[0][0]
            image_prediction = prediction

            if prediction > 0.5:
                image_result = "Parasitized"
                st.error("Malaria Detected")
            else:
                image_result = "Uninfected"
                st.success("No Malaria Detected")

        # -----------------------------
        # PATIENT + WEATHER INPUT
        # -----------------------------

        st.subheader("📊 Enter Clinical & Environmental Data")

        hemoglobin = st.slider("Hemoglobin", 5.0, 18.0, 11.5)
        fever = st.selectbox("Fever?", ["No","Yes"])
        temperature = st.slider("Temperature (°C)", 15, 45, 30)
        humidity = st.slider("Humidity (%)", 0, 100, 70)
        rainfall = st.slider("Rainfall (mm)", 0, 300, 120)

        # -----------------------------
# RUN MULTI-MODEL PREDICTION
# -----------------------------

if st.button("Run Multi-Model Prediction"):

    if image_prediction is None:
        st.warning("Upload image first")
    else:

        # -------------------------
        # Risk Score Calculation
        # -------------------------
        risk_score = 0

        if image_prediction > 0.5:
            risk_score += 50
        if fever == "Yes":
            risk_score += 15
        if hemoglobin < 10:
            risk_score += 15
        if humidity > 65:
            risk_score += 10
        if rainfall > 150:
            risk_score += 10

        if risk_score >= 70:
            risk_level = "HIGH"
        elif risk_score >= 40:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"


        # Example random location (India region)
   

        # -------------------------
        # Save Results in Session
        # -------------------------
        st.session_state["risk_level"] = risk_level
        st.session_state["risk_score"] = risk_score
        st.session_state["image_result"] = image_result
        st.session_state["prediction_done"] = True

        # -------------------------
        # Save to Database
        # -------------------------
        c.execute(
            "INSERT INTO records (username, image_result, risk_level) VALUES (?, ?, ?)",
            (user[0], image_result, risk_level)
        )
        conn.commit()

        lat = random.uniform(8.0, 37.0)
    lon = random.uniform(68.0, 97.0)

    c.execute(
    "INSERT INTO records (username, image_result, risk_level, latitude, longitude) VALUES (?, ?, ?, ?, ?)",
    (st.session_state["username"], image_result, risk_level, lat, lon)
)
    conn.commit()

    st.success("Prediction complete & record saved!")

# -----------------------------
# SHOW RESULT IF EXISTS
# -----------------------------

if "risk_level" in st.session_state:

    st.write("### Final Risk Level:", st.session_state["risk_level"])

    if st.button("Generate PDF Report"):

        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch

        file_name = f"{user[2]}_report.pdf"
        doc = SimpleDocTemplate(file_name)
        elements = []

        styles = getSampleStyleSheet()

        elements.append(Paragraph("Malaria AI Diagnosis Report", styles['Title']))
        elements.append(Spacer(1, 0.5*inch))

        report_text = f"""
        Patient Name: {user[2]}<br/>
        Age: {user[3]}<br/>
        Gender: {user[4]}<br/><br/>

        Image Result: {st.session_state["image_result"]}<br/>
        Hemoglobin: {hemoglobin}<br/>
        Fever: {fever}<br/>
        Temperature: {temperature}<br/>
        Humidity: {humidity}<br/>
        Rainfall: {rainfall}<br/><br/>

        Final Risk Level: {st.session_state["risk_level"]}
        """

        elements.append(Paragraph(report_text, styles['Normal']))
        doc.build(elements)

        with open(file_name, "rb") as f:
            st.download_button(
                label="Download Report",
                data=f,
                file_name=file_name,
                mime="application/pdf"
            )
                # -----------------------------
                # REPORT GENERATION
                # -----------------------------

            if st.button("Generate PDF Report"):

                    file_name = f"{user[2]}_report.pdf"
                    doc = SimpleDocTemplate(file_name)
                    elements = []

                    styles = getSampleStyleSheet()
                    elements.append(Paragraph("Malaria AI Diagnosis Report", styles['Title']))
                    elements.append(Spacer(1, 0.5*inch))

                    report_text = f"""
                    Patient Name: {user[2]}<br/>
                    Age: {user[3]}<br/>
                    Gender: {user[4]}<br/><br/>

                    Image Result: {image_result}<br/>
                    Hemoglobin: {hemoglobin}<br/>
                    Fever: {fever}<br/>
                    Temperature: {temperature}<br/>
                    Humidity: {humidity}<br/>
                    Rainfall: {rainfall}<br/><br/>

                    Final Risk Level: {risk_level}
                    """

                    elements.append(Paragraph(report_text, styles['Normal']))
                    doc.build(elements)

                    with open(file_name, "rb") as f:
                        st.download_button(
                            label="Download Report",
                            data=f,
                            file_name=file_name,
                            mime="application/pdf"
                        )