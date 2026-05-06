import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import sqlite3
import random

from auth import create_table, signup, login
from risk_model import RiskModel
from fusion_model import final_decision

# -----------------------------
# Create database tables
# -----------------------------
create_table()

# Create records table if not exists
conn = sqlite3.connect("patients.db")
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS records(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    image_result TEXT,
    risk_level TEXT,
    latitude REAL,
    longitude REAL
)
""")

conn.commit()
conn.close()

# -----------------------------
# Load AI Models
# -----------------------------
model = tf.keras.models.load_model(
    "malaria_model.h5",
    compile=False
)

risk_model = RiskModel()

# -----------------------------
# Session State
# -----------------------------
if "user" not in st.session_state:
    st.session_state.user = None

# -----------------------------
# Title
# -----------------------------
st.title("🦟 Multi-Model Rural Malaria AI System")

menu = ["Login", "Sign Up"]

# =========================================================
# LOGIN / SIGNUP
# =========================================================
if st.session_state.user is None:

    choice = st.sidebar.selectbox("Menu", menu)

    # -----------------------------
    # SIGN UP
    # -----------------------------
    if choice == "Sign Up":

        st.subheader("Create Patient Account")

        name = st.text_input("Full Name")
        age = st.number_input("Age", 1, 100)
        gender = st.selectbox("Gender", ["Male", "Female"])
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Register"):

            if signup(name, age, gender, email, password):
                st.success("✅ Account Created Successfully")
            else:
                st.error("❌ Email already exists")

    # -----------------------------
    # LOGIN
    # -----------------------------
    elif choice == "Login":

        st.subheader("Patient Login")

        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Login"):

            user = login(email, password)

            if user:
                st.session_state.user = user
                st.success("✅ Login successful")
                st.rerun()

            else:
                st.error("❌ Invalid credentials")

# =========================================================
# MAIN SYSTEM
# =========================================================
else:

    st.success(f"Welcome {st.session_state.user[1]}")

    # -----------------------------
    # IMAGE UPLOAD
    # -----------------------------
    st.header("🧬 Step 1: Upload Blood Smear Image")

    uploaded_file = st.file_uploader(
        "Upload Blood Smear Image",
        type=["jpg", "jpeg", "png"]
    )

    image_prob = None
    image_result = "Unknown"

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize((128, 128))

        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert image to numpy
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array)

        image_prob = float(prediction[0][0])

        if image_prob > 0.5:
            st.error("⚠ Parasitized Detected")
            image_result = "Parasitized"
        else:
            st.success("✅ Uninfected")
            image_result = "Uninfected"

    # -----------------------------
    # PATIENT DATA INPUT
    # -----------------------------
    st.header("📊 Step 2: Patient + Weather Data")

    temperature = st.slider("Temperature (°C)", 15, 45, 30)
    rainfall = st.slider("Rainfall (mm)", 0, 300, 100)
    humidity = st.slider("Humidity (%)", 0, 100, 70)
    hemoglobin = st.slider("Hemoglobin Level", 5.0, 18.0, 11.0)
    previous_infection = st.selectbox(
        "Previous Infection History",
        [0, 1]
    )

    # -----------------------------
    # RUN PREDICTION
    # -----------------------------
    if st.button("Run Full Multi-Model Prediction"):

        if image_prob is None:
            st.warning("⚠ Please upload image first")

        else:

            # Risk model input
            risk_input = [
                temperature,
                rainfall,
                humidity,
                st.session_state.user[2],  # age
                hemoglobin,
                previous_infection,
                image_prob
            ]

            # Predict probabilities
            risk_probs = risk_model.model.predict_proba(
                risk_model.scaler.transform([risk_input])
            )[0]

            # Final fusion decision
            result = final_decision(image_prob, risk_probs)

            # -----------------------------
            # DISPLAY RESULTS
            # -----------------------------
            st.header("🤖 Final Multi-Model AI Decision")

            st.write("### Species")
            st.write(result["species"])

            st.write("### Infection Confidence")
            st.write(round(result["infection_confidence"], 2))

            st.write("### Risk Level")
            st.write(result["risk_level"])

            st.write("### Risk Confidence")
            st.write(round(result["risk_confidence"], 2))

            st.write("### Overall Status")
            st.write(result["final_status"])

            # -----------------------------
            # SAVE RESULTS
            # -----------------------------
            conn = sqlite3.connect("patients.db")
            c = conn.cursor()

            # Random India coordinates
            lat = random.uniform(8.0, 37.0)
            lon = random.uniform(68.0, 97.0)

            c.execute(
                """
                INSERT INTO records
                (username, image_result, risk_level, latitude, longitude)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    st.session_state.user[1],
                    image_result,
                    result["risk_level"],
                    lat,
                    lon
                )
            )

            conn.commit()
            conn.close()

            st.success("✅ Prediction saved successfully!")

    # -----------------------------
    # LOGOUT
    # -----------------------------
    if st.button("Logout"):

        st.session_state.user = None
        st.rerun()

