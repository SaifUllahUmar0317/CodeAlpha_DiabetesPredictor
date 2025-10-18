import pandas as pd
import joblib
import numpy as np
import streamlit as st

# -------------------------------
# App Configuration & Styling
# -------------------------------
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="wide")

st.markdown(
    """
    <style>
    body {
        background-color: #f9fafc;
        color: #333333;
    }
    .main-title {
        text-align: center;
        font-size: 40px;
        color: #1a73e8;
        font-weight: bold;
        margin-bottom: -10px;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 18px;
    }
    .stButton>button {
        width: 100%;
        background-color: #1a73e8;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px 0;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #155ab6;
    }
    .result-box {
        text-align: center;
        padding: 20px;
        border-radius: 15px;
        background-color: #e3f2fd;
        margin-top: 20px;
        font-size: 22px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# Title Section
# -------------------------------
st.markdown('<h1 class="main-title">🩺 Diabetes Prediction App</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict diabetes risk by entering patient details below.</p>', unsafe_allow_html=True)
st.write("---")

# -------------------------------
# Sidebar for Input
# -------------------------------
st.sidebar.header("🔍 Patient Information")
st.sidebar.markdown("Enter the following details to get prediction:")

features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

pregnancies = st.sidebar.number_input("🤰 Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.sidebar.number_input("🩸 Glucose Level", min_value=0, max_value=300, value=100)
blood_pressure = st.sidebar.number_input("💓 Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.sidebar.number_input("🧬 Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.sidebar.number_input("💉 Insulin Level", min_value=0, max_value=900, value=80)
bmi = st.sidebar.number_input("⚖️ BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
dpf = st.sidebar.number_input("🧠 Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.2f")
age = st.sidebar.number_input("🎂 Age", min_value=0, max_value=120, value=30)

data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
X = pd.DataFrame(np.reshape(data, (1, 8)), columns=features, index=range(1))

# -------------------------------
# Load Scaler & Model
# -------------------------------
with open("Scaler.pkl", 'rb') as sc:
    scaler = joblib.load(sc)
X_scaled = scaler.transform(X)

with open("Diabetes pedictor model.pkl", 'rb') as mdl:
    model = joblib.load(mdl)

# -------------------------------
# Prediction Button
# -------------------------------
st.markdown("### 🧾 Prediction Result")
if st.button("🔍 Predict Diabetes Risk"):
    Y = model.predict(X_scaled)
    prob = model.predict_proba(X_scaled)[0][1]

    if Y[0] == 1:
        st.markdown(
            f'<div class="result-box" style="background-color:#fdecea;color:#d32f2f;">'
            f'⚠️ <b>Diabetes Detected</b><br>Estimated Risk: <b>{prob:.2%}</b></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-box" style="background-color:#e8f5e9;color:#2e7d32;">'
            f'✅ <b>No Diabetes Detected</b><br>Estimated Risk: <b>{prob:.2%}</b></div>',
            unsafe_allow_html=True
        )

# -------------------------------
# Developer Section
# -------------------------------
st.markdown("---")
st.markdown(
    """
    <p style='text-align: center; font-size: 16px; color: #777;'>
    Developed by <a href="https://github.com/SaifUllahUmar0317" target="_blank" style="color:#1a73e8;text-decoration:none;"><b>Saifullah Umar</b></a> 🚀
    </p>
    """,
    unsafe_allow_html=True
)
