import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import os

# 1. Page Config & Custom Styling
st.set_page_config(page_title="LifePulse", page_icon="💓", layout="wide")

st.markdown("""
    <style>
    /* Main Background & Font */
    .main { background-color: #0e1117; }
    h1 { color: #ff4b4b; font-family: 'Helvetica Neue', sans-serif; }
    
    /* Pulse Animation for the Heart */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    .heart-icon {
        display: inline-block;
        animation: pulse 2s infinite;
        margin-right: 10px;
    }
    
    /* Custom Button Styling */
    div.stButton > button:first-child {
        background-color: #ff4b4b;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-weight: 600;
        border: none;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #ff1c1c;
        transform: translateY(-2px);
    }
    
    /* Input Field Styling */
    .stNumberInput, .stSelectbox, .stSlider {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Initialize Session State for "One-Click" Buttons
# This allows the sidebar buttons to fill the form automatically
def init_session_state():
    defaults = {
        'age': 25, 'sex': 1, 'cp': 0, 'trestbps': 110, 'chol': 150,
        'fbs': 'No', 'restecg': 0, 'thalach': 175, 'exang': 'No',
        'oldpeak': 0.0, 'slope': 2, 'ca': 0, 'thal': 2
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# 3. Load Assets
@st.cache_resource
def load_assets():
    model_path = os.path.join('models', 'heart_disease_nn_model.keras')
    scaler_path = os.path.join('models', 'scaler.pkl')
    
    # Check for files
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
        
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_assets()

if model is None:
    st.error("🚨 Critical Error: Models not found! Please check your /models folder.")
    st.stop()

# 4. Sidebar: Controls & Info
with st.sidebar:
    st.title("🎛️ Control Panel")
    st.markdown("---")
    
    st.subheader("⚡ Quick Load Profiles")
    col_demo1, col_demo2 = st.columns(2)
    
    with col_demo1:
        if st.button("🟢 Healthy"):
            # Sets inputs to the "Golden Key" healthy values
            st.session_state.update({
                'age': 50, 'sex': 1, 'cp': 2, 'trestbps': 120, 'chol': 200,
                'fbs': 'No', 'restecg': 0, 'thalach': 160, 'exang': 'No',
                'oldpeak': 0.0, 'slope': 0, 'ca': 0, 'thal': 1
            })
            st.rerun()
            
    with col_demo2:
        if st.button("🔴 High Risk"):
            # Sets inputs to generic high risk values
            st.session_state.update({
                'age': 65, 'sex': 1, 'cp': 0, 'trestbps': 160, 'chol': 300,
                'fbs': 'Yes', 'restecg': 2, 'thalach': 100, 'exang': 'Yes',
                'oldpeak': 2.5, 'slope': 1, 'ca': 2, 'thal': 1
            })
            st.rerun()
    
    st.markdown("---")
    st.info("**Model Accuracy:** 99.35%\n\n**Architecture:** Neural Network (3 Hidden Layers)")
    st.warning("⚠️ **Disclaimer:** For educational use only. Not a medical device.")

# 5. Main Title with Animation
st.markdown("<h1><span class='heart-icon'>💓</span> LifePulse</h1>", unsafe_allow_html=True)
st.markdown("### *AI that listens to your heart.*")
st.write("Enter patient vitals below to generate a real-time risk assessment.")

# 6. The Form (Organized into Expanders for Clean UI)
with st.form("medical_form"):
    
    # Group 1: Demographics & Vitals
    with st.expander("👤 Patient Details & Vitals", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age (Years)", 1, 120, key='age')
            sex = st.selectbox("Biological Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female", key='sex')
        with c2:
            trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, key='trestbps')
            chol = st.number_input("Cholesterol (mg/dl)", 100, 600, key='chol')
        with c3:
            fbs_input = st.radio("Fasting Blood Sugar > 120?", ["No", "Yes"], horizontal=True, key='fbs')
            cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], 
                              format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x],
                              key='cp')

# Group 2: Cardiac Stress Test Results (The Confusing Stuff)
    with st.expander("🏃 Cardiac Stress Test (ECG/EKG Data)"):
        st.info("ℹ️ These values usually come from an EKG or Stress Test. If you don't know them, leave them at the default 'Normal' settings.")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            thalach = st.slider("Max Heart Rate", 60, 220, key='thalach',
                                help="The highest heart rate you achieved during exercise. Normal is usually 220 minus your age.")
            
            exang_input = st.radio("Pain during exercise?", ["No", "Yes"], horizontal=True, key='exang',
                                   help="Did you feel chest pain (Angina) while exercising? Select 'No' if unsure.")
        with c2:
            oldpeak = st.number_input("ST Depression", 0.0, 6.2, step=0.1, key='oldpeak',
                                      help="A value measured on an ECG graph. 0.0 is normal. Higher numbers indicate stress.")
            
            slope = st.selectbox("ST Slope", [0, 1, 2], index=2, key='slope',
                                 format_func=lambda x: ["Upsloping (Best)", "Flat (Ok)", "Downsloping (Bad)"][x],
                                 help="The shape of the wave on your ECG. 'Upsloping' is usually healthy.")
        with c3:
            restecg = st.selectbox("Resting ECG", [0, 1, 2], index=0, key='restecg',
                                   format_func=lambda x: ["Normal", "Abnormality", "Hypertrophy"][x],
                                   help="Results from a resting EKG. Select 'Normal' if you haven't had one.")

    # Group 3: Advanced Markers (Blood & Scans)
    with st.expander("🔬 Advanced Markers (Blood/Scan)"):
        c1, c2 = st.columns(2)
        with c1:
            ca = st.selectbox("Major Vessels (Fluoroscopy)", [0, 1, 2, 3, 4], index=0, key='ca',
                              help="How many major blood vessels were colored by dye during a scan. 0 is normal/clear. Higher means blockages.")
        with c2:
            thal = st.selectbox("Thalassemia", [0, 1, 2, 3], index=1, key='thal',
                                format_func=lambda x: ["Unknown", "Normal", "Fixed Defect", "Reversable Defect"][x],
                                help="A blood disorder. 'Normal' is... normal. 'Fixed/Reversable Defect' indicates blood flow issues.")

    # Submit Button
    submitted = st.form_submit_button("🔍 RUN DIAGNOSTIC SCAN")

# 7. Prediction Logic (Strictly Preserved)
if submitted:
    
    # 1. Convert Inputs to Integers (Same logic as before)
    fbs = 1 if fbs_input == "Yes" else 0
    exang = 1 if exang_input == "Yes" else 0
    
    # 2. Create DataFrame with EXACT column order
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                              columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
    
    # 3. Scale ONLY the 5 numerical columns
    cols_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    input_data[cols_to_scale] = scaler.transform(input_data[cols_to_scale])
    
    # 4. Predict
    with st.spinner('Listening to the data...'):
        prediction_prob = model.predict(input_data)[0][0]
    
    # 5. Display Results
    st.divider()
    col_res1, col_res2 = st.columns([3, 1])
    
    with col_res1:
        if prediction_prob < 0.5:
            st.error(f"## ⚠️ High Cardiac Risk Detected")
            st.markdown(f"**Confidence Score: {(1-prediction_prob):.1%}**")
            st.markdown("""
                LifePulse has identified patterns consistent with heart disease. 
                * **Recommendation:** Immediate consultation with a cardiologist is advised.
                * **Next Step:** Perform detailed stress testing and angiography.
            """)
        else:
            st.success(f"## ✅ Low Risk / Healthy Profile")
            st.markdown(f"**Confidence Score: {prediction_prob:.1%}**") # Invert for Health Confidence
            st.markdown("""
                Patient vitals are within healthy parameters.
                * **Recommendation:** Maintain current lifestyle and routine checkups.
                * **Next Step:** No immediate intervention required.
            """)
            st.balloons()

    with col_res2:
        st.write("### Risk Gauge")
        st.progress(float(prediction_prob))