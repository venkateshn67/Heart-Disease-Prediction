# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import plotly.express as px

# ----------------------------------------------------------------------
# A. Helper Functions
# ----------------------------------------------------------------------

@st.cache_data
def get_binary_file_downloader_html(df, file_label='Prediction CSV'):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    
    html_link = f"""
    <a href="data:file/csv;base64,{b64}" 
       download="predicted_heart.csv"
       target="_blank"
    >
        <button style="
            background-color: #4CAF50; border: none; color: white;
            padding: 10px 24px; text-align: center; text-decoration: none;
            display: inline-block; font-size: 16px; margin: 4px 2px;
            cursor: pointer; border-radius: 8px;
        ">{file_label}</button>
    </a>
    """
    return html_link

# ----------------------------------------------------------------------
# B. Global Variables and Setup
# ----------------------------------------------------------------------

st.set_page_config(layout="wide")
st.title("❤️ Heart Disease Predictor")

# Adjusted paths for files located in the root directory
MODEL_PATH_PREFIX = ''

MODEL_FILES = {
    'Decision Tree': 'dt_model.pkl',
    'Logistic Regression': 'lr_model.pkl',
    'Random Forest': 'rf_model.pkl',
    'Support Vector Machine': 'svm_model.pkl',
    'Grid Random Forest': 'gridrf_model.pkl' 
}

FEATURE_ORDER = [
    'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
    'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 
    'Oldpeak', 'ST_Slope'
]

try:
    with open(f'{MODEL_PATH_PREFIX}scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    SCALER_LOADED = True
except FileNotFoundError:
    st.error("FATAL ERROR: 'scaler.pkl' not found in the root directory. Rerun training script and ensure scaler is saved.")
    SCALER_LOADED = False


def predict_heart_disease(input_data):
    """Loads all models and gets predictions for the input data."""
    if not SCALER_LOADED:
        return [-1] * len(MODEL_FILES)
        
    # Scale the input data before prediction
    input_array_scaled = scaler.transform(input_data) 

    predictions = []
    for model_name, file_name in MODEL_FILES.items():
        try:
            with open(file_name, 'rb') as file:
                model = pickle.load(file)
            
            # Predict using the scaled array
            pred = model.predict(input_array_scaled) 
            predictions.append(pred[0])
            
        except FileNotFoundError:
            predictions.append(-1)
            
    return predictions

# ----------------------------------------------------------------------
# UI Code 
# ----------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["Predict", "Bulk Predict", "Model Information"])

# ======================================================================
# TAB 1: Single Instance Prediction (Predict)
# ======================================================================

with tab1:
    st.header("Single Patient Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age (Years)", min_value=28, max_value=150, value=50)
        sex_text = st.selectbox("Sex", ["Male", "Female"])
        cpt_text = st.selectbox("Chest Pain Type", ["Typical Angina (TA)", "Atypical Angina (ATA)", "Non-Anginal Pain (NAP)", "Asymptomatic (ASY)"])
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=70, max_value=200, value=120)
        
    with col2:
        cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=650, value=200)
        fbs_text = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
        ecg_text = st.selectbox("Resting ECG", ["Normal", "ST-T wave abnormality (ST)", "Left Ventricular Hypertrophy (LVH)"])
        max_hr = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=202, value=150)
        
    with col3:
        exang_text = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        oldpeak = st.number_input("Oldpeak (ST depression)", min_value=-3.0, max_value=7.0, value=1.0, step=0.1)
        slope_text = st.selectbox("ST Slope", ["Up-sloping", "Flat", "Down-sloping"])

    # --- Pre-processing/Encoding ---
    sex = 0 if sex_text == "Male" else 1
    cpt_mapping = {"Atypical Angina (ATA)": 0, "Non-Anginal Pain (NAP)": 1, "Asymptomatic (ASY)": 2, "Typical Angina (TA)": 3}
    cpt = cpt_mapping.get(cpt_text, 0) 
    fbs = 0 if fbs_text == "<= 120 mg/dl" else 1
    ecg_mapping = {"Normal": 0, "Left Ventricular Hypertrophy (LVH)": 1, "ST-T wave abnormality (ST)": 2}
    ecg = ecg_mapping.get(ecg_text, 0)
    exang = 0 if exang_text == "No" else 1
    slope_mapping = {"Down-sloping": 0, "Flat": 1, "Up-sloping": 2}
    slope = slope_mapping.get(slope_text, 2)


    # Create the input DataFrame 
    input_list = [
        age, sex, cpt, resting_bp, cholesterol, fbs, ecg, max_hr, exang, oldpeak, slope
    ]
    input_data = pd.DataFrame([input_list], columns=FEATURE_ORDER).astype(float)


    # --- Prediction and Results ---
    st.markdown("---")
    
    if st.button("Submit for Prediction"):
        if SCALER_LOADED:
            st.subheader("Results")
            results = predict_heart_disease(input_data)
            model_names = list(MODEL_FILES.keys())
            
            for i, model_name in enumerate(model_names):
                prediction = results[i]
                res_col1, res_col2 = st.columns([1, 2])
                
                with res_col1:
                    st.write(f"**{model_name}**:")
                
                with res_col2:
                    if prediction == 1:
                        st.error("Heart Disease Detected! 💔")
                    elif prediction == 0:
                        st.success("No Heart Disease Detected! ✅")
                    else:
                        st.warning("Prediction Error. Check console.")
        else:
            st.warning("Cannot predict. Please resolve the file loading error.")


# ======================================================================
# TAB 2: Bulk Prediction
# ======================================================================

with tab2:
    st.header("Bulk Prediction from CSV File")

    st.subheader("Instructions for CSV Upload")
    st.info("""
    1. **No NaN values allowed.** All cells must be filled.
    2. **11 features** must be in this **exact order**: `Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope`.
    3. **Input the numerical codes** (0, 1, 2...) for categorical columns (e.g., Sex: 0 for Male, 1 for Female).
    """)

    uploaded_file = st.file_uploader("Upload CSV File to Get Predictions", type=["csv"])

    if uploaded_file is not None and SCALER_LOADED:
        try:
            input_df = pd.read_csv(uploaded_file)
            
            if list(input_df.columns) == FEATURE_ORDER:
                
                # Load LR model and scale bulk data
                with open(f'{MODEL_PATH_PREFIX}lr_model.pkl', 'rb') as file:
                    lr_model = pickle.load(file)
                
                # Scale the entire bulk DataFrame
                input_array_scaled = scaler.transform(input_df.astype(float))
                
                # Predict using the scaled array
                predictions = lr_model.predict(input_array_scaled)
                input_df['Prediction (LR)'] = predictions
                
                st.subheader("Prediction Results")
                st.dataframe(input_df)

                st.markdown(get_binary_file_downloader_html(input_df, 'Download Predicted CSV'), unsafe_allow_html=True)
                
                st.success("Bulk prediction complete!")
                
            else:
                st.warning("Please ensure the uploaded CSV has the correct columns in the right order.")

        except Exception as e:
            st.error(f"An error occurred during file processing: {e}")
            st.warning("Check your CSV file format and data types. Ensure all cells are numerical for prediction.")
    elif not SCALER_LOADED:
         st.warning("Cannot process bulk predictions due to missing 'scaler.pkl' file.")
    else:
        st.info("Upload a CSV file to get heart disease predictions in bulk.")


# ======================================================================
# TAB 3: Model Information
# ======================================================================

with tab3:
    st.header("Model Performance Comparison")
    
    data = {
        'Model': ['Decision Tree', 'Logistic Regression', 'Random Forest', 'Support Vector Machine', 'Grid Random Forest'],
        'Accuracy (%)': [80.97, 85.86, 82.60, 84.20, 89.00] 
    }
    accuracy_df = pd.DataFrame(data)

    st.dataframe(accuracy_df, use_container_width=True)
    
    fig = px.bar(
        accuracy_df, 
        x='Model', 
        y='Accuracy (%)', 
        title='Accuracy Scores of Trained Models',
        color='Model',
        text='Accuracy (%)',
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    
    st.plotly_chart(fig, use_container_width=True)