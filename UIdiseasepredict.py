import streamlit as st
import pandas as pd
import joblib
import numpy as np
import cloudpickle
from diseasepredictmodel import DiseaseEnsembleModel

# Title of the app
st.title("🩺 Disease Prediction Tool")

# Load the model
try:
   with open("disease_model.pkl", "rb") as f:
        ensemble = cloudpickle.load(f)
except FileNotFoundError:
    st.error("Model file 'disease_model.pkl' not found. Please upload the model.")
    st.stop()

# Symptom list
symptoms = [
    "fever", "headache", "nausea", "vomiting", "fatigue",
    "jointpain", "skin rash", "cough", "weight loss", "yellow eyes"
]

# Create checkboxes for symptoms
st.header("Select the symptoms you are experiencing:")
user_input = []
for symptom in symptoms:
    value = st.checkbox(symptom.capitalize())
    user_input.append(1 if value else 0)

# Predict button
if st.button("Predict Disease"):
    input_array = np.array([user_input])
    try:
        prediction = ensemble.predict(input_array)[0]
        st.success(f"Predicted Disease: {prediction}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Optional: View raw dataset
with st.expander("📂 View Sample Dataset (datasheet.csv)"):
    try:
        df = pd.read_csv("datasheet.csv")
        st.dataframe(df.head())
    except FileNotFoundError:
        st.warning("Dataset file 'datasheet.csv' not found. Upload it to preview.")
