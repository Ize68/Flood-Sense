# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 20:37:23 2025

@author: USER
"""

import streamlit as st
import pickle
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

MODEL_FILE = "flood_sense_model.pkl"

# 1. Create model if it doesn't exist
if not os.path.exists(MODEL_FILE):
    st.info("Model not found, creating model...")
    data = {
        'rainfall_mm': [10, 50, 200, 5, 300, 150, 0, 100, 250, 80],
        'river_level_m': [1.0, 2.5, 5.0, 0.8, 6.5, 4.0, 0.5, 3.0, 5.5, 2.0],
        'soil_moisture': [30, 50, 90, 20, 95, 80, 15, 60, 85, 45],
        'temperature_C': [25, 28, 30, 22, 31, 27, 21, 29, 30, 26],
        'flood_risk': [0, 0, 1, 0, 1, 1, 0, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    X = df.drop('flood_risk', axis=1)
    y = df['flood_risk']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    st.success("Model created successfully!")
else:
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)

# 2. Streamlit app interface
st.title("🌊 Flood Sense Predictor")

rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 50.0)
river_level = st.number_input("River Level (m)", 0.0, 10.0, 2.0)
soil_moisture = st.number_input("Soil Moisture (%)", 0.0, 100.0, 40.0)
temperature = st.number_input("Temperature (°C)", -10.0, 50.0, 25.0)

if st.button("Predict Flood Risk"):
    input_features = np.array([[rainfall, river_level, soil_moisture, temperature]])
    prediction = model.predict(input_features)[0]
    risk = "⚠️ High Flood Risk" if prediction == 1 else "✅ Low Flood Risk"
    st.success(f"Prediction: {risk}")
