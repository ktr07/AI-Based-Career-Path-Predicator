import streamlit as st
import numpy as np
import pickle
import os

st.set_page_config(page_title="AI Career Path Predictor", page_icon="🧠")

# Title
st.title("🧠 AI Career Path Predictor")
st.markdown("Enter your skills and preferences to get your ideal tech career suggestion!")

# Load model and encoders safely
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Required file `model.pkl` or `label_encoders.pkl` is missing.")
    st.stop()

# Sliders for input
programming = st.slider("Programming Skills (0-10)", 0, 10, 5)
math = st.slider("Math/Statistics Skills (0-10)", 0, 10, 5)
communication = st.slider("Communication Skills (0-10)", 0, 10, 5)
creativity = st.slider("Creativity (0-10)", 0, 10, 5)
research = st.slider("Interest in Research (0-10)", 0, 10, 5)
projects = st.slider("Projects Completed", 0, 10, 2)
domain_knowledge = st.slider("Domain Knowledge (0-10)", 0, 10, 5)

# Dropdowns
preference = st.selectbox("Domain Preference", label_encoders['Preference'].classes_)
stream = st.selectbox("Academic Stream", label_encoders['Academic_Stream'].classes_)

# Encode
encoded_preference = label_encoders['Preference'].transform([preference])[0]
encoded_stream = label_encoders['Academic_Stream'].transform([stream])[0]

# Input Array
input_data = np.array([[programming, math, communication, creativity, research,
                        projects, domain_knowledge, encoded_preference, encoded_stream]])

# Predict
if st.button("Predict Career Path"):
    prediction = model.predict(input_data)[0]
    career = label_encoders['Predicted_Career_Path'].inverse_transform([prediction])[0]
    st.success(f"🎯 Your Ideal Career Path: **{career}**")
