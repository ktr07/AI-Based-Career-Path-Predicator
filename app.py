import streamlit as st
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

# Load label encoders
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
domain_knowledge = st.slider("Domain Knowledge (0-10)", 0, 10, 5)

# Title and subtitle
st.title("🧠 AI Career Path Predictor")
st.markdown("Enter your skills and preferences to get your ideal tech career suggestion!")

# Input fields
programming = st.slider("Programming Skills (0-10)", 0, 10, 5)
math = st.slider("Math/Statistics Skills (0-10)", 0, 10, 5)
communication = st.slider("Communication Skills (0-10)", 0, 10, 5)
creativity = st.slider("Creativity (0-10)", 0, 10, 5)
research = st.slider("Interest in Research (0-10)", 0, 10, 5)
projects = st.slider("Projects Completed", 0, 10, 2)

# Dropdowns for Preference and Stream
preference = st.selectbox("Domain Preference", label_encoders['Preference'].classes_)
stream = st.selectbox("Academic Stream", label_encoders['Academic_Stream'].classes_)

# Encode the categorical inputs
encoded_preference = label_encoders['Preference'].transform([preference])[0]
encoded_stream = label_encoders['Academic_Stream'].transform([stream])[0]

# Combine all inputs
input_data = np.array([[programming, math, communication, creativity, research, projects,
                        domain_knowledge, encoded_preference, encoded_stream]])


# Predict on button click
if st.button("Predict Career Path"):
    prediction = model.predict(input_data)[0]
    career = label_encoders['Predicted_Career_Path'].inverse_transform([prediction])[0]
    st.success(f"🎯 Your Ideal Career Path: **{career}**")
