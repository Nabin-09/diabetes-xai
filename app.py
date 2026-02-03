import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Explainable Diabetes Prediction",
    layout="wide"
)

# Load model
model = joblib.load("diabetes_model.pkl")

st.title("🩺 Explainable AI for Early Diabetes Detection")
st.write(
    "This application predicts diabetes risk and explains predictions using SHAP."
)


st.header("Patient Clinical Information")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose Level", 0, 300, 120)
    blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)
    skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)

with col2:
    insulin = st.number_input("Insulin Level", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 1, 120, 30)

input_df = pd.DataFrame(
    [[pregnancies, glucose, blood_pressure,
      skin_thickness, insulin, bmi, dpf, age]],
    columns=model.feature_names_in_
)


if st.button("Predict Diabetes"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"High Risk of Diabetes (Probability: {probability:.2f})")
    else:
        st.success(f"Low Risk of Diabetes (Probability: {probability:.2f})")


    st.subheader("🔍 Explainability (SHAP)")

    explainer = shap.Explainer(model, input_df)
    explanation = explainer(input_df)

  
    shap_values = explanation.values[0, :, 1]
    base_value = explanation.base_values[0, 1]

    shap_exp = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=input_df.iloc[0],
        feature_names=input_df.columns
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(shap_exp, show=False)
    st.pyplot(fig)
