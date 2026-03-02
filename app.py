import os
import streamlit as st
import pandas as pd
from src.train import train_models
from src.utils import load_model

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("📊 Customer Churn Prediction System")
st.write("Enter the customer details below to get a prediction.")

@st.cache_resource
def load_churn_model():
    model_path = "models/best_model.pkl"
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        df = pd.read_csv("data/processed/clean_telco_churn.csv")
        results = train_models(df)
        return results["best_model"]

model = load_churn_model()

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.number_input("Tenure Months", min_value=0, max_value=100, value=1)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])

    with col2:
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

    with col3:
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
        total_charges = st.number_input("Total Charges", min_value=0.0, value=50.0)

    submitted = st.form_submit_button("Predict Churn")

    if submitted:
        input_data = pd.DataFrame({
            "Gender": [gender],
            "Senior Citizen": [senior_citizen],
            "Partner": [partner],
            "Dependents": [dependents],
            "Tenure Months": [int(tenure)],
            "Phone Service": [phone_service],
            "Multiple Lines": [multiple_lines],
            "Internet Service": [internet_service],
            "Online Security": [online_security],
            "Online Backup": [online_backup],
            "Device Protection": [device_protection],
            "Tech Support": [tech_support],
            "Streaming TV": [streaming_tv],
            "Streaming Movies": [streaming_movies],
            "Contract": [contract],
            "Paperless Billing": [paperless_billing],
            "Payment Method": [payment_method],
            "Monthly Charges": [float(monthly_charges)],
            "Total Charges": [str(total_charges)]
        })

        # Get expected columns from the training set, minus target and leakage
        df_cols = pd.read_csv("data/processed/clean_telco_churn.csv", nrows=0)
        expected_cols = [c for c in df_cols.columns if c not in ["Churn Label", "Churn Value"]]
        
        input_data = input_data[expected_cols]

        prob = model.predict_proba(input_data)[0][1]
        pred = model.predict(input_data)[0]

        st.markdown("---")
        if pred == 1:
            st.error(f"⚠️ **High Risk of Churn!** (Probability: {round(prob * 100, 2)}%)")
        else:
            st.success(f"✅ **Low Risk of Churn.** (Probability: {round(prob * 100, 2)}%)")