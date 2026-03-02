import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from src.train import train_models
from src.utils import load_model

st.set_page_config(page_title="Customer Churn Prediction", layout="wide", page_icon="📊")

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fc; }
    .stApp { background-color: #f8f9fc; }
    h1 { color: #004B87; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #004B87;
    }
    .risk-high {
        background: linear-gradient(135deg, #ff4b4b22, #ff4b4b11);
        border: 2px solid #ff4b4b;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
    }
    .risk-low {
        background: linear-gradient(135deg, #21c45422, #21c45411);
        border: 2px solid #21c454;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
    }
    .section-header {
        color: #004B87;
        font-size: 1.2rem;
        font-weight: 700;
        border-bottom: 2px solid #004B87;
        padding-bottom: 6px;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .tip-box {
        background: #e8f4fd;
        border-left: 4px solid #004B87;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        font-size: 0.95rem;
    }
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("<h1>📊 Customer Churn Prediction System</h1>", unsafe_allow_html=True)
st.markdown("Fill in the customer details below and click **Predict Churn** to get an instant risk assessment.")
st.markdown("---")

# ─── Load Model ──────────────────────────────────────────────────────────────
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

# ─── Input Form ──────────────────────────────────────────────────────────────
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='section-header'>👤 Demographics</div>", unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=1)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])

    with col2:
        st.markdown("<div class='section-header'>🌐 Services</div>", unsafe_allow_html=True)
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

    with col3:
        st.markdown("<div class='section-header'>💳 Account & Billing</div>", unsafe_allow_html=True)
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=50.0)

    submitted = st.form_submit_button("🔍 Predict Churn", use_container_width=True)

# ─── Prediction & Results ─────────────────────────────────────────────────────
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

    df_cols = pd.read_csv("data/processed/clean_telco_churn.csv", nrows=0)
    expected_cols = [c for c in df_cols.columns if c not in ["Churn Label", "Churn Value"]]
    input_data = input_data[expected_cols]

    prob = model.predict_proba(input_data)[0][1]
    pred = model.predict(input_data)[0]
    prob_pct = round(prob * 100, 2)

    st.markdown("---")
    st.markdown("## 🧾 Prediction Results")

    # ── Top Result Banner ────────────────────────────────────────────────────
    res_col1, res_col2 = st.columns([1.2, 1])

    with res_col1:
        if pred == 1:
            st.markdown(f"""
            <div class='risk-high'>
                <h2 style='color:#ff4b4b; margin:0;'>⚠️ HIGH CHURN RISK</h2>
                <p style='font-size:3rem; font-weight:900; color:#ff4b4b; margin:8px 0;'>{prob_pct}%</p>
                <p style='color:#555; margin:0;'>This customer is likely to cancel their subscription.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='risk-low'>
                <h2 style='color:#21c454; margin:0;'>✅ LOW CHURN RISK</h2>
                <p style='font-size:3rem; font-weight:900; color:#21c454; margin:8px 0;'>{prob_pct}%</p>
                <p style='color:#555; margin:0;'>This customer is likely to stay with the service.</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Probability Gauge ────────────────────────────────────────────────────
    with res_col2:
        st.markdown("<div class='section-header'>📉 Risk Probability Gauge</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 2.2))
        bar_color = "#ff4b4b" if pred == 1 else "#21c454"
        ax.barh(["Churn Risk"], [prob_pct], color=bar_color, height=0.4)
        ax.barh(["Churn Risk"], [100 - prob_pct], left=[prob_pct], color="#e0e0e0", height=0.4)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Probability (%)", fontsize=10)
        ax.axvline(x=50, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        ax.text(50, 0.35, '50% threshold', ha='center', va='bottom', fontsize=7, color='gray')
        ax.text(prob_pct / 2, 0, f"{prob_pct}%", ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.patch.set_facecolor('#f8f9fc')
        ax.set_facecolor('#f8f9fc')
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # ── Key Risk Factors & Recommendations ───────────────────────────────────
    risk_col, rec_col = st.columns(2)

    with risk_col:
        st.markdown("<div class='section-header'>🔍 Key Risk Factors Detected</div>", unsafe_allow_html=True)

        risk_factors = []
        if contract == "Month-to-month":
            risk_factors.append("Month-to-month contract — no long-term commitment")
        if tenure <= 12:
            risk_factors.append(f"Low tenure ({tenure} months) — still in high-risk window")
        if monthly_charges > 70:
            risk_factors.append(f"High monthly charges (${monthly_charges}) — cost sensitivity risk")
        if tech_support == "No":
            risk_factors.append("No Tech Support — reduces platform stickiness")
        if online_security == "No":
            risk_factors.append("No Online Security — linked to higher churn")
        if internet_service == "Fiber optic" and monthly_charges > 70:
            risk_factors.append("Fiber optic + high bill — common pattern in churned users")
        if payment_method == "Electronic check":
            risk_factors.append("Electronic check payment — statistically higher churn method")

        if risk_factors:
            for r in risk_factors:
                st.markdown(f"<div class='warning-box'>⚠️ {r}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='tip-box'>✅ No major risk factors detected for this customer.</div>", unsafe_allow_html=True)

    with rec_col:
        st.markdown("<div class='section-header'>💡 Recommended Retention Actions</div>", unsafe_allow_html=True)

        recommendations = []
        if contract == "Month-to-month":
            recommendations.append("Offer a discounted long-term contract (1 or 2 year) to lock in commitment")
        if tenure <= 12:
            recommendations.append("Trigger a welcome loyalty program — first-year customers need extra engagement")
        if monthly_charges > 70:
            recommendations.append("Offer a temporary bill discount or a bundle deal to improve perceived value")
        if tech_support == "No":
            recommendations.append("Offer a free 1-month Tech Support trial to increase stickiness")
        if online_security == "No":
            recommendations.append("Highlight Online Security as a safety add-on with a promotional offer")
        if payment_method == "Electronic check":
            recommendations.append("Incentivize switching to auto-pay (bank transfer or credit card) with a small discount")

        if not recommendations:
            recommendations.append("Continue monitoring — this customer appears stable and satisfied")

        for rec in recommendations:
            st.markdown(f"<div class='tip-box'>💡 {rec}</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Summary Stats Cards ──────────────────────────────────────────────────
    st.markdown("<div class='section-header'>📋 Customer Profile Summary</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📅 Tenure", f"{tenure} months")
    c2.metric("💵 Monthly Charges", f"${monthly_charges}")
    c3.metric("📄 Contract Type", contract)
    c4.metric("🌐 Internet Type", internet_service)