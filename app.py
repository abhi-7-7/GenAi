# import os
# import streamlit as st
# import pandas as pd
# from src.train import train_models
# from src.utils import load_model

# st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# st.title("📊 Customer Churn Prediction System")
# st.write("Enter the customer details below to get a prediction.")

# @st.cache_resource
# def load_churn_model():
#     model_path = "models/best_model.pkl"
#     if os.path.exists(model_path):
#         return load_model(model_path)
#     else:
#         df = pd.read_csv("data/processed/clean_telco_churn.csv")
#         results = train_models(df)
#         return results["best_model"]

# model = load_churn_model()

# with st.form("prediction_form"):
#     col1, col2, col3 = st.columns(3)

#     with col1:
#         gender = st.selectbox("Gender", ["Male", "Female"])
#         senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
#         partner = st.selectbox("Partner", ["No", "Yes"])
#         dependents = st.selectbox("Dependents", ["No", "Yes"])
#         tenure = st.number_input("Tenure Months", min_value=0, max_value=100, value=1)
#         phone_service = st.selectbox("Phone Service", ["Yes", "No"])

#     with col2:
#         multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
#         internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
#         online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
#         online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
#         device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
#         tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

#     with col3:
#         streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
#         streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
#         contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
#         paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
#         payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
#         monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
#         total_charges = st.number_input("Total Charges", min_value=0.0, value=50.0)

#     submitted = st.form_submit_button("Predict Churn")

#     if submitted:
#         input_data = pd.DataFrame({
#             "Gender": [gender],
#             "Senior Citizen": [senior_citizen],
#             "Partner": [partner],
#             "Dependents": [dependents],
#             "Tenure Months": [int(tenure)],
#             "Phone Service": [phone_service],
#             "Multiple Lines": [multiple_lines],
#             "Internet Service": [internet_service],
#             "Online Security": [online_security],
#             "Online Backup": [online_backup],
#             "Device Protection": [device_protection],
#             "Tech Support": [tech_support],
#             "Streaming TV": [streaming_tv],
#             "Streaming Movies": [streaming_movies],
#             "Contract": [contract],
#             "Paperless Billing": [paperless_billing],
#             "Payment Method": [payment_method],
#             "Monthly Charges": [float(monthly_charges)],
#             "Total Charges": [str(total_charges)]
#         })

#         # Get expected columns from the training set, minus target and leakage
#         df_cols = pd.read_csv("data/processed/clean_telco_churn.csv", nrows=0)
#         expected_cols = [c for c in df_cols.columns if c not in ["Churn Label", "Churn Value"]]
        
#         input_data = input_data[expected_cols]

#         prob = model.predict_proba(input_data)[0][1]
#         pred = model.predict(input_data)[0]

#         st.markdown("---")
#         if pred == 1:
#             st.error(f"⚠️ **High Risk of Churn!** (Probability: {round(prob * 100, 2)}%)")
#         else:
#             st.success(f"✅ **Low Risk of Churn.** (Probability: {round(prob * 100, 2)}%)")














import os
import streamlit as st
import pandas as pd
from src.train import train_models
from src.preprocess import fix_total_charges
from src.utils import load_model
from src.rag import build_vector_store
from src.agent import run_churn_agent

st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #f5f7fb;
    color: #1f2937;
}

section[data-testid="stSidebar"] {
    background-color: #fbfcfe;
    border-right: 1px solid #e5e7eb;
}

section[data-testid="stSidebar"] * {
    color: #111827 !important;
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stMarkdown p {
    color: #111827 !important;
}

section[data-testid="stSidebar"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea {
    background-color: #ffffff !important;
    color: #111827 !important;
    border-color: #d1d5db !important;
}

section[data-testid="stSidebar"] [data-baseweb="select"] svg {
    fill: #6b7280 !important;
}

section[data-testid="stSidebar"] .stButton > button {
    background: #4f46e5 !important;
    color: #ffffff !important;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: #4338ca !important;
}

.sidebar-brand {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: #111827 !important;
    margin-bottom: 2px;
}

.sidebar-caption {
    font-size: 0.78rem;
    color: #6b7280 !important;
    margin-bottom: 1.2rem;
}

.sidebar-section {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6b7280 !important;
    margin: 1.2rem 0 0.5rem 0;
}

.page-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #111827;
    margin-bottom: 0;
}

.page-subtitle {
    font-size: 0.95rem;
    color: #6b7280;
    margin-bottom: 1.5rem;
}

.card {
    background: #ffffff;
    border-radius: 14px;
    padding: 20px 24px;
    border: 1px solid #e5e7eb;
    margin-bottom: 16px;
}

.card-title {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 6px;
}

.verdict-high {
    background: #fff7f7;
    border: 1.5px solid #fca5a5;
    border-radius: 14px;
    padding: 18px 24px;
    text-align: center;
}

.verdict-low {
    background: #f7fdf8;
    border: 1.5px solid #86efac;
    border-radius: 14px;
    padding: 18px 24px;
    text-align: center;
}

.verdict-title-high {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #b91c1c;
}

.verdict-title-low {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #166534;
}

.verdict-sub {
    font-size: 0.85rem;
    color: #6b7280;
    margin-top: 2px;
}

.pipeline-step {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid #eef2f7;
    font-size: 0.88rem;
    color: #374151;
}

.step-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #6366f1;
    flex-shrink: 0;
}

.step-done {
    background: #10b981;
}

.info-pill {
    display: inline-block;
    background: #eef2ff;
    color: #3730a3;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.78rem;
    margin: 2px;
}

div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 14px 18px;
}

div[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    color: #6b7280 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

div[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    color: #111827 !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: #ffffff;
    border-radius: 10px;
    border: 1px solid #e5e7eb;
    padding: 4px;
    gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-size: 0.85rem;
    color: #6b7280;
}

.stTabs [aria-selected="true"] {
    background: #eef2ff !important;
    color: #3730a3 !important;
    font-weight: 600;
}

.stButton > button {
    background: #4f46e5;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 0.9rem;
    width: 100%;
    transition: background 0.2s;
}

.stButton > button:hover {
    background: #4338ca;
    color: white;
}

.stInfo, .stSuccess, .stWarning {
    border-radius: 10px;
    font-size: 0.9rem;
}

.landing-info {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 24px;
    height: 100%;
}

.landing-info h4 {
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 12px;
}

.landing-info p, .landing-info li {
    font-size: 0.88rem;
    color: #374151;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# Load Resources
# ---------------------------------
@st.cache_resource
def load_churn_model():
    model_path = "models/best_model.pkl"
    if os.path.exists(model_path):
        return load_model(model_path)
    df = pd.read_csv("data/processed/clean_telco_churn.csv")
    results = train_models(df)
    return results["best_model"]

@st.cache_resource
def load_vector_store():
    df = pd.read_csv("data/processed/clean_telco_churn.csv")
    df = fix_total_charges(df)
    return build_vector_store(df)

model = load_churn_model()
vector_store = load_vector_store()

# ---------------------------------
# Sidebar
# ---------------------------------
with st.sidebar:
    st.markdown('<p class="sidebar-brand">🛡️ ChurnGuard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-caption">AI-Powered Retention Intelligence</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<p class="sidebar-section">👤 Demographics</p>', unsafe_allow_html=True)
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

    st.markdown('<p class="sidebar-section">📱 Services</p>', unsafe_allow_html=True)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    st.markdown('<p class="sidebar-section">💳 Billing</p>', unsafe_allow_html=True)
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=65.0, step=5.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=float(tenure * 65), step=10.0)

    st.markdown("---")
    submitted = st.button("🔍 Analyse Customer", type="primary")

# ---------------------------------
# Main Content
# ---------------------------------
st.markdown('<h1 class="page-title">Customer Churn Intelligence</h1>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">Enter customer details in the sidebar and run the AI agent to get churn prediction, explanation, and retention strategies.</p>', unsafe_allow_html=True)

if not submitted:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="landing-info">
            <h4>🤖 Agent Pipeline</h4>
            <div class="pipeline-step"><div class="step-dot"></div> Predict — XGBoost ML Model</div>
            <div class="pipeline-step"><div class="step-dot"></div> Summarize — Profile to Text</div>
            <div class="pipeline-step"><div class="step-dot"></div> Retrieve — FAISS RAG Lookup</div>
            <div class="pipeline-step"><div class="step-dot"></div> Explain — LLM Grounded Analysis</div>
            <div class="pipeline-step" style="border:none"><div class="step-dot"></div> Recommend — Retention Strategy</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="landing-info">
            <h4>🧠 AI Stack</h4>
            <p>
            <span class="info-pill">LangGraph</span>
            <span class="info-pill">Groq LLM</span>
            <span class="info-pill">FAISS</span>
            <span class="info-pill">MiniLM Embeddings</span>
            <span class="info-pill">XGBoost</span>
            <span class="info-pill">Scikit-learn</span>
            <span class="info-pill">Streamlit</span>
            </p>
            <br>
            <p>LLM: <b>Llama 3.3 70B</b> via Groq<br>
            Orchestration: <b>LangGraph StateGraph</b><br>
            Vector Store: <b>FAISS + sentence embeddings</b></p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="landing-info">
            <h4>📊 Dataset</h4>
            <p>
            Source: <b>Telco Customer Churn</b><br>
            Customers: <b>7,043 records</b><br>
            Churn Rate: <b>26.5%</b><br>
            Features: <b>19 input features</b><br>
            Task: <b>Binary Classification</b><br>
            Model Selection: <b>F1 Score</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

else:
    try:
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
            "Total Charges": [float(total_charges)]
        })
        input_data = fix_total_charges(input_data)

        with st.spinner("Running agent pipeline..."):
            result = run_churn_agent(model, vector_store, input_data)

        # --- Verdict Row ---
        col_v, col_p, col_t, col_c = st.columns([1.4, 1, 1, 1])

        with col_v:
            if result["prediction"] == 1:
                st.markdown(f"""
                <div class="verdict-high">
                    <div class="verdict-title-high">⚠️ High Churn Risk</div>
                    <div class="verdict-sub">This customer is likely to leave</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-low">
                    <div class="verdict-title-low">✅ Low Churn Risk</div>
                    <div class="verdict-sub">This customer is likely to stay</div>
                </div>""", unsafe_allow_html=True)

        with col_p:
            st.metric("Churn Probability", f"{result['probability']}%")
        with col_t:
            st.metric("Tenure", f"{tenure} months")
        with col_c:
            st.metric("Monthly Charges", f"${monthly_charges}")

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Tabs ---
        tab1, tab2, tab3, tab4 = st.tabs([
            "🧠 AI Explanation",
            "💡 Retention Strategy",
            "📂 Similar Cases",
            "🔗 Pipeline Trace"
        ])

        with tab1:
            st.markdown("#### Why is this customer at risk?")
            st.info(result["explanation"])
            st.markdown("#### Customer Profile")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"- **Contract**: {contract}")
                st.markdown(f"- **Internet**: {internet_service}")
                st.markdown(f"- **Payment**: {payment_method}")
                st.markdown(f"- **Online Security**: {online_security}")
                st.markdown(f"- **Tech Support**: {tech_support}")
            with c2:
                st.markdown(f"- **Tenure**: {tenure} months")
                st.markdown(f"- **Senior Citizen**: {senior_citizen}")
                st.markdown(f"- **Partner**: {partner}")
                st.markdown(f"- **Dependents**: {dependents}")
                st.markdown(f"- **Paperless Billing**: {paperless_billing}")

        with tab2:
            st.markdown("#### Personalised Retention Recommendations")
            st.success(result["recommendation"])

        with tab3:
            st.markdown("#### Top 3 Similar Churned Customers (Retrieved via FAISS)")
            if result["similar_cases"]:
                for i, case in enumerate(result["similar_cases"].split("\n\n"), 1):
                    if case.strip():
                        with st.expander(f"Case {i}"):
                            st.write(case.strip())
            else:
                st.info("No similar cases — customer predicted as low risk.")

        with tab4:
            st.markdown("#### LangGraph Agent Execution Trace")
            steps = [
                ("Predict", f"XGBoost predicted {'Churn' if result['prediction']==1 else 'No Churn'} — probability {result['probability']}%"),
                ("Summarize", "Customer features converted to natural language profile"),
                ("Retrieve", "FAISS vector store queried — top 3 similar churned customers retrieved"),
                ("Explain", "Llama 3.3 70B generated churn explanation grounded in retrieved cases"),
                ("Recommend", "Llama 3.3 70B generated 3 personalised retention strategies"),
            ]
            for name, detail in steps:
                st.markdown(f"""
                <div class="pipeline-step">
                    <div class="step-dot step-done"></div>
                    <span><b>{name} Node</b> — {detail}</span>
                </div>""", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Agent pipeline failed: {str(e)}")
        st.info("Check your GROQ_API_KEY in the .env file and try again.")