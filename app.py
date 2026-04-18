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
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none; }
.stApp { background: #f8f9fc; }
.block-container { padding: 2rem 3rem; max-width: 1200px; }

/* Header */
.app-header {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 2rem; padding-bottom: 1.5rem;
    border-bottom: 1px solid #e5e7eb;
}
.app-logo {
    width: 40px; height: 40px; background: #4f46e5;
    border-radius: 10px; display: flex;
    align-items: center; justify-content: center; font-size: 1.2rem;
}
.app-name { font-size: 1.2rem; font-weight: 700; color: #111827; }
.app-tag { font-size: 0.75rem; color: #6b7280; margin-top: 1px; }

/* Section labels */
.section-label {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #9ca3af; margin: 1.5rem 0 0.75rem 0;
}

/* Cards */
.info-card {
    background: #ffffff; border: 1px solid #e5e7eb;
    border-radius: 12px; padding: 20px;
}
.info-card-title {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: #9ca3af; margin-bottom: 12px;
}

/* Verdict */
.verdict-high {
    background: #fef2f2; border: 1.5px solid #fca5a5;
    border-radius: 12px; padding: 16px 20px;
}
.verdict-low {
    background: #f0fdf4; border: 1.5px solid #86efac;
    border-radius: 12px; padding: 16px 20px;
}
.verdict-label {
    font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 4px;
}
.verdict-label-high { color: #ef4444; }
.verdict-label-low { color: #22c55e; }
.verdict-value { font-size: 1.3rem; font-weight: 700; color: #111827; }
.verdict-sub { font-size: 0.82rem; color: #6b7280; margin-top: 2px; }

/* Metrics */
div[data-testid="stMetric"] {
    background: #ffffff !important; border: 1px solid #e5e7eb !important;
    border-radius: 12px !important; padding: 16px 20px !important;
}
div[data-testid="stMetricLabel"] p {
    font-size: 0.72rem !important; font-weight: 600 !important;
    letter-spacing: 0.08em !important; text-transform: uppercase !important;
    color: #9ca3af !important;
}
div[data-testid="stMetricValue"] {
    font-size: 1.5rem !important; font-weight: 700 !important; color: #111827 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff; border: 1px solid #e5e7eb;
    border-radius: 10px; padding: 4px; gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px; font-size: 0.85rem;
    font-weight: 500; color: #374151; padding: 6px 14px;
}
.stTabs [aria-selected="true"] {
    background: #eef2ff !important; color: #4f46e5 !important; font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-panel"] { color: #111827; }

/* Button */
div[data-testid="stButton"] > button {
    background: #4f46e5 !important; color: #ffffff !important;
    border: none !important; border-radius: 10px !important;
    padding: 0.65rem 2rem !important; font-weight: 600 !important;
    font-size: 0.95rem !important; transition: background 0.15s !important;
}
div[data-testid="stButton"] > button:hover { background: #4338ca !important; }

/* Pills */
.pill {
    display: inline-block; background: #eef2ff; color: #4338ca;
    border-radius: 20px; padding: 3px 10px; font-size: 0.75rem;
    font-weight: 500; margin: 2px;
}

/* Pipeline steps */
.pipe-step {
    display: flex; align-items: center; gap: 10px;
    padding: 7px 0; border-bottom: 1px solid #f3f4f6;
    font-size: 0.85rem; color: #374151;
}
.pipe-step:last-child { border-bottom: none; }
.dot { width: 7px; height: 7px; border-radius: 50%; background: #6366f1; flex-shrink: 0; }
.dot-green { background: #10b981; }

/* ALL form inputs - force visible text */
div[data-baseweb="select"] > div {
    border-radius: 8px !important; border-color: #e5e7eb !important;
    background-color: #ffffff !important; color: #111827 !important;
}
div[data-baseweb="select"] span,
div[data-baseweb="select"] div,
div[data-baseweb="select"] p {
    color: #111827 !important;
}
div[data-baseweb="select"] svg { fill: #6b7280 !important; }
[data-baseweb="popover"] li,
[data-baseweb="popover"] div,
[data-baseweb="popover"] span {
    color: #111827 !important; background-color: #ffffff !important;
}
[data-baseweb="menu"] { background-color: #ffffff !important; }
[data-baseweb="menu"] li:hover { background-color: #f3f4f6 !important; }

input[type="number"], input[type="text"] {
    color: #111827 !important; background-color: #ffffff !important;
    border-color: #e5e7eb !important; border-radius: 8px !important;
}

/* Slider value visible */
div[data-testid="stSlider"] p { color: #111827 !important; }
div[data-testid="stSlider"] span { color: #111827 !important; }

/* Widget labels */
label[data-testid="stWidgetLabel"] p {
    font-size: 0.82rem !important; font-weight: 500 !important; color: #374151 !important;
}

/* Info / success boxes */
div[data-testid="stAlert"] {
    border-radius: 10px !important; color: #111827 !important;
}
div[data-testid="stAlert"] p,
div[data-testid="stAlert"] div {
    color: #111827 !important; font-size: 0.88rem !important; line-height: 1.7 !important;
}

/* Expander */
details summary p { color: #111827 !important; font-weight: 500 !important; }
details { background: #ffffff !important; border: 1px solid #e5e7eb !important; border-radius: 10px !important; }
details div { color: #111827 !important; font-size: 0.88rem !important; line-height: 1.7 !important; }

/* Markdown text always dark */
.stMarkdown p, .stMarkdown li, .stMarkdown h4 { color: #111827 !important; }

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
    return train_models(df)["best_model"]

@st.cache_resource
def load_vector_store():
    df = pd.read_csv("data/processed/clean_telco_churn.csv")
    return build_vector_store(fix_total_charges(df))

model = load_churn_model()
vector_store = load_vector_store()

# ---------------------------------
# Header
# ---------------------------------
st.markdown("""
<div class="app-header">
    <div class="app-logo">🛡️</div>
    <div>
        <div class="app-name">ChurnGuard AI</div>
        <div class="app-tag">AI-Powered Customer Retention Intelligence</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------
# Input Form
# ---------------------------------
with st.expander("📋 Customer Details", expanded=True):
    st.markdown('<p class="section-label">👤 Demographics</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: gender = st.selectbox("Gender", ["Male", "Female"])
    with c2: senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    with c3: partner = st.selectbox("Partner", ["No", "Yes"])
    with c4: dependents = st.selectbox("Dependents", ["No", "Yes"])

    st.markdown('<p class="section-label">📱 Services</p>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    with c2: multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    with c3: internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    with c4: online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    with c5: online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

    c1, c2, c3, c4 = st.columns(4)
    with c1: device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    with c2: tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    with c3: streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    with c4: streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    st.markdown('<p class="section-label">💳 Billing</p>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: tenure = st.slider("Tenure (Months)", 0, 72, 12)
    with c2: contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    with c3: paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    with c4: payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    with c5:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=65.0, step=5.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=float(tenure * 65), step=10.0)

    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([3, 2, 3])
    with btn_col:
        submitted = st.button("🔍 Analyse Customer", use_container_width=True)

# ---------------------------------
# Landing Info
# ---------------------------------
if not submitted:
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="info-card">
            <div class="info-card-title">🤖 Agent Pipeline</div>
            <div class="pipe-step"><div class="dot"></div>Predict — XGBoost ML Model</div>
            <div class="pipe-step"><div class="dot"></div>Summarize — Profile to Text</div>
            <div class="pipe-step"><div class="dot"></div>Retrieve — FAISS RAG Lookup</div>
            <div class="pipe-step"><div class="dot"></div>Explain — LLM Grounded Analysis</div>
            <div class="pipe-step"><div class="dot"></div>Recommend — Retention Strategy</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-card">
            <div class="info-card-title">🧠 AI Stack</div>
            <span class="pill">LangGraph</span>
            <span class="pill">Groq LLM</span>
            <span class="pill">FAISS</span>
            <span class="pill">MiniLM</span>
            <span class="pill">XGBoost</span>
            <span class="pill">Streamlit</span>
            <br><br>
            <p style="font-size:0.83rem;color:#374151;line-height:1.6">
            LLM: <b>Llama 3.3 70B</b> via Groq<br>
            Orchestration: <b>LangGraph StateGraph</b><br>
            Vector Store: <b>FAISS + MiniLM embeddings</b>
            </p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="info-card">
            <div class="info-card-title">📊 Dataset</div>
            <p style="font-size:0.83rem;color:#374151;line-height:1.9">
            Source: <b>Telco Customer Churn</b><br>
            Customers: <b>7,043 records</b><br>
            Churn Rate: <b>26.5%</b><br>
            Features: <b>19 input features</b><br>
            Task: <b>Binary Classification</b><br>
            Model Selection: <b>F1 Score</b>
            </p>
        </div>""", unsafe_allow_html=True)

# ---------------------------------
# Results
# ---------------------------------
else:
    try:
        input_data = pd.DataFrame({
            "Gender": [gender], "Senior Citizen": [senior_citizen],
            "Partner": [partner], "Dependents": [dependents],
            "Tenure Months": [int(tenure)], "Phone Service": [phone_service],
            "Multiple Lines": [multiple_lines], "Internet Service": [internet_service],
            "Online Security": [online_security], "Online Backup": [online_backup],
            "Device Protection": [device_protection], "Tech Support": [tech_support],
            "Streaming TV": [streaming_tv], "Streaming Movies": [streaming_movies],
            "Contract": [contract], "Paperless Billing": [paperless_billing],
            "Payment Method": [payment_method],
            "Monthly Charges": [float(monthly_charges)],
            "Total Charges": [float(total_charges)]
        })
        input_data = fix_total_charges(input_data)

        with st.spinner("Running agentic pipeline..."):
            result = run_churn_agent(model, vector_store, input_data)

        st.markdown("<br>", unsafe_allow_html=True)

        # Metrics row
        col_v, col_p, col_t, col_c = st.columns([1.4, 1, 1, 1])
        with col_v:
            if result["prediction"] == 1:
                st.markdown("""
                <div class="verdict-high">
                    <div class="verdict-label verdict-label-high">Churn Risk</div>
                    <div class="verdict-value">⚠️ High Risk</div>
                    <div class="verdict-sub">Customer likely to leave</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="verdict-low">
                    <div class="verdict-label verdict-label-low">Churn Risk</div>
                    <div class="verdict-value">✅ Low Risk</div>
                    <div class="verdict-sub">Customer likely to stay</div>
                </div>""", unsafe_allow_html=True)
        with col_p: st.metric("Churn Probability", f"{result['probability']}%")
        with col_t: st.metric("Tenure", f"{tenure} months")
        with col_c: st.metric("Monthly Charges", f"${monthly_charges}")

        st.markdown("<br>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs([
            "🧠 AI Explanation", "💡 Retention Strategy",
            "📂 Similar Cases", "🔗 Pipeline Trace"
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
            st.markdown("#### Top 3 Similar Churned Customers (FAISS Retrieval)")
            if result["similar_cases"]:
                cases = result["similar_cases"].split("\n\n")
                for i, case in enumerate(cases, 1):
                    if case.strip():
                        with st.expander(f"📁 Similar Case {i}", expanded=False):
                            # Parse and display as clean key-value
                            lines = case.replace("Similar Case " + str(i) + ": ", "").split(". ")
                            for line in lines:
                                if line.strip():
                                    st.markdown(f"- {line.strip()}.")
            else:
                st.info("No similar cases — customer predicted as low risk.")

        with tab4:
            st.markdown("#### LangGraph Agent Execution Trace")
            steps = [
                ("Predict", f"XGBoost → {'Churn' if result['prediction']==1 else 'No Churn'} at {result['probability']}% probability"),
                ("Summarize", "Customer features converted to natural language profile"),
                ("Retrieve", "FAISS queried — top 3 similar churned customers retrieved"),
                ("Explain", "Llama 3.3 70B generated explanation grounded in retrieved cases"),
                ("Recommend", "Llama 3.3 70B generated 3 personalised retention strategies"),
            ]
            for name, detail in steps:
                st.markdown(f"""
                <div class="pipe-step">
                    <div class="dot dot-green"></div>
                    <span><b>{name} Node</b> — {detail}</span>
                </div>""", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Agent pipeline failed: {str(e)}")
        st.info("Check your GROQ_API_KEY in the .env file and try again.")