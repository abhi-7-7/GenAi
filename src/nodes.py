# src/nodes.py

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    from langchain.schema import HumanMessage, SystemMessage

load_dotenv()

# ---------------------------------
# Initialize LLM
# ---------------------------------
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0.3
)


def predict_node(state: dict) -> dict:
    """
    Node 1: Runs the ML model on input customer data.
    Adds prediction and probability to state.
    """
    model = state["model"]
    input_data = state["input_data"]

    prob = model.predict_proba(input_data)[0][1]
    pred = model.predict(input_data)[0]

    state["prediction"] = int(pred)
    state["probability"] = round(float(prob) * 100, 2)

    return state


def summarize_customer_node(state: dict) -> dict:
    """
    Node 2: Converts input DataFrame into a readable customer summary string.
    Used for RAG retrieval and LLM context.
    """
    row = state["input_data"].iloc[0]

    summary = (
        f"Contract: {row['Contract']}. "
        f"Internet Service: {row['Internet Service']}. "
        f"Tenure: {row['Tenure Months']} months. "
        f"Monthly Charges: ${row['Monthly Charges']}. "
        f"Total Charges: ${row['Total Charges']}. "
        f"Payment Method: {row['Payment Method']}. "
        f"Online Security: {row['Online Security']}. "
        f"Tech Support: {row['Tech Support']}. "
        f"Senior Citizen: {row['Senior Citizen']}. "
        f"Partner: {row['Partner']}. "
        f"Dependents: {row['Dependents']}."
    )

    state["customer_summary"] = summary
    return state


def retrieve_similar_cases_node(state: dict) -> dict:
    """
    Node 3: Retrieves similar churned customer cases from FAISS vector store.
    Only runs if customer is predicted as churn risk.
    """
    if state["prediction"] == 0:
        state["similar_cases"] = ""
        return state

    vector_store = state["vector_store"]
    customer_summary = state["customer_summary"]

    from src.rag import get_similar_churn_cases
    similar_cases = get_similar_churn_cases(vector_store, customer_summary, k=3)

    state["similar_cases"] = similar_cases
    return state


def explain_node(state: dict) -> dict:
    """
    Node 4: Uses LLM to explain why the customer is at churn risk,
    grounded in similar past cases retrieved via RAG.
    """
    if state["prediction"] == 0:
        state["explanation"] = "This customer has a low risk of churning based on their profile."
        return state

    customer_summary = state["customer_summary"]
    similar_cases = state["similar_cases"]
    probability = state["probability"]

    system_prompt = (
        "You are a telecom customer retention analyst. "
        "Your job is to explain why a customer is at risk of churning "
        "based on their profile and similar past churn cases. "
        "Be concise, specific, and use the similar cases as evidence. "
        "Write in 3-4 sentences max."
    )

    user_prompt = (
        f"Customer Profile: {customer_summary}\n\n"
        f"Churn Probability: {probability}%\n\n"
        f"Similar Past Churn Cases:\n{similar_cases}\n\n"
        "Explain why this customer is at risk of churning."
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    state["explanation"] = response.content
    return state


def recommend_node(state: dict) -> dict:
    """
    Node 5: Uses LLM to suggest personalized retention strategies
    based on the customer profile and churn explanation.
    """
    if state["prediction"] == 0:
        state["recommendation"] = "No immediate retention action needed. Continue standard engagement."
        return state

    customer_summary = state["customer_summary"]
    explanation = state["explanation"]

    system_prompt = (
        "You are a telecom customer retention specialist. "
        "Based on the customer profile and churn risk explanation, "
        "suggest 3 specific, actionable retention strategies. "
        "Format as a numbered list. Be practical and personalized."
    )

    user_prompt = (
        f"Customer Profile: {customer_summary}\n\n"
        f"Churn Risk Explanation: {explanation}\n\n"
        "Suggest 3 retention strategies for this customer."
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    state["recommendation"] = response.content
    return state