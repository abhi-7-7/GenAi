# src/agent.py

from langgraph.graph import StateGraph, END
from typing import TypedDict, Any
import pandas as pd

from src.nodes import (
    predict_node,
    summarize_customer_node,
    retrieve_similar_cases_node,
    explain_node,
    recommend_node
)


# ---------------------------------
# Define State Schema
# ---------------------------------
class ChurnAgentState(TypedDict):
    model: Any                  # trained sklearn pipeline
    vector_store: Any           # FAISS vector store
    input_data: pd.DataFrame    # customer input
    prediction: int             # 0 or 1
    probability: float          # churn probability %
    customer_summary: str       # readable customer description
    similar_cases: str          # retrieved RAG context
    explanation: str            # LLM churn explanation
    recommendation: str         # LLM retention strategies


# ---------------------------------
# Build LangGraph Agent
# ---------------------------------
def build_churn_agent():
    """
    Builds and compiles the LangGraph agentic workflow.

    Flow:
    predict → summarize → retrieve → explain → recommend → END
    """

    graph = StateGraph(ChurnAgentState)

    # Add nodes
    graph.add_node("predict", predict_node)
    graph.add_node("summarize", summarize_customer_node)
    graph.add_node("retrieve", retrieve_similar_cases_node)
    graph.add_node("explain", explain_node)
    graph.add_node("recommend", recommend_node)

    # Define edges (sequential flow)
    graph.set_entry_point("predict")
    graph.add_edge("predict", "summarize")
    graph.add_edge("summarize", "retrieve")
    graph.add_edge("retrieve", "explain")
    graph.add_edge("explain", "recommend")
    graph.add_edge("recommend", END)

    return graph.compile()


# ---------------------------------
# Run Agent
# ---------------------------------
def run_churn_agent(model, vector_store, input_data: pd.DataFrame) -> dict:
    """
    Runs the full churn agent pipeline for a given customer.

    Args:
        model: trained sklearn pipeline (best_model.pkl)
        vector_store: FAISS vector store from churn dataset
        input_data: single-row DataFrame of customer features

    Returns:
        Final state dict with prediction, explanation, and recommendation
    """
    agent = build_churn_agent()

    initial_state = ChurnAgentState(
        model=model,
        vector_store=vector_store,
        input_data=input_data,
        prediction=0,
        probability=0.0,
        customer_summary="",
        similar_cases="",
        explanation="",
        recommendation=""
    )

    result = agent.invoke(initial_state)
    return result