# src/rag.py

import os
import pandas as pd
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

load_dotenv()


def build_churn_documents(df: pd.DataFrame) -> list[Document]:
    """
    Converts each churned customer row into a LangChain Document.
    Only churned customers (Churn Value == 1) are used as knowledge base.
    """
    churned_df = df[df["Churn Value"] == 1].copy()

    documents = []
    for _, row in churned_df.iterrows():
        content = (
            f"Customer churned. "
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
        documents.append(Document(page_content=content))

    return documents


def build_vector_store(df: pd.DataFrame) -> FAISS:
    """
    Builds and returns a FAISS vector store from churned customer records.
    Uses HuggingFace sentence-transformers (no API key needed for embeddings).
    """
    documents = build_churn_documents(df)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store


def get_similar_churn_cases(vector_store: FAISS, customer_summary: str, k: int = 3) -> str:
    """
    Retrieves k most similar churned customer cases for a given customer summary.

    Args:
        vector_store: FAISS vector store built from churned customers
        customer_summary: string description of the current customer
        k: number of similar cases to retrieve

    Returns:
        A formatted string of similar churn cases for LLM context
    """
    results = vector_store.similarity_search(customer_summary, k=k)

    context = ""
    for i, doc in enumerate(results, 1):
        context += f"Similar Case {i}: {doc.page_content}\n\n"

    return context.strip()