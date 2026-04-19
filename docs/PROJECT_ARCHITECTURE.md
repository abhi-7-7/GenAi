# Project Architecture Diagram

```text
┌────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE LAYER                        │
│                    Streamlit App: app.py                           │
│       - Customer form (demographics/services/billing)              │
│       - Analyse button triggers full pipeline                      │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                    RESOURCE LOADING (app.py)                       │
│ 1) Model Loader                                                    │
│    - load models/best_model.pkl if present                         │
│    - else train via src/train.py                                   │
│ 2) Vector Store Loader                                             │
│    - read data/processed/clean_telco_churn.csv                     │
│    - fix_total_charges() from src/preprocess.py                    │
│    - build_vector_store() from src/rag.py                          │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                  AGENT ORCHESTRATION LAYER                         │
│                    src/agent.py (LangGraph)                        │
│                  Shared state: ChurnAgentState                     │
│                                                                    │
│   predict_node  → summarize_customer_node → retrieve_similar_cases │
│         │                          │                      │         │
│         └──────────────→ explain_node → recommend_node → END       │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                ┌───────────────┴────────────────┐
                ▼                                ▼
┌──────────────────────────────┐      ┌──────────────────────────────┐
│       MODEL INFERENCE         │      │          RAG LAYER           │
│      src/nodes.py             │      │         src/rag.py           │
│ - model.predict_proba()       │      │ - Build documents from       │
│ - model.predict()             │      │   churned rows only          │
│ - model is sklearn Pipeline   │      │ - Embeddings: MiniLM         │
│   (best selected model)       │      │ - Vector DB: FAISS           │
└──────────────────────────────┘      └──────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                           LLM LAYER                                │
│                     src/nodes.py (ChatGroq)                        │
│ - explain_node: grounded churn explanation                          │
│ - recommend_node: retention strategies                              │
│ - Model: llama-3.3-70b-versatile                                   │
│ - Auth: GROQ_API_KEY (env var)                                     │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                          OUTPUT LAYER                              │
│                    Streamlit Dashboard Tabs                         │
│ - Churn verdict + probability                                      │
│ - AI explanation                                                    │
│ - Similar cases (RAG retrieval)                                    │
│ - Retention recommendations                                        │
│ - Pipeline trace                                                    │
└────────────────────────────────────────────────────────────────────┘
```

## Notes

- The best model is selected by F1 score in `src/train.py` and saved as `models/best_model.pkl`.
- Runtime preprocessing in `app.py` uses `fix_total_charges()`; encoding/scaling is inside the trained sklearn pipeline.
- `retrieve_similar_cases_node()` runs only when churn prediction is positive.
