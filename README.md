# Customer Churn Prediction Project

## Overview

This project predicts telecom customer churn using a machine learning pipeline plus an AI agent workflow.

The current app includes:
- an XGBoost-based churn model
- a LangGraph agent pipeline
- FAISS-based retrieval of similar churn cases
- Groq LLM explanations and retention recommendations
- a Streamlit dashboard for interactive analysis

## Project Goals

- identify customers likely to churn
- explain churn risk in plain language
- retrieve similar past churn cases for context
- suggest retention actions
- provide a clean Streamlit-based user interface

## Dataset

- **Source**: Telco Customer Churn dataset
- **Raw file**: `data/raw/Telco_customer_churn.xlsx`
- **Processed file**: `data/processed/clean_telco_churn.csv`

## Current Project Structure

```
customer-churn-project/
├── app.py                    # Streamlit app with AI agent flow
├── environment.yml           # Conda environment definition
├── models/
│   └── best_model.pkl        # Saved trained model
├── notebooks/
│   ├── 1_data_exploration.ipynb
│   └── 2_feature_engineering.ipynb
├── src/
│   ├── agent.py              # LangGraph agent workflow
│   ├── nodes.py              # Agent nodes for predict/explain/recommend
│   ├── preprocess.py         # Feature preprocessing helpers
│   ├── rag.py                # FAISS retrieval logic
│   ├── train.py              # Model training and selection
│   ├── evaluate.py           # Model evaluation helpers
│   └── utils.py              # Data/model utility functions
├── data/
│   ├── raw/
│   │   └── Telco_customer_churn.xlsx
│   └── processed/
│       └── clean_telco_churn.csv
└── README.md
```

## Technologies Used

- **Python 3.11**
- **Streamlit** for UI
- **pandas** and **numpy** for data handling
- **scikit-learn** for preprocessing and ML pipelines
- **xgboost** for model training
- **langgraph** for agent orchestration
- **langchain**, **langchain-groq**, **langchain-community** for LLM/RAG integration
- **FAISS** for similarity search
- **sentence-transformers** for embeddings
- **matplotlib** and **seaborn** for visualization
- **openpyxl** for Excel file support

## How It Works

1. user enters customer details in the Streamlit sidebar
2. XGBoost predicts churn risk
3. the agent summarizes the customer profile
4. FAISS retrieves similar churn cases
5. Groq LLM explains the risk and suggests retention actions
6. results are displayed in a clean dashboard

## Installation & Setup

### Option 1: Conda recommended

```bash
git clone https://github.com/abhi-7-7/GenAi.git
cd GenAi/customer-churn-project
conda env create -f environment.yml
conda activate space
```

### Run the app

```bash
python -m streamlit run app.py
```

### Run the test script

```bash
python test_train.py
```

### Open notebooks

```bash
jupyter notebook
```

## Main Files

- [app.py](app.py): Streamlit UI and agent execution
- [src/train.py](src/train.py): trains Logistic Regression, Decision Tree, Random Forest, and XGBoost, then saves the best model
- [src/preprocess.py](src/preprocess.py): prepares features and handles preprocessing
- [src/rag.py](src/rag.py): builds the vector store and retrieves similar churn cases
- [src/agent.py](src/agent.py): connects the full agent flow
- [src/nodes.py](src/nodes.py): prediction, explanation, and recommendation nodes
- [src/utils.py](src/utils.py): helper functions for loading/saving data and models
- [test_train.py](test_train.py): quick validation script for model training

## Output

The app shows:
- churn prediction
- churn probability
- risk explanation
- retention recommendation
- similar customer cases
- pipeline trace

## Notes

- Use the Conda environment `space` for all commands.
- If the shell points to another Python, run:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate space
```

- The project is designed for Python 3.11.
