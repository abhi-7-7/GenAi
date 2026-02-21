# src/utils.py

import pandas as pd
import joblib


def load_data(file_path):
    """
    Load dataset from CSV file
    """
    return pd.read_csv(file_path)


def encode_target(df, target_column="Churn"):
    """
    Convert target column Yes/No to 1/0
    """
    df[target_column] = df[target_column].map({"Yes": 1, "No": 0})
    return df


def save_model(model, path):
    """
    Save trained model
    """
    joblib.dump(model, path)


def load_model(path):
    """
    Load saved model
    """
    return joblib.load(path)