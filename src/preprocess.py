# src/preprocess.py

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def fix_total_charges(df):
    """
    'Total Charges' comes as object type in raw data due to spaces.
    Convert to float and fill missing values with median.
    """
    df = df.copy()
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    median_val = df["Total Charges"].median()
    df["Total Charges"] = df["Total Charges"].fillna(median_val)
    return df


def build_preprocessing_pipeline(df, target_column="Churn Value"):
    """
    Builds a sklearn preprocessing pipeline for the churn dataset.

    Steps:
    - Numeric: Impute missing with median → StandardScaler
    - Categorical: Impute missing with most frequent → OneHotEncoder

    Returns:
        preprocessor: ColumnTransformer pipeline
        numerical_cols: list of numeric feature names
        categorical_cols: list of categorical feature names
    """

    # Fix Total Charges dtype before column detection
    df = fix_total_charges(df)

    # Separate features from target
    X = df.drop(columns=[target_column])

    # Identify numeric & categorical columns
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # ----------------------------
    # Numeric Transformer
    # ----------------------------
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # ----------------------------
    # Categorical Transformer
    # ----------------------------
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # ----------------------------
    # Combine into ColumnTransformer
    # ----------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    return preprocessor, numerical_cols, categorical_cols