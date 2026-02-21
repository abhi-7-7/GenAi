# src/preprocess.py

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


def build_preprocessing_pipeline(df, target_column="Churn Value"):

    # Separate features from target
    X = df.drop(columns=[target_column])

    # Identify numeric & categorical columns
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # ----------------------------
    # Numeric Transformer
    # ----------------------------
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    # ----------------------------
    # Categorical Transformer
    # ----------------------------
    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
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