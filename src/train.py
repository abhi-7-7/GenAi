# src/train.py

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from src.preprocess import build_preprocessing_pipeline
from src.utils import save_model


def train_models(df, target_column="Churn Value"):

    # ---------------------------------
    # Validate dataset
    # ---------------------------------
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset."
        )

    # ---------------------------------
    # Remove leakage columns
    # ---------------------------------
    leakage_columns = ["Churn Label"]  # This leaks target information

    # Drop target + leakage
    X = df.drop(columns=[target_column] + leakage_columns)
    y = df[target_column]

    # ---------------------------------
    # Train-Test Split
    # ---------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ---------------------------------
    # Build Preprocessing Pipeline
    # ---------------------------------
    preprocessor, _, _ = build_preprocessing_pipeline(
        df.drop(columns=leakage_columns),
        target_column
    )

    # ---------------------------------
    # Logistic Regression
    # ---------------------------------
    logistic_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    logistic_pipeline.fit(X_train, y_train)

    # ---------------------------------
    # Decision Tree
    # ---------------------------------
    tree_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=42))
    ])

    tree_pipeline.fit(X_train, y_train)

    # ---------------------------------
    # Compare Accuracy
    # ---------------------------------
    log_accuracy = accuracy_score(y_test, logistic_pipeline.predict(X_test))
    tree_accuracy = accuracy_score(y_test, tree_pipeline.predict(X_test))

    if log_accuracy >= tree_accuracy:
        best_model = logistic_pipeline
        best_model_name = "Logistic Regression"
        best_accuracy = log_accuracy
    else:
        best_model = tree_pipeline
        best_model_name = "Decision Tree"
        best_accuracy = tree_accuracy

    # ---------------------------------
    # Save Best Model
    # ---------------------------------
    save_model(best_model, "models/best_model.pkl")

    return {
        "log_model": logistic_pipeline,
        "tree_model": tree_pipeline,
        "best_model": best_model,
        "best_model_name": best_model_name,
        "best_accuracy": best_accuracy,
        "X_test": X_test,
        "y_test": y_test
    }