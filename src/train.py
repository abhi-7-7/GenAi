from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from src.preprocess import build_preprocessing_pipeline, fix_total_charges
from src.utils import save_model


def get_model_definitions():
    """Returns a fresh dict of unfitted models."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced", max_depth=6, min_samples_leaf=10),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10, min_samples_leaf=5, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=200, random_state=42, scale_pos_weight=2.77, learning_rate=0.05, max_depth=5, eval_metric="logloss", verbosity=0)
    }


def train_models(df, target_column="Churn Value"):
    """
    Trains multiple classification models on the churn dataset.
    Selects the best model based on F1 score (appropriate for imbalanced data).

    Each model gets its own fresh preprocessor to avoid shared state bugs.

    Returns a dict with all models, best model, metrics, and test data.
    """

    # ---------------------------------
    # Validate + fix data
    # ---------------------------------
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    df = fix_total_charges(df)
    leakage_columns = ["Churn Label"]

    X = df.drop(columns=[target_column] + leakage_columns)
    y = df[target_column]

    # ---------------------------------
    # Train-Test Split
    # ---------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------------------------------
    # Train & Evaluate All Models
    # Each model gets its own fresh preprocessor
    # ---------------------------------
    trained_pipelines = {}
    f1_scores = {}

    for name, clf in get_model_definitions().items():
        # Fresh preprocessor for each model — avoids shared state bug
        preprocessor, _, _ = build_preprocessing_pipeline(
            df.drop(columns=leakage_columns),
            target_column
        )

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        score = f1_score(y_test, y_pred)

        trained_pipelines[name] = pipeline
        f1_scores[name] = score
        print(f"{name} — F1: {round(score, 4)}")

    # ---------------------------------
    # Select Best Model by F1
    # ---------------------------------
    best_model_name = max(f1_scores, key=f1_scores.get)
    best_model = trained_pipelines[best_model_name]
    best_f1 = f1_scores[best_model_name]

    print(f"\nBest Model: {best_model_name} (F1: {round(best_f1, 4)})")

    # ---------------------------------
    # Save Best Model
    # ---------------------------------
    save_model(best_model, "models/best_model.pkl")

    return {
        "models": trained_pipelines,
        "f1_scores": f1_scores,
        "best_model": best_model,
        "best_model_name": best_model_name,
        "best_f1": best_f1,
        "X_test": X_test,
        "y_test": y_test
    }









# # src/train.py

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score

# from src.preprocess import build_preprocessing_pipeline
# from src.utils import save_model


# def train_models(df, target_column="Churn Value"):

#     # ---------------------------------
#     # Validate dataset
#     # ---------------------------------
#     if target_column not in df.columns:
#         raise ValueError(
#             f"Target column '{target_column}' not found in dataset."
#         )

#     # ---------------------------------
#     # Remove leakage columns
#     # ---------------------------------
#     leakage_columns = ["Churn Label"]  # This leaks target information

#     # Drop target + leakage
#     X = df.drop(columns=[target_column] + leakage_columns)
#     y = df[target_column]

#     # ---------------------------------
#     # Train-Test Split
#     # ---------------------------------
#     X_train, X_test, y_train, y_test = train_test_split(
#         X,
#         y,
#         test_size=0.2,
#         random_state=42,
#         stratify=y
#     )

#     # ---------------------------------
#     # Build Preprocessing Pipeline
#     # ---------------------------------
#     preprocessor, _, _ = build_preprocessing_pipeline(
#         df.drop(columns=leakage_columns),
#         target_column
#     )

#     # ---------------------------------
#     # Logistic Regression
#     # ---------------------------------
#     logistic_pipeline = Pipeline(steps=[
#         ("preprocessor", preprocessor),
#         ("classifier", LogisticRegression(max_iter=1000))
#     ])

#     logistic_pipeline.fit(X_train, y_train)

#     # ---------------------------------
#     # Decision Tree
#     # ---------------------------------
#     tree_pipeline = Pipeline(steps=[
#         ("preprocessor", preprocessor),
#         ("classifier", DecisionTreeClassifier(random_state=42))
#     ])

#     tree_pipeline.fit(X_train, y_train)

#     # ---------------------------------
#     # Compare Accuracy
#     # ---------------------------------
#     log_accuracy = accuracy_score(y_test, logistic_pipeline.predict(X_test))
#     tree_accuracy = accuracy_score(y_test, tree_pipeline.predict(X_test))

#     if log_accuracy >= tree_accuracy:
#         best_model = logistic_pipeline
#         best_model_name = "Logistic Regression"
#         best_accuracy = log_accuracy
#     else:
#         best_model = tree_pipeline
#         best_model_name = "Decision Tree"
#         best_accuracy = tree_accuracy

#     # ---------------------------------
#     # Save Best Model
#     # ---------------------------------
#     save_model(best_model, "models/best_model.pkl")

#     return {
#         "log_model": logistic_pipeline,
#         "tree_model": tree_pipeline,
#         "best_model": best_model,
#         "best_model_name": best_model_name,
#         "best_accuracy": best_accuracy,
#         "X_test": X_test,
#         "y_test": y_test
#     }

# src/train.py
# src/train.py
# src/train.py
