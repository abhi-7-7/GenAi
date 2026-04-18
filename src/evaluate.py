
# src/evaluate.py

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluates a fitted sklearn Pipeline on held-out test data.

    Computes:
        - Accuracy   : proportion of correct predictions
        - Precision  : of predicted churners, fraction that truly churned
        - Recall     : of actual churners, fraction correctly identified
        - F1 Score   : harmonic mean of precision & recall (primary metric)
        - AUC-ROC    : area under ROC curve — probability ranking quality
        - Confusion Matrix : [[TN, FP], [FN, TP]]

    Args:
        model      : fitted sklearn Pipeline (preprocessor + classifier)
        X_test     : held-out feature DataFrame
        y_test     : held-out binary labels (0 = No Churn, 1 = Churn)
        model_name : string label used in printed output

    Returns:
        dict with all metric values (rounded to 4 d.p.)
    """

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model":            model_name,
        "Accuracy":         round(accuracy_score(y_test, y_pred), 4),
        "Precision":        round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":           round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1 Score":         round(f1_score(y_test, y_pred, zero_division=0), 4),
        "AUC-ROC":          round(roc_auc_score(y_test, y_prob), 4),
        "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    return metrics


def print_metrics(metrics):
    """
    Pretty-prints a metrics dict returned by evaluate_model().

    Args:
        metrics : dict returned by evaluate_model()
    """
    name = metrics.get("Model", "Model")
    cm   = metrics["Confusion Matrix"]

    print(f"\n{'='*52}")
    print(f"  {name}")
    print(f"{'='*52}")
    print(f"  Accuracy   : {metrics['Accuracy']}")
    print(f"  Precision  : {metrics['Precision']}")
    print(f"  Recall     : {metrics['Recall']}")
    print(f"  F1 Score   : {metrics['F1 Score']}")
    print(f"  AUC-ROC    : {metrics['AUC-ROC']}")
    print(f"  Confusion Matrix:")
    print(f"             Predicted")
    print(f"             No Churn   Churn")
    print(f"  Actual No Churn  {cm[0][0]:>5}   {cm[0][1]:>5}")
    print(f"  Actual Churn     {cm[1][0]:>5}   {cm[1][1]:>5}")
    print()


def print_comparison_table(all_metrics):
    """
    Prints a ranked comparison table for all evaluated models.
    Models are sorted by F1 Score descending.

    Args:
        all_metrics : dict of {model_name: metrics_dict}
    """
    print(f"\n{'='*70}")
    print(f"  MODEL COMPARISON — Ranked by F1 Score (Test Set, 20% hold-out)")
    print(f"{'='*70}")
    print(f"  {'Model':<22} {'Accuracy':>9} {'Precision':>10} "
          f"{'Recall':>8} {'F1':>8} {'AUC-ROC':>9}")
    print(f"  {'-'*67}")

    sorted_models = sorted(
        all_metrics.items(),
        key=lambda x: x[1]["F1 Score"],
        reverse=True
    )

    for name, m in sorted_models:
        marker = " ✅" if name == sorted_models[0][0] else "   "
        print(
            f"  {name:<22} {m['Accuracy']:>9} {m['Precision']:>10} "
            f"{m['Recall']:>8} {m['F1 Score']:>8} {m['AUC-ROC']:>9}{marker}"
        )

    print(f"{'='*70}")
    best_name, best_m = sorted_models[0]
    print(f"\n  Best Model : {best_name}")
    print(f"  F1 Score   : {best_m['F1 Score']}")
    print(f"  AUC-ROC    : {best_m['AUC-ROC']}")
    print(f"  Recall     : {best_m['Recall']}  "
          f"← catches {int(best_m['Recall']*100)}% of actual churners\n")


def print_classification_report(model, X_test, y_test, model_name="Best Model"):
    """
    Prints the full sklearn classification report for a model.
    Useful for the LaTeX report and viva.

    Args:
        model      : fitted sklearn Pipeline
        X_test     : held-out features
        y_test     : held-out labels
        model_name : label for printed header
    """
    y_pred = model.predict(X_test)
    print(f"\n--- Classification Report: {model_name} ---")
    print(classification_report(
        y_test, y_pred,
        target_names=["No Churn", "Churn"],
        zero_division=0
    ))