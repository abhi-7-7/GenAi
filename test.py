# test.py
# Run this to train all models and get the full evaluation report.
# Usage: python test.py

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is not installed. Run: pip install pandas")
    sys.exit(1)

from src.train import train_models
from src.evaluate import (
    evaluate_model,
    print_metrics,
    print_comparison_table,
    print_classification_report,
)

# ---------------------------------
# Load Data
# ---------------------------------
DATA_PATH = "data/processed/clean_telco_churn.csv"

print(f"\n📂 Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"   Shape     : {df.shape}")
print(f"   Churn Rate: {df['Churn Value'].mean():.3f} ({df['Churn Value'].mean()*100:.1f}%)")

# ---------------------------------
# Train All Models
# ---------------------------------
print("\n🚀 Training all models...\n")
results = train_models(df)

X_test = results["X_test"]
y_test = results["y_test"]

# ---------------------------------
# Evaluate Every Model Individually
# ---------------------------------
print("\n📊 Per-Model Evaluation Reports:")

all_metrics = {}
for name, model in results["models"].items():
    metrics = evaluate_model(model, X_test, y_test, model_name=name)
    print_metrics(metrics)
    all_metrics[name] = metrics

# ---------------------------------
# Comparison Table (copy this into your LaTeX report)
# ---------------------------------
print_comparison_table(all_metrics)

# ---------------------------------
# Full Classification Report for Best Model
# ---------------------------------
best_name  = results["best_model_name"]
best_model = results["best_model"]

print_classification_report(best_model, X_test, y_test, model_name=best_name)

# ---------------------------------
# Summary for LaTeX / README
# ---------------------------------
print("\n" + "="*52)
print("  COPY THESE INTO YOUR LaTeX RESULTS TABLE")
print("="*52)

header = f"{'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}"
print(header)
print("-" * len(header))

for name in ["Logistic Regression", "Decision Tree", "Random Forest", best_name]:
    if name in all_metrics:
        m = all_metrics[name]
        tag = " <- BEST" if name == best_name else ""
        print(
            f"{name:<22} {m['Accuracy']:>6} {m['Precision']:>6} "
            f"{m['Recall']:>6} {m['F1 Score']:>6} {m['AUC-ROC']:>6}{tag}"
        )

print("\n✅ Done. Paste the table above into your LaTeX report.\n")