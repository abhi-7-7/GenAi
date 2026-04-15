# test_train.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import pandas as pd
except ImportError:
    
    print("Error: pandas is not installed. Please install it using: pip install pandas")
    sys.exit(1)

from src.train import train_models

df = pd.read_csv("data/processed/clean_telco_churn.csv")
results = train_models(df)

print("Best Model:", results["best_model_name"])
print("F1 Scores:", results["f1_scores"])