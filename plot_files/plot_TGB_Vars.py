import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from pathlib import Path
import os
import pandas as pd

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent

BASEDIR = current
results_dir = BASEDIR / "TGB_Variables_Feature_Importance"
os.makedirs(results_dir, exist_ok=True)

parameters = ["nest40_depth1", "nest40_depth2", "nest40_depth3", "nest100_depth1", "nest100_depth2", "nest100_depth3", "nest200_depth1", "nest200_depth2", "nest200_depth3"]
datasets = ["bike", "compas", "breast_cancer", "spambase"]
for i in datasets:
    for j in parameters:
        base = BASEDIR / "TGB_Variables_Feature_Importance" / i / j
        file_path = base / "binary_variable_counts.csv"
        if not file_path.exists():
            continue
        df = pd.read_csv(file_path)
        feature = df["binary_variable"].values
        importance = df["importance"].values
        auc_str = ""
        x_path = base / "X_train_guessed.csv"
        y_path = base / "y_train.csv"
        if x_path.exists() and y_path.exists():
            X = pd.read_csv(x_path)
            y = pd.read_csv(y_path).iloc[:, 0].values
            if len(np.unique(y)) == 2:
                per_feature_auc = {
                    col: roc_auc_score(y, X[col].values)
                    for col in X.columns
                    if col in df["binary_variable"].values
                }
                if per_feature_auc:
                    mean_auc = np.mean(list(per_feature_auc.values()))
                    auc_str = f"\nMean per-feature AUC: {mean_auc:.4f}"

        plt.figure(figsize=(10, 6))
        plt.bar(feature, importance, color='skyblue')
        plt.xlabel('Binary Variable', fontsize=14)
        plt.ylabel('Importance', fontsize=14)
        plt.title(f'Feature Importance for {i} - {j}{auc_str}', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(base / f"{i}_{j}_feature_importance.png")
        plt.close()
