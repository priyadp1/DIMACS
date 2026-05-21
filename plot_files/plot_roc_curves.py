import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent

BASEDIR = current
results_dir = BASEDIR / "TGB_Variables_Feature_Importance"

parameters = [
    "nest40_depth1", "nest40_depth2", "nest40_depth3",
    "nest100_depth1", "nest100_depth2", "nest100_depth3",
    "nest200_depth1", "nest200_depth2", "nest200_depth3",
]
datasets = ["bike", "compas", "breast_cancer", "spambase",
            "creditcard_fraud", "creditcard_fraud_smote",
            "diabetes", "diabetes_smote"]

COLORS = ["tab:blue", "tab:orange", "tab:green"]

for dataset in datasets:
    for param in parameters:
        base = results_dir / dataset / param
        fi_path = base / "binary_variable_counts.csv"
        if not fi_path.exists():
            continue

        fi_df = pd.read_csv(fi_path).sort_values("importance", ascending=False)
        top3 = fi_df["binary_variable"].head(3).tolist()

        splits = {}
        for split in ("train", "test"):
            x_path = base / f"X_{split}_guessed.csv"
            y_path = base / f"y_{split}.csv"
            if not x_path.exists() or not y_path.exists():
                continue
            X = pd.read_csv(x_path)
            y = pd.read_csv(y_path).iloc[:, 0].values
            if len(np.unique(y)) != 2:
                continue
            splits[split] = (X, y)

        if not splits:
            continue

        fig, axes = plt.subplots(1, len(splits), figsize=(7 * len(splits), 6), squeeze=False)

        for ax, (split, (X, y)) in zip(axes[0], splits.items()):
            for feat, color in zip(top3, COLORS):
                if feat not in X.columns:
                    continue
                fpr, tpr, _ = roc_curve(y, X[feat].values)
                roc_auc = auc(fpr, tpr)
                # Flip if AUC < 0.5 (binary feature direction)
                if roc_auc < 0.5:
                    fpr, tpr, _ = roc_curve(y, 1 - X[feat].values)
                    roc_auc = auc(fpr, tpr)
                label = f"{feat}\n(AUC={roc_auc:.3f})"
                ax.plot(fpr, tpr, color=color, lw=2, label=label)

            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.02])
            ax.set_xlabel("False Positive Rate", fontsize=12)
            ax.set_ylabel("True Positive Rate", fontsize=12)
            ax.set_title(f"{dataset} | {param} | {split}", fontsize=13)
            ax.legend(loc="lower right", fontsize=9)

        plt.tight_layout()
        out_path = base / f"{dataset}_{param}_roc_curves.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path.relative_to(BASEDIR)}")
