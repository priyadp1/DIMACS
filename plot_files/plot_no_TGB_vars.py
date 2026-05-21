import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from pathlib import Path
import os
import pandas as pd
import re
from io import StringIO

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent

BASEDIR = current

DATASET_CONFIG = {
    "bike":          (BASEDIR / "datasets/Mine/bike.csv",               "cnt_binary"),
    "compas":        (BASEDIR / "datasets/Mine/compas.csv",             "two_year_recid"),
    "breast_cancer": (BASEDIR / "datasets/Mine/breast_cancer_data.csv", "diagnosis"),
    "spambase":      (BASEDIR / "datasets/Mine/spambase.csv",           "class"),
}

algorithms = ["lightgbm", "catboost", "xgboost_binarized"]

output_dir = BASEDIR / "benchmarks_no_TGB_results" / "feature_auc_plots"
os.makedirs(output_dir, exist_ok=True)


def parse_top3_features(filepath):
    """Return list of (feature_name, importance) from a results file's Top 3 Features block."""
    with open(filepath, "r") as f:
        content = f.read()
    idx = content.find("Top 3 Features:")
    if idx == -1:
        return []
    block = content[idx + len("Top 3 Features:"):].strip()
    lines = block.splitlines()
    # lines[0] is the header (contains "Feature" and "Importance")
    # lines[1..3] are the data rows; stop at blank or "Tree Size"
    data_lines = []
    for line in lines[1:]:
        if not line.strip() or line.strip().startswith("Tree Size"):
            break
        data_lines.append(line)
    if not data_lines:
        return []
    table_text = lines[0] + "\n" + "\n".join(data_lines)
    df = pd.read_fwf(StringIO(table_text))
    df.columns = ["Feature", "Importance"]
    return list(zip(df["Feature"].astype(str).str.strip(), df["Importance"]))


def resolve_feature(name, columns):
    """Return the matching column name, trying underscore/space variants."""
    if name in columns:
        return name
    alt = name.replace("_", " ")
    if alt in columns:
        return alt
    alt2 = name.replace(" ", "_")
    if alt2 in columns:
        return alt2
    return None


def compute_feature_auc(feature_col, y):
    """AUC for a single feature used as a raw score; flipped if < 0.5."""
    auc = roc_auc_score(y, feature_col)
    return max(auc, 1 - auc)


for dataset, (csv_path, target_col) in DATASET_CONFIG.items():
    df_data = pd.read_csv(csv_path)
    y = df_data[target_col]
    if y.dtype == object:
        classes = sorted(y.unique())
        y = (y == classes[-1]).astype(int)  # alphabetically last → 1 (e.g. 'M' > 'B')
    else:
        y = y.values

    fig, axes = plt.subplots(1, len(algorithms), figsize=(5 * len(algorithms), 5), sharey=False)
    fig.suptitle(f"Per-Feature ROC AUC  —  {dataset}", fontsize=15, fontweight="bold")

    for ax, algo in zip(axes, algorithms):
        results_file = BASEDIR / "benchmarks_no_TGB_results" / dataset / f"{algo}_results.txt"
        if not results_file.exists():
            ax.set_title(f"{algo}\n(no results)", fontsize=11)
            ax.axis("off")
            continue

        top3 = parse_top3_features(results_file)
        if not top3:
            ax.set_title(f"{algo}\n(parse failed)", fontsize=11)
            ax.axis("off")
            continue

        feature_names, aucs, missing = [], [], []
        for raw_name, _ in top3:
            col = resolve_feature(raw_name, df_data.columns)
            if col is None:
                missing.append(raw_name)
                continue
            auc = compute_feature_auc(df_data[col].values, y)
            feature_names.append(raw_name)
            aucs.append(auc)

        if missing:
            print(f"[WARN] {dataset}/{algo}: could not find columns {missing}")

        colors = ["#4C72B0", "#DD8452", "#55A868"]
        bars = ax.bar(range(len(feature_names)), aucs, color=colors[: len(feature_names)],
                      edgecolor="white", linewidth=0.8)
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0.4, 1.05)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="random")
        ax.set_title(algo, fontsize=11)
        ax.set_ylabel("AUC (per-feature)", fontsize=9)
        for bar, auc in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{auc:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_path = output_dir / f"{dataset}_feature_auc.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")
