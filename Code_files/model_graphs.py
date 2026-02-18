import os
import re
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ============================
# CONFIGURATION
# ============================

BASE_DIR = "/Users/prishapriyadashini/Downloads/DIMACS/model_results"

SELECTED_DATASETS = [
    "breast_cancer_data",
    "compas"
]

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================
# PARSE RESULT FILES
# ============================

results = []

for dataset in os.listdir(BASE_DIR):
    dataset_path = os.path.join(BASE_DIR, dataset)

    if dataset not in SELECTED_DATASETS:
        continue

    if not os.path.isdir(dataset_path):
        continue

    for file in os.listdir(dataset_path):
        if not file.endswith("_results.txt"):
            continue

        model_name = file.replace("_results.txt", "")
        file_path = os.path.join(dataset_path, file)

        with open(file_path, "r") as f:
            content = f.read()

        acc_match = re.search(r"Accuracy:\s*([0-9.]+)", content)
        accuracy = float(acc_match.group(1)) if acc_match else None

        macro_match = re.search(
            r"macro avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)",
            content
        )
        macro_f1 = float(macro_match.group(1)) if macro_match else None

        n_leaves = None
        size_json = os.path.join(dataset_path, f"{model_name}_tree_size.json")
        if os.path.exists(size_json):
            with open(size_json) as fj:
                sz = json.load(fj)
            if "error" not in sz:
                n_leaves = sz.get("n_leaves") or sz.get("total_leaves")

        results.append({
            "Dataset": dataset,
            "Model": model_name,
            "Accuracy": accuracy,
            "Macro F1": macro_f1,
            "Leaves": n_leaves,
        })

df = pd.DataFrame(results)

if df.empty:
    raise ValueError("No data found. Check dataset names and folder structure.")

# ============================
# Accuracy Heatmap
# ============================

accuracy_matrix = df.pivot(
    index="Model",
    columns="Dataset",
    values="Accuracy"
)

accuracy_matrix = accuracy_matrix.loc[
    accuracy_matrix.mean(axis=1).sort_values(ascending=False).index
]

plt.figure(figsize=(6,5))
sns.heatmap(
    accuracy_matrix,
    annot=True,
    fmt=".3f",
    cmap="viridis",
    linewidths=0.5
)

plt.title("Test Accuracy Across Models and Datasets")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_heatmap.png"))
plt.close()

# ============================
# Macro F1 Heatmap
# ============================

f1_matrix = df.pivot(
    index="Model",
    columns="Dataset",
    values="Macro F1"
)

f1_matrix = f1_matrix.loc[
    f1_matrix.mean(axis=1).sort_values(ascending=False).index
]

plt.figure(figsize=(6,5))
sns.heatmap(
    f1_matrix,
    annot=True,
    fmt=".3f",
    cmap="magma",
    linewidths=0.5
)

plt.title("Macro F1 Score Across Models and Datasets")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "f1_heatmap.png"))
plt.close()

# ============================
# Accuracy vs Tree Size (Log Scale)
# ============================

for dataset in SELECTED_DATASETS:
    subset = df[df["Dataset"] == dataset]

    plt.figure(figsize=(6,4))
    plt.scatter(subset["Leaves"], subset["Accuracy"])

    for _, row in subset.iterrows():
        plt.text(row["Leaves"], row["Accuracy"], row["Model"])

    plt.xscale("log")  # 🔥 added log scaling only
    plt.title(f"Accuracy vs Model Complexity ({dataset})")
    plt.xlabel("Number of Leaves (log scale)")
    plt.ylabel("Test Accuracy")

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"accuracy_vs_tree_size_{dataset}.png"
        )
    )
    plt.close()

print("All figures saved in 'figures/' directory.")