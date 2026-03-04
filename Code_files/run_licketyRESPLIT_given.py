import os
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from licketyresplit import LicketyRESPLIT
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent

BASEDIR = current
results_dir = BASEDIR / "model_results"
os.makedirs(results_dir, exist_ok=True)

DATASETS = {
    "spambase": {
        "path": BASEDIR / "datasets" / "Mine" / "spambase.csv",
        "target_col": "class",
        "drop_cols": ["class"],
        "label_map": None,
    },
    "bike": {
        "path": BASEDIR / "datasets" / "Mine" / "bike.csv",
        "target_col": "cnt_binary",
        "drop_cols": ["instant", "cnt_binary"],
        "label_map": None,
    },
    "compas": {
        "path": BASEDIR / "datasets" / "Mine" / "compas.csv",
        "target_col": "two_year_recid",
        "drop_cols": ["two_year_recid"],
        "label_map": None,
    },
    "heloc": {
        "path": BASEDIR / "datasets" / "Mine" / "heloc_original.csv",
        "target_col": "RiskPerformance",
        "drop_cols": ["RiskPerformance"],
        "label_map": None,
    },
}

depth_budget = 5
lambda_reg = 0.001
rashomon_mult = 0.05

for dataset_name, cfg in DATASETS.items():
    print(f"\n{'='*50}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*50}")

    df = pd.read_csv(cfg["path"])
    df = df.dropna(axis=1, how="all")

    if cfg["label_map"]:
        df[cfg["target_col"]] = df[cfg["target_col"]].map(cfg["label_map"])

    X = df.drop(columns=cfg["drop_cols"]).values
    Y = df[cfg["target_col"]].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    model = LicketyRESPLIT()

    start = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        lambda_reg=lambda_reg,
        depth_budget=depth_budget,
        rashomon_mult=rashomon_mult,
        multiplicative_slack=0,
        key_mode="hash",
        trie_cache_enabled=False,
        lookahead_k=1,
    )
    duration = time.perf_counter() - start

    print("Done training.")
    print("Minimum objective:", model.get_min_objective())
    print("Rashomon set size:", model.count_trees())

    tree_idx = 0
    test_preds = model.get_predictions(tree_idx, X_test)
    print("Test Accuracy:", accuracy_score(y_test, test_preds))
    print(classification_report(y_test, test_preds))

    all_preds_mat = model.get_all_predictions(X_test, stack=True)
    majority_vote = (all_preds_mat.mean(axis=0) >= 0.5).astype(np.uint8)
    ensemble_acc = accuracy_score(y_test, majority_vote)
    print("Ensemble Accuracy:", ensemble_acc)

    try:
        _paths, _preds = model.get_tree_paths(tree_idx)
        _n_leaves = len(_paths)
        _n_nodes = 2 * _n_leaves - 1
        tree_size = {"n_leaves": _n_leaves, "n_nodes": _n_nodes, "n_trees_in_set": model.count_trees()}
    except Exception as e:
        tree_size = {"error": str(e)}

    out_dir = BASEDIR / "LicketyRESPLIT_EXP" / f"{depth_budget}_{lambda_reg}_{rashomon_mult}"
    os.makedirs(out_dir, exist_ok=True)

    with open(out_dir / f"{dataset_name}_tree_size.json", "w") as f:
        json.dump(tree_size, f)

    with open(out_dir / f"{dataset_name}_results.txt", "w") as f:
        f.write(f"\nAccuracy: {accuracy_score(y_test, test_preds)}")
        f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, test_preds)}")
        f.write(f"\nClassification Report:\n{classification_report(y_test, test_preds)}")
        f.write(f"\nEnsemble Accuracy: {ensemble_acc}")
        f.write(f"\nLicketyRESPLIT completed in {duration:.2f} seconds with {model.count_trees()} trees")
        if "error" not in tree_size:
            f.write(f"\nTree Size (tree 0): {tree_size['n_leaves']} leaves, {tree_size['n_nodes']} total nodes")
