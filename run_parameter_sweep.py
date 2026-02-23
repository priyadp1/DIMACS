#!/usr/bin/env python3
"""
Run parameter sweep experiments for LicketyRESPLIT, RESPLIT, and TREEFARMS.
Results are saved to: model_results/<dataset_name>/<depth>_<reg>_<rashomon>/
"""

import os
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import models
from licketyresplit import LicketyRESPLIT
from resplit import RESPLIT
from resplit.model.treefarms import TREEFARMS
from split._tree import Leaf as SplitLeaf

# ============================
# CONFIGURATION - EDIT THIS SECTION
# ============================

BASEDIR = Path(__file__).resolve().parent

# Dataset configuration (copy these from run_all.py DATASET CONFIGURATION section)
DATAPATH = BASEDIR / "datasets" / "Mine" / "breast_cancer_data.csv"
TARGET_COLUMN = 'diagnosis'
DROP_COLUMNS = ['diagnosis']
LABEL_MAP = {"M": 1, "B": 0}  # set to None if labels are already numeric

# Results directory
dataset_name = DATAPATH.stem
results_dir = BASEDIR / "model_results" / dataset_name
os.makedirs(results_dir, exist_ok=True)

# Assign to internal variables
target_col = TARGET_COLUMN
drop_cols = DROP_COLUMNS
label_map = LABEL_MAP

# Parameter grids (customize these as needed)
DEPTH_BUDGETS = [3, 5]
LAMBDA_REGS = [0.01, 0.05]
RASHOMON_MULTS = [0.01, 0.05]

# For RESPLIT, also define cart_lookahead_depth (can loop over this too if desired)
CART_LOOKAHEAD_DEPTHS = [1, 3]

# ============================
# HELPER FUNCTIONS
# ============================

def count_tree_nodes(node):
    """Count (total_nodes, n_leaves) for split/gosdt Node/Leaf tree."""
    if hasattr(node, 'left_child'):
        l_n, l_l = count_tree_nodes(node.left_child)
        r_n, r_l = count_tree_nodes(node.right_child)
        return 1 + l_n + r_n, l_l + r_l
    return 1, 1

def count_dict_tree(source):
    """Count (total_nodes, n_leaves) for TreeClassifier dict tree."""
    if "prediction" in source:
        return 1, 1
    l_n, l_l = count_dict_tree(source["true"])
    r_n, r_l = count_dict_tree(source["false"])
    return 1 + l_n + r_n, l_l + r_l

# ============================
# LOAD DATASET
# ============================

print(f"Loading dataset: {DATAPATH}")
df = pd.read_csv(DATAPATH)
df = df.dropna(axis=1, how="all")

if label_map:
    df[target_col] = df[target_col].map(label_map)

X = df.drop(columns=drop_cols)
Y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

print(f"Train size: {X_train.shape}, Test size: {X_test.shape}\n")

# ============================
# LICKETYRESPLIT SWEEP
# ============================

print("=" * 60)
print("Running LicketyRESPLIT parameter sweep...")
print("=" * 60)

X_train_np = X_train.values.astype(np.uint8)
X_test_np = X_test.values.astype(np.uint8)
y_train_np = y_train.values.astype(int)
y_test_np = y_test.values.astype(int)

for depth, reg, rash in product(DEPTH_BUDGETS, LAMBDA_REGS, RASHOMON_MULTS):
    print(f"\nLicketyRESPLIT: depth={depth}, lambda={reg}, rashomon={rash}")

    # Create param directory
    param_dir = results_dir / f"{depth}_{reg}_{rash}"
    os.makedirs(param_dir, exist_ok=True)

    # Train model
    model = LicketyRESPLIT()
    start = time.perf_counter()
    model.fit(
        X_train_np, y_train_np,
        lambda_reg=reg,
        depth_budget=depth,
        rashomon_mult=rash,
        multiplicative_slack=0,
        key_mode="hash",
        trie_cache_enabled=False,
        lookahead_k=1,
    )
    duration = time.perf_counter() - start

    # Evaluate
    test_preds = model.get_predictions(0, X_test_np)
    acc = accuracy_score(y_test_np, test_preds)
    all_preds = model.get_all_predictions(X_test_np, stack=True)
    majority_vote = (all_preds.mean(axis=0) >= 0.5).astype(np.uint8)
    ensemble_acc = accuracy_score(y_test_np, majority_vote)

    # Tree size
    try:
        paths, _ = model.get_tree_paths(0)
        n_leaves = len(paths)
        n_nodes = 2 * n_leaves - 1
        tree_size = {"n_leaves": n_leaves, "n_nodes": n_nodes, "n_trees_in_set": model.count_trees()}
    except Exception as e:
        tree_size = {"error": str(e)}

    # Save results
    with open(param_dir / "licketyresplit_tree_size.json", "w") as f:
        json.dump(tree_size, f)

    with open(param_dir / "licketyresplit_results.txt", "w") as f:
        f.write(f"\nAccuracy: {acc}")
        f.write(f"\nEnsemble Accuracy: {ensemble_acc}")
        f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test_np, test_preds)}")
        f.write(f"\nClassification Report:\n{classification_report(y_test_np, test_preds)}")
        f.write(f"\nLicketyRESPLIT completed in {duration:.2f} seconds with {model.count_trees()} trees")
        if "error" not in tree_size:
            f.write(f"\nTree Size (tree 0): {tree_size['n_leaves']} leaves, {tree_size['n_nodes']} total nodes")

    print(f"  Accuracy: {acc:.4f}, Ensemble: {ensemble_acc:.4f}, Rashomon size: {model.count_trees()}")

# ============================
# RESPLIT SWEEP
# ============================

print("\n" + "=" * 60)
print("Running RESPLIT parameter sweep...")
print("=" * 60)

for depth, reg, rash, lookahead in product(DEPTH_BUDGETS, LAMBDA_REGS, RASHOMON_MULTS, CART_LOOKAHEAD_DEPTHS):
    print(f"\nRESPLIT: depth={depth}, reg={reg}, rashomon={rash}, lookahead={lookahead}")

    # Create param directory
    param_dir = results_dir / f"{depth}_{reg}_{rash}_{lookahead}"
    os.makedirs(param_dir, exist_ok=True)

    # Train model
    config = {
        "regularization": reg,
        "rashomon_bound_multiplier": rash,
        "depth_budget": depth,
        "cart_lookahead_depth": lookahead,
        "verbose": False
    }
    model = RESPLIT(config, fill_tree='treefarms')
    start = time.perf_counter()
    model.fit(X_train, y_train)
    duration = time.perf_counter() - start

    # Evaluate
    y_pred = model.predict(X_test, idx=0)
    acc = accuracy_score(y_test, y_pred)

    # Tree size
    try:
        n_nodes, n_leaves = count_tree_nodes(model[0])
        tree_size = {"n_leaves": n_leaves, "n_nodes": n_nodes, "n_trees_in_set": len(model)}
    except Exception as e:
        tree_size = {"error": str(e)}

    # Save results
    with open(param_dir / "resplit_tree_size.json", "w") as f:
        json.dump(tree_size, f)

    with open(param_dir / "resplit_results.txt", "w") as f:
        f.write(f"\nAccuracy: {acc}")
        f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        f.write(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
        f.write(f"\nRESPLIT completed in {duration:.2f} seconds with {len(model)} trees")
        if "error" not in tree_size:
            f.write(f"\nTree Size (tree 0): {tree_size['n_leaves']} leaves, {tree_size['n_nodes']} total nodes")

    print(f"  Accuracy: {acc:.4f}, Rashomon size: {len(model)}")

# ============================
# TREEFARMS SWEEP
# ============================

print("\n" + "=" * 60)
print("Running TREEFARMS parameter sweep...")
print("=" * 60)

for depth, reg, rash in product(DEPTH_BUDGETS, LAMBDA_REGS, RASHOMON_MULTS):
    print(f"\nTREEFARMS: depth={depth}, reg={reg}, rashomon={rash}")

    # Create param directory
    param_dir = results_dir / f"treefarms_{depth}_{reg}_{rash}"
    os.makedirs(param_dir, exist_ok=True)

    # Train model
    config = {
        "regularization": reg,
        "rashomon_bound_multiplier": rash,
        "depth_budget": depth,
        "verbose": False
    }
    model = TREEFARMS(config)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    duration = time.perf_counter() - start

    # Evaluate
    tree = model[0]
    y_pred = tree.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Tree size
    try:
        n_nodes, n_leaves = count_dict_tree(vars(model[0])['source'])
        tree_size = {"n_leaves": n_leaves, "n_nodes": n_nodes, "n_trees_in_set": model.get_tree_count()}
    except Exception as e:
        tree_size = {"error": str(e)}

    # Save results
    with open(param_dir / "treefarms_tree_size.json", "w") as f:
        json.dump(tree_size, f)

    with open(param_dir / "treefarms_results.txt", "w") as f:
        f.write(f"\nAccuracy: {acc}")
        f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        f.write(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
        f.write(f"\nTREEFARMS completed in {duration:.2f} seconds with {model.get_tree_count()} trees")
        if "error" not in tree_size:
            f.write(f"\nTree Size (tree 0): {tree_size['n_leaves']} leaves, {tree_size['n_nodes']} total nodes")

    print(f"  Accuracy: {acc:.4f}, Rashomon size: {model.get_tree_count()}")

print("\n" + "=" * 60)
print("Parameter sweep complete!")
print(f"All results saved to: {results_dir}")
print("=" * 60)
