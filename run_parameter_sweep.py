#!/usr/bin/env python3
"""
Run parameter sweep experiments for XGBoost, Threshold Guessing (GOSDT),
and LicketyRESPLIT on every dataset in datasets/Mine/.
Results are saved to: model_results/<dataset_name>/<model>_<params>/

Note: gosdt and licketyresplit (resplit) may conflict via pybind11 if loaded
in the same process. If you see import errors, run their sweeps separately.
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
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from gosdt import ThresholdGuessBinarizer, GOSDTClassifier
from licketyresplit import LicketyRESPLIT
# from resplit.model.treefarms import TREEFARMS

# ============================
# CONFIGURATION
# ============================

BASEDIR = Path(__file__).resolve().parent

DATASETS = [
    {
        "path":          BASEDIR / "datasets/Mine/bike.csv",
        "target_column": "cnt_binary",
        "drop_columns":  ["cnt_binary"],
        "label_map":     None,
    },
    {
        "path":          BASEDIR / "datasets/Mine/breast_cancer_data.csv",
        "target_column": "diagnosis",
        "drop_columns":  ["id", "diagnosis"],
        "label_map":     {"M": 1, "B": 0},
    },
    {
        "path":          BASEDIR / "datasets/Mine/heloc_original.csv",
        "target_column": "RiskPerformance",
        "drop_columns":  ["RiskPerformance"],
        "label_map":     None,
    },
    {
        "path":          BASEDIR / "datasets/Mine/spambase.csv",
        "target_column": "class",
        "drop_columns":  ["class"],
        "label_map":     None,
    },
]

# Shared parameter grids
DEPTH_BUDGETS  = [3, 5]
LAMBDA_REGS    = [0.01, 0.05]
RASHOMON_MULTS = [0.01, 0.05]

# XGBoost-specific
XGB_N_ESTIMATORS = [25, 50]

# GOSDT-specific (threshold guessing binarizer params are fixed)
GOSDT_REGS       = [0.001, 0.01]
GBDT_N_EST       = 40
GBDT_MAX_DEPTH   = 1

# ============================
# HELPER FUNCTIONS
# ============================

def count_tree_nodes(node):
    if hasattr(node, 'left_child'):
        l_n, l_l = count_tree_nodes(node.left_child)
        r_n, r_l = count_tree_nodes(node.right_child)
        return 1 + l_n + r_n, l_l + r_l
    return 1, 1

def count_dict_tree(source):
    if "prediction" in source:
        return 1, 1
    l_n, l_l = count_dict_tree(source["true"])
    r_n, r_l = count_dict_tree(source["false"])
    return 1 + l_n + r_n, l_l + r_l

# ============================
# MAIN LOOP OVER DATASETS
# ============================

for dataset in DATASETS:
    datapath   = dataset["path"]
    target_col = dataset["target_column"]
    drop_cols  = dataset["drop_columns"]
    label_map  = dataset["label_map"]

    dataset_name = datapath.stem
    results_dir  = BASEDIR / "model_results" / dataset_name
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_name}")
    print(f"{'='*60}")
    print(f"Loading dataset: {datapath}")

    df = pd.read_csv(datapath)
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
    # XGBOOST SWEEP
    # ============================

    print("=" * 60)
    print(f"Running XGBoost parameter sweep on {dataset_name}...")
    print("=" * 60)

    # XGBoost disallows [ ] < in feature names
    X_train_xgb = X_train.copy()
    X_test_xgb  = X_test.copy()
    X_train_xgb.columns = X_train_xgb.columns.str.replace('[', '{', regex=False).str.replace(']', '}', regex=False).str.replace('<', 'lt', regex=False)
    X_test_xgb.columns  = X_test_xgb.columns.str.replace('[', '{', regex=False).str.replace(']', '}', regex=False).str.replace('<', 'lt', regex=False)

    for max_depth, n_est in product(DEPTH_BUDGETS, XGB_N_ESTIMATORS):
        print(f"\nXGBoost: max_depth={max_depth}, n_estimators={n_est}")

        param_dir = results_dir / f"xgboost_{max_depth}_{n_est}"
        os.makedirs(param_dir, exist_ok=True)

        xgb = XGBClassifier(
            max_depth=max_depth,
            n_estimators=n_est,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=1.0,
            reg_alpha=0.0,
            eval_metric="logloss",
            random_state=42,
        )
        start = time.perf_counter()
        xgb.fit(X_train_xgb, y_train)
        duration = time.perf_counter() - start

        y_pred = xgb.predict(X_test_xgb)
        acc    = accuracy_score(y_test, y_pred)

        try:
            trees_df = xgb.get_booster().trees_to_dataframe()
            n_leaves = int((trees_df['Feature'] == 'Leaf').sum())
            n_nodes  = int(len(trees_df))
            n_trees  = int(trees_df['Tree'].nunique())
            tree_size = {
                "n_trees": n_trees,
                "total_leaves": n_leaves,
                "total_nodes": n_nodes,
                "avg_leaves_per_tree": round(n_leaves / n_trees, 2),
            }
        except Exception as e:
            tree_size = {"error": str(e)}

        with open(param_dir / "xgboost_tree_size.json", "w") as f:
            json.dump(tree_size, f)

        with open(param_dir / "xgboost_results.txt", "w") as f:
            f.write(f"\nAccuracy: {acc}")
            f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
            f.write(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
            f.write(f"\nXGBoost completed in {duration:.2f} seconds")
            if "error" not in tree_size:
                f.write(f"\nTree Size: {tree_size['n_trees']} trees, {tree_size['total_leaves']} total leaves, {tree_size['avg_leaves_per_tree']:.1f} avg leaves/tree")

        print(f"  Accuracy: {acc:.4f}")

    # ============================
    # THRESHOLD GUESSING (GOSDT) SWEEP
    # ============================

    print("\n" + "=" * 60)
    print(f"Running Threshold Guessing (GOSDT) parameter sweep on {dataset_name}...")
    print("=" * 60)

    for depth, reg in product(DEPTH_BUDGETS, GOSDT_REGS):
        print(f"\nThreshold Guessing: depth={depth}, reg={reg}")

        param_dir = results_dir / f"gosdt_{depth}_{reg}"
        os.makedirs(param_dir, exist_ok=True)

        # Step 1: Threshold guessing binarization
        enc = ThresholdGuessBinarizer(n_estimators=GBDT_N_EST, max_depth=GBDT_MAX_DEPTH, random_state=42)
        enc.set_output(transform="pandas")
        X_train_bin = enc.fit_transform(X_train, y_train)
        X_test_bin  = enc.transform(X_test)

        # Step 2: Warm labels
        gbdt = GradientBoostingClassifier(n_estimators=GBDT_N_EST, max_depth=GBDT_MAX_DEPTH, random_state=42)
        gbdt.fit(X_train_bin, y_train)
        warm_labels = gbdt.predict(X_train_bin)

        # Step 3: Train GOSDT
        clf = GOSDTClassifier(
            regularization=reg,
            depth_budget=depth,
            similar_support=False,
            time_limit=60,
            verbose=False,
        )
        start = time.perf_counter()
        clf.fit(X_train_bin, y_train, y_ref=warm_labels)
        duration = time.perf_counter() - start

        y_pred = clf.predict(X_test_bin)
        acc    = accuracy_score(y_test, y_pred)

        try:
            n_nodes, n_leaves = count_tree_nodes(clf.trees_[0].tree)
            tree_size = {"n_leaves": n_leaves, "n_nodes": n_nodes}
        except Exception as e:
            tree_size = {"error": str(e)}

        with open(param_dir / "gosdt_tree_size.json", "w") as f:
            json.dump(tree_size, f)

        with open(param_dir / "gosdt_results.txt", "w") as f:
            f.write(f"\nAccuracy: {acc}")
            f.write(f"\nTraining Accuracy: {clf.score(X_train_bin, y_train)}")
            f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
            f.write(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
            f.write(f"\nGOSDT completed in {duration:.2f} seconds")
            if "error" not in tree_size:
                f.write(f"\nTree Size: {tree_size['n_leaves']} leaves, {tree_size['n_nodes']} total nodes")

        print(f"  Accuracy: {acc:.4f}")

    # ============================
    # LICKETYRESPLIT SWEEP
    # ============================

    print("\n" + "=" * 60)
    print(f"Running LicketyRESPLIT parameter sweep on {dataset_name}...")
    print("=" * 60)

    X_train_np = X_train.values.astype(np.uint8)
    X_test_np  = X_test.values.astype(np.uint8)
    y_train_np = y_train.values.astype(int)
    y_test_np  = y_test.values.astype(int)

    for depth, reg, rash in product(DEPTH_BUDGETS, LAMBDA_REGS, RASHOMON_MULTS):
        print(f"\nLicketyRESPLIT: depth={depth}, lambda={reg}, rashomon={rash}")

        param_dir = results_dir / f"{depth}_{reg}_{rash}"
        os.makedirs(param_dir, exist_ok=True)

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

        test_preds    = model.get_predictions(0, X_test_np)
        acc           = accuracy_score(y_test_np, test_preds)
        all_preds     = model.get_all_predictions(X_test_np, stack=True)
        majority_vote = (all_preds.mean(axis=0) >= 0.5).astype(np.uint8)
        ensemble_acc  = accuracy_score(y_test_np, majority_vote)

        try:
            paths, _ = model.get_tree_paths(0)
            n_leaves  = len(paths)
            n_nodes   = 2 * n_leaves - 1
            tree_size = {"n_leaves": n_leaves, "n_nodes": n_nodes, "n_trees_in_set": model.count_trees()}
        except Exception as e:
            tree_size = {"error": str(e)}

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
    # TREEFARMS SWEEP
    # ============================

    # print("\n" + "=" * 60)
    # print(f"Running TREEFARMS parameter sweep on {dataset_name}...")
    # print("=" * 60)

    # for depth, reg, rash in product(DEPTH_BUDGETS, LAMBDA_REGS, RASHOMON_MULTS):
    #     print(f"\nTREEFARMS: depth={depth}, reg={reg}, rashomon={rash}")

    #     param_dir = results_dir / f"treefarms_{depth}_{reg}_{rash}"
    #     os.makedirs(param_dir, exist_ok=True)

    #     config = {
    #         "regularization":          reg,
    #         "rashomon_bound_multiplier": rash,
    #         "depth_budget":            depth,
    #         "verbose":                 False,
    #     }
    #     model = TREEFARMS(config)
    #     start = time.perf_counter()
    #     model.fit(X_train, y_train)
    #     duration = time.perf_counter() - start

    #     tree   = model[0]
    #     y_pred = tree.predict(X_test)
    #     acc    = accuracy_score(y_test, y_pred)

    #     try:
    #         n_nodes, n_leaves = count_dict_tree(vars(model[0])['source'])
    #         tree_size = {"n_leaves": n_leaves, "n_nodes": n_nodes, "n_trees_in_set": model.get_tree_count()}
    #     except Exception as e:
    #         tree_size = {"error": str(e)}

    #     with open(param_dir / "treefarms_tree_size.json", "w") as f:
    #         json.dump(tree_size, f)

    #     with open(param_dir / "treefarms_results.txt", "w") as f:
    #         f.write(f"\nAccuracy: {acc}")
    #         f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    #         f.write(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    #         f.write(f"\nTREEFARMS completed in {duration:.2f} seconds with {model.get_tree_count()} trees")
    #         if "error" not in tree_size:
    #             f.write(f"\nTree Size (tree 0): {tree_size['n_leaves']} leaves, {tree_size['n_nodes']} total nodes")

    #     print(f"  Accuracy: {acc:.4f}, Rashomon size: {model.get_tree_count()}")

print("\n" + "=" * 60)
print("Parameter sweep complete!")
print("=" * 60)
