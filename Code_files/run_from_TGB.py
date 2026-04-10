print("Loading standard libraries...")
import json
import subprocess
import sys
import time
import numpy as np
import pandas as pd
print("  numpy, pandas OK")

print("Loading LightGBM...")
import lightgbm as lgb
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
print("  LightGBM OK")

print("Loading CatBoost...")
from catboost import CatBoostClassifier
print("  CatBoost OK")

print("Loading SPLIT / LicketySPLIT...")
from split import SPLIT, LicketySPLIT
print("  SPLIT OK")

print("Loading RESPLIT...")
from resplit import RESPLIT  # keep import to validate install at startup
print("  RESPLIT OK")

from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Loading LicketyRESPLIT...")
from licketyresplit import LicketyRESPLIT
print("  LicketyRESPLIT OK")

print("Loading XGBoost...")
from xgboost import XGBClassifier
print("  XGBoost OK")

print("\nAll imports successful.")

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent
BASEDIR = current

TGB_DIR     = BASEDIR / "TGB_Variables"
RESULTS_DIR = BASEDIR / "benchmarks_TGB_results"
RESULTS_DIR.mkdir(exist_ok=True)
print(f"Base dir : {BASEDIR}")
print(f"TGB dir  : {TGB_DIR}")
print(f"Results  : {RESULTS_DIR}")

DATASETS = [
    "spambase",
    "bike",
    "compas",
    "breast_cancer",
    "diabetes",
    "diabetes_smote",
    "creditcard_fraud_smote",
    "creditcard_fraud",
]  # Add more dataset names as needed, matching subdirectories in TGB_Variables

# ── GOSDT parameters (match run_gosdt.py) ────────────────────────────────────
GOSDT_REG         = 0.001
GOSDT_DEPTH       = 6
GOSDT_TIME_LIMIT  = 60
GOSDT_SIM_SUPPORT = False

# ── LicketySPLIT parameters ───────────────────────────────────────────────────
LS_DEPTH = 3
LS_REG   = 0.003

# ── LicketyRESPLIT parameters (match run_licketyRESPLIT_given.py) ─────────────
LR_DEPTH    = 3
LR_LAMBDA   = 0.003
LR_RASHOMON = 0.05

# ── XGBoost parameters (match run_xgboost.py) ────────────────────────────────
XGB_MAX_DEPTH    = 3
XGB_N_ESTIMATORS = 25

# ── LightGBM parameters ───────────────────────────────────────────────────────
LGB_MAX_DEPTH    = 3
LGB_N_ESTIMATORS = 25
LGB_NUM_LEAVES   = 31

# ── CatBoost parameters ───────────────────────────────────────────────────────
CB_DEPTH      = 3
CB_ITERATIONS = 25

# SPLIT parameters (match run_split.py) ───────────────────────────────────────
lookahead_depth = 2
depth_buget = 5
regularization = 0.01

#RESPLIT parameters (match run_resplit.py) ───────────────────────────────────────
resplit_regularization = 0.01
resplit_rashomon_bound_multiplier = 0.05
resplit_depth_budget = 5
resplit_cart_lookahead_depth = 3


def count_gosdt_tree_nodes(node):
    if hasattr(node, "left_child"):
        l_n, l_l = count_gosdt_tree_nodes(node.left_child)
        r_n, r_l = count_gosdt_tree_nodes(node.right_child)
        return 1 + l_n + r_n, l_l + r_l
    return 1, 1


for dataset_name in DATASETS:
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'='*60}")

    tgb_dir = TGB_DIR / dataset_name
    if not tgb_dir.exists():
        print(f"  [SKIP] No TGB variables found at {tgb_dir}")
        continue

    # Load pre-computed TGB outputs
    X_train = pd.read_csv(tgb_dir / "X_train_guessed.csv")
    X_test  = pd.read_csv(tgb_dir / "X_test_guessed.csv")
    y_train = pd.read_csv(tgb_dir / "y_train.csv").squeeze()
    y_test  = pd.read_csv(tgb_dir / "y_test.csv").squeeze()
    warm_labels = pd.read_csv(tgb_dir / "warm_labels.csv").squeeze().to_numpy()

    out_dir = RESULTS_DIR / dataset_name
    out_dir.mkdir(exist_ok=True)

    _expected_results = [
        "gosdt_results.txt",
        "split_results.txt",
        "resplit_results.txt",
        "xgboost_binarized_results.txt",
        "lightgbm_results.txt",
        "catboost_results.txt",
        "licketysplit_results.txt",
        "licketyresplit_binarized_results.txt",
    ]
    if all((out_dir / f).exists() for f in _expected_results):
        print(f"  [SKIP] Results already exist for {dataset_name}")
        continue

    # ── GOSDT (subprocess to avoid _libgosdt pybind11 conflict with SPLIT) ──────
    print(f"\n  [GOSDT] Training on {dataset_name}...")
    _gosdt_script = Path(__file__).parent / "run_gosdt.py"
    _result = subprocess.run(
        [sys.executable, str(_gosdt_script), dataset_name],
        capture_output=False,
    )
    if _result.returncode != 0:
        print(f"  [GOSDT] ERROR: subprocess exited with code {_result.returncode}")

    # SPLIT
    print(f"\n  [SPLIT] Training on {dataset_name}...")
    model = SPLIT(
        lookahead_depth_budget=lookahead_depth,
        reg=regularization,
        full_depth_budget=depth_buget,
        verbose=True,
        binarize=True,
        time_limit=100,
    )
    start = time.perf_counter()
    model.fit(X_train, y_train)
    split_duration = time.perf_counter() - start
    y_pred_split = model.predict(X_test)

    try:
        _split_root = model.tree if model.tree is not None else model.clf.trees_[0].tree
        split_n_nodes, split_n_leaves = count_gosdt_tree_nodes(_split_root)
        split_tree_size = {"n_leaves": split_n_leaves, "n_nodes": split_n_nodes}
    except Exception as e:
        split_tree_size = {"error": str(e)}

    with open(out_dir / "split_results.txt", "w") as f:
        f.write(f"\nAccuracy: {accuracy_score(y_test, y_pred_split)}")
        f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_split)}")
        f.write(f"\nClassification Report:\n{classification_report(y_test, y_pred_split)}")
        f.write(f"\nSPLIT completed in {split_duration:.2f} seconds")
        if "error" not in split_tree_size:
            f.write(f"\nTree Size: {split_tree_size['n_leaves']} leaves, {split_tree_size['n_nodes']} total nodes")
        else:
            f.write(f"\nTree Size: Error - {split_tree_size['error']}")
    print(f"  [SPLIT] Accuracy: {accuracy_score(y_test, y_pred_split):.4f} | "f"Time: {split_duration:.2f}s")

    # ── RESPLIT (subprocess to avoid C++ abort() crash) ───────────────────────
    print(f"\n  [RESPLIT] Training on {dataset_name}...")
    _resplit_script = Path(__file__).parent / "run_resplit.py"
    _result = subprocess.run(
        [sys.executable, str(_resplit_script), dataset_name],
        capture_output=False,
    )
    if _result.returncode not in (0, 1):
        print(f"  [RESPLIT] ERROR: subprocess exited with code {_result.returncode}")
    # ── XGBoost ───────────────────────────────────────────────────────────────
    print(f"\n  [XGBoost] Training on {dataset_name}...")

    # XGBoost forbids '[', ']', '<' in feature names — sanitize
    X_train_xgb = X_train.copy()
    X_test_xgb  = X_test.copy()
    X_train_xgb.columns = (X_train_xgb.columns
                           .str.replace("[", "{", regex=False)
                           .str.replace("]", "}", regex=False)
                           .str.replace("<", "lt", regex=False))
    X_test_xgb.columns  = X_train_xgb.columns

    xgb = XGBClassifier(
        max_depth=XGB_MAX_DEPTH,
        n_estimators=XGB_N_ESTIMATORS,
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
    xgb_duration = time.perf_counter() - start
    y_pred_xgb = xgb.predict(X_test_xgb)

    try:
        trees_df    = xgb.get_booster().trees_to_dataframe()
        xgb_leaves  = int((trees_df["Feature"] == "Leaf").sum())
        xgb_nodes   = int(len(trees_df))
        xgb_n_trees = int(trees_df["Tree"].nunique())
        xgb_tree_size = {
            "n_trees": xgb_n_trees,
            "total_leaves": xgb_leaves,
            "total_nodes": xgb_nodes,
            "avg_leaves_per_tree": round(xgb_leaves / xgb_n_trees, 2),
        }
    except Exception as e:
        xgb_tree_size = {"error": str(e)}

    importance_df = (pd.DataFrame({
        "Feature":    X_train_xgb.columns,
        "Importance": xgb.feature_importances_,
    }).sort_values("Importance", ascending=False))

    with open(out_dir / "xgboost_tree_size_binarized.json", "w") as f:
        json.dump(xgb_tree_size, f)

    with open(out_dir / "xgboost_binarized_results.txt", "w") as f:
        f.write(f"Accuracy: {accuracy_score(y_test, y_pred_xgb)}")
        f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_xgb)}")
        f.write(f"\nClassification Report:\n{classification_report(y_test, y_pred_xgb)}")
        f.write(f"\nXGBoost completed in {xgb_duration:.2f} seconds")
        f.write(f"\nTop 3 Features:\n{importance_df.head(3).to_string(index=False)}")
        if "error" not in xgb_tree_size:
            f.write(f"\nTree Size: {xgb_tree_size['n_trees']} trees, "
                    f"{xgb_tree_size['total_leaves']} total leaves, "
                    f"{xgb_tree_size['avg_leaves_per_tree']:.1f} avg leaves/tree")
        else:
            f.write(f"\nTree Size: Error - {xgb_tree_size['error']}")

    print(f"  [XGBoost]  Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f} | Time: {xgb_duration:.2f}s")


    # ── LightGBM ──────────────────────────────────────────────────────────────
    print(f"\n  [LightGBM] Training on {dataset_name}...")

    # LightGBM forbids '[', ']', ',' in feature names — sanitize
    X_train_lgb = X_train.copy()
    X_test_lgb  = X_test.copy()
    X_train_lgb.columns = (X_train_lgb.columns
                           .str.replace("[", "{", regex=False)
                           .str.replace("]", "}", regex=False)
                           .str.replace("<", "lt", regex=False))
    X_test_lgb.columns = X_train_lgb.columns

    lgbm = LGBMClassifier(
        max_depth=LGB_MAX_DEPTH,
        n_estimators=LGB_N_ESTIMATORS,
        num_leaves=LGB_NUM_LEAVES,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=42,
        verbose=-1,
    )
    start = time.perf_counter()
    lgbm.fit(
        X_train_lgb, y_train,
        eval_set=[(X_test_lgb, y_test)],
        callbacks=[early_stopping(10, verbose=False), log_evaluation(period=-1)],
    )
    lgb_duration = time.perf_counter() - start
    y_pred_lgb = lgbm.predict(X_test_lgb)

    try:
        lgb_booster  = lgbm.booster_
        trees_info   = lgb_booster.dump_model()["tree_info"]
        lgb_n_trees  = len(trees_info)
        lgb_leaves   = sum(t["num_leaves"] for t in trees_info)
        lgb_tree_size = {
            "n_trees": lgb_n_trees,
            "total_leaves": lgb_leaves,
            "avg_leaves_per_tree": round(lgb_leaves / lgb_n_trees, 2),
        }
    except Exception as e:
        lgb_tree_size = {"error": str(e)}

    lgb_importance_df = (pd.DataFrame({
        "Feature":    X_train_lgb.columns,
        "Importance": lgbm.feature_importances_,
    }).sort_values("Importance", ascending=False))

    with open(out_dir / "lightgbm_tree_size.json", "w") as f:
        json.dump(lgb_tree_size, f)

    with open(out_dir / "lightgbm_results.txt", "w") as f:
        f.write(f"Accuracy: {accuracy_score(y_test, y_pred_lgb)}")
        f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_lgb)}")
        f.write(f"\nClassification Report:\n{classification_report(y_test, y_pred_lgb)}")
        f.write(f"\nLightGBM completed in {lgb_duration:.2f} seconds")
        f.write(f"\nTop 3 Features:\n{lgb_importance_df.head(3).to_string(index=False)}")
        if "error" not in lgb_tree_size:
            f.write(f"\nTree Size: {lgb_tree_size['n_trees']} trees, "
                    f"{lgb_tree_size['total_leaves']} total leaves, "
                    f"{lgb_tree_size['avg_leaves_per_tree']:.1f} avg leaves/tree")
        else:
            f.write(f"\nTree Size: Error - {lgb_tree_size['error']}")

    print(f"  [LightGBM] Accuracy: {accuracy_score(y_test, y_pred_lgb):.4f} | Time: {lgb_duration:.2f}s")

    # ── CatBoost ───────────────────────────────────────────────────────────────
    print(f"\n  [CatBoost] Training on {dataset_name}...")

    cb = CatBoostClassifier(
        depth=CB_DEPTH,
        iterations=CB_ITERATIONS,
        learning_rate=0.1,
        l2_leaf_reg=1.0,
        random_state=42,
        verbose=0,
    )
    start = time.perf_counter()
    cb.fit(X_train, y_train, eval_set=(X_test, y_test))
    cb_duration = time.perf_counter() - start
    y_pred_cb = cb.predict(X_test)

    try:
        cb_n_trees  = cb.tree_count_
        cb_leaves   = int(cb.get_leaf_values().shape[0])
        cb_tree_size = {
            "n_trees": cb_n_trees,
            "total_leaves": cb_leaves,
            "avg_leaves_per_tree": round(cb_leaves / cb_n_trees, 2),
        }
    except Exception as e:
        cb_tree_size = {"error": str(e)}

    cb_importance_df = (pd.DataFrame({
        "Feature":    X_train.columns,
        "Importance": cb.get_feature_importance(),
    }).sort_values("Importance", ascending=False))

    with open(out_dir / "catboost_tree_size.json", "w") as f:
        json.dump(cb_tree_size, f)

    with open(out_dir / "catboost_results.txt", "w") as f:
        f.write(f"Accuracy: {accuracy_score(y_test, y_pred_cb)}")
        f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_cb)}")
        f.write(f"\nClassification Report:\n{classification_report(y_test, y_pred_cb)}")
        f.write(f"\nCatBoost completed in {cb_duration:.2f} seconds")
        f.write(f"\nTop 3 Features:\n{cb_importance_df.head(3).to_string(index=False)}")
        if "error" not in cb_tree_size:
            f.write(f"\nTree Size: {cb_tree_size['n_trees']} trees, "
                    f"{cb_tree_size['total_leaves']} total leaves, "
                    f"{cb_tree_size['avg_leaves_per_tree']:.1f} avg leaves/tree")
        else:
            f.write(f"\nTree Size: Error - {cb_tree_size['error']}")

    print(f"  [CatBoost] Accuracy: {accuracy_score(y_test, y_pred_cb):.4f} | Time: {cb_duration:.2f}s")

    # ── LicketySPLIT ──────────────────────────────────────────────────────────
    print(f"\n  [LicketySPLIT] Training on {dataset_name}...")
    ls_model = LicketySPLIT(
        reg=LS_REG,
        full_depth_budget=LS_DEPTH,
        verbose=False,
    )
    start = time.perf_counter()
    ls_model.fit(X_train, y_train)
    ls_duration = time.perf_counter() - start
    y_pred_ls = ls_model.predict(X_test)

    try:
        _ls_root = ls_model.tree if ls_model.tree is not None else ls_model.clf.trees_[0].tree
        ls_n_nodes, ls_n_leaves = count_gosdt_tree_nodes(_ls_root)
        ls_tree_size = {"n_leaves": ls_n_leaves, "n_nodes": ls_n_nodes}
    except Exception as e:
        ls_tree_size = {"error": str(e)}

    with open(out_dir / "licketysplit_results.txt", "w") as f:
        f.write(f"Accuracy: {accuracy_score(y_test, y_pred_ls)}")
        f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_ls)}")
        f.write(f"\nClassification Report:\n{classification_report(y_test, y_pred_ls)}")
        f.write(f"\nLicketySPLIT completed in {ls_duration:.2f} seconds")
        if "error" not in ls_tree_size:
            f.write(f"\nTree Size: {ls_tree_size['n_leaves']} leaves, {ls_tree_size['n_nodes']} total nodes")
        else:
            f.write(f"\nTree Size: Error - {ls_tree_size['error']}")

    print(f"  [LicketySPLIT] Accuracy: {accuracy_score(y_test, y_pred_ls):.4f} | Time: {ls_duration:.2f}s")

    # ── LicketyRESPLIT (binarized) ────────────────────────────────────────────
    print(f"\n  [LicketyRESPLIT] Training on {dataset_name}...")
    model = LicketyRESPLIT()
    start = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        lambda_reg=LR_LAMBDA,
        depth_budget=LR_DEPTH,
        rashomon_mult=LR_RASHOMON,
        multiplicative_slack=0,
        key_mode="hash",
        trie_cache_enabled=False,
        lookahead_k=1,
    )
    lr_duration = time.perf_counter() - start

    test_preds_lr = model.get_predictions(0, X_test)

    n_trees = model.count_trees()
    votes = np.zeros(X_test.shape[0], dtype=np.int32)
    for idx in range(n_trees):
        votes += model.get_predictions(idx, X_test)
    ensemble_preds = (votes >= (n_trees / 2)).astype(int)
    ensemble_acc = accuracy_score(y_test, ensemble_preds)

    try:
        _paths, _ = model.get_tree_paths(0)
        lr_n_leaves = len(_paths)
        lr_n_nodes  = 2 * lr_n_leaves - 1
        lr_tree_size = {"n_leaves": lr_n_leaves, "n_nodes": lr_n_nodes, "n_trees_in_set": n_trees}
    except Exception as e:
        lr_tree_size = {"error": str(e)}

    with open(out_dir / "licketyresplit_binarized_tree_size.json", "w") as f:
        json.dump(lr_tree_size, f)

    with open(out_dir / "licketyresplit_binarized_results.txt", "w") as f:
        f.write(f"Accuracy: {accuracy_score(y_test, test_preds_lr)}")
        f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, test_preds_lr)}")
        f.write(f"\nClassification Report:\n{classification_report(y_test, test_preds_lr)}")
        f.write(f"\nEnsemble Accuracy: {ensemble_acc}")
        f.write(f"\nLicketyRESPLIT completed in {lr_duration:.2f} seconds with {n_trees} trees")
        if "error" not in lr_tree_size:
            f.write(f"\nTree Size (tree 0): {lr_tree_size['n_leaves']} leaves, {lr_tree_size['n_nodes']} total nodes")
        else:
            f.write(f"\nTree Size: Error - {lr_tree_size['error']}")

    print(f"  [LicketyRESPLIT] Accuracy: {accuracy_score(y_test, test_preds_lr):.4f} | "
          f"Ensemble: {ensemble_acc:.4f} | Trees: {n_trees} | Time: {lr_duration:.2f}s")




print(f"\n{'='*60}")
print(f"All results saved to: {RESULTS_DIR}")
print(f"{'='*60}")
