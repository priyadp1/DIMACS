import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from gosdt import GOSDTClassifier
from licketyresplit import LicketyRESPLIT
from xgboost import XGBClassifier

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent
BASEDIR = current

TGB_DIR     = BASEDIR / "TGB_Variables"
RESULTS_DIR = BASEDIR / "model_results"
RESULTS_DIR.mkdir(exist_ok=True)

DATASETS = ["creditcard_fraud_smote", "creditcard_fraud"]  # Add more dataset names as needed, matching subdirectories in TGB_Variables

# ── GOSDT parameters (match run_gosdt.py) ────────────────────────────────────
GOSDT_REG         = 0.001
GOSDT_DEPTH       = 6
GOSDT_TIME_LIMIT  = 60
GOSDT_SIM_SUPPORT = False

# ── LicketyRESPLIT parameters (match run_licketyRESPLIT_given.py) ─────────────
LR_DEPTH    = 3
LR_LAMBDA   = 0.003
LR_RASHOMON = 0.05

# ── XGBoost parameters (match run_xgboost.py) ────────────────────────────────
XGB_MAX_DEPTH    = 3
XGB_N_ESTIMATORS = 25


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

    # ── GOSDT ─────────────────────────────────────────────────────────────────
    print(f"\n  [GOSDT] Training on {dataset_name}...")
    clf = GOSDTClassifier(
        regularization=GOSDT_REG,
        similar_support=GOSDT_SIM_SUPPORT,
        time_limit=GOSDT_TIME_LIMIT,
        depth_budget=GOSDT_DEPTH,
        verbose=True,
    )
    warm_classes = set(pd.Series(warm_labels).unique())
    y_classes = set(y_train.unique())
    if warm_classes == y_classes:
        clf.fit(X_train, y_train, y_ref=warm_labels)
    else:
        print(f"  [GOSDT] Warning: warm_labels classes {sorted(warm_classes)} != y classes {sorted(y_classes)}, skipping y_ref")
        clf.fit(X_train, y_train)
    y_pred_gosdt = clf.predict(X_test)

    try:
        n_nodes, n_leaves = count_gosdt_tree_nodes(clf.trees_[0].tree)
        gosdt_tree_size = {"n_leaves": n_leaves, "n_nodes": n_nodes}
    except Exception as e:
        gosdt_tree_size = {"error": str(e)}

    with open(out_dir / "gosdt_tree_size.json", "w") as f:
        json.dump(gosdt_tree_size, f)

    with open(out_dir / "gosdt_results.txt", "w") as f:
        f.write(f"Accuracy: {accuracy_score(y_test, y_pred_gosdt)}")
        f.write(f"\nTraining Accuracy: {clf.score(X_train, y_train)}")
        f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_gosdt)}")
        f.write(f"\nClassification Report:\n{classification_report(y_test, y_pred_gosdt)}")
        f.write(f"\nGOSDT completed in {clf.result_.time:.2f} seconds")
        if "error" not in gosdt_tree_size:
            f.write(f"\nTree Size: {gosdt_tree_size['n_leaves']} leaves, {gosdt_tree_size['n_nodes']} total nodes")
        else:
            f.write(f"\nTree Size: Error - {gosdt_tree_size['error']}")

    print(f"  [GOSDT] Accuracy: {accuracy_score(y_test, y_pred_gosdt):.4f} | "
          f"Time: {clf.result_.time:.2f}s")

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
