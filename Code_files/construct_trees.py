import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from gosdt import GOSDTClassifier
from licketyresplit import LicketyRESPLIT
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Path setup
# -----------------------------
current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent

BASEDIR     = current
TGB_DIR     = BASEDIR / "TGB_Variables"
RESULTS_DIR = BASEDIR / "model_results" / "construct_trees"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Datasets (names must match subdirectories in TGB_Variables/)
# -----------------------------
DATASETS = [
    "spambase",
    "bike",
    "compas",
    "breast_cancer",
    "diabetes",
    "diabetes_smote",
    "creditcard_fraud_smote",
    "creditcard_fraud",
]

# -----------------------------
# Model hyperparameters
# -----------------------------
GOSDT_REG          = 0.001
GOSDT_DEPTH        = 6
GOSDT_TIME_LIMIT   = 60
GOSDT_SIM_SUPPORT  = False
GOSDT_MAX_FEATURES = 50
GOSDT_MAX_ROWS     = 10000

XGB_MAX_DEPTH    = 3
XGB_N_ESTIMATORS = 25

FEATURE_SELECTION_THRESHOLD = 500
TOP_K_FEATURES              = 200

LR_DEPTH    = 3
LR_LAMBDA   = 0.003
LR_RASHOMON = 0.05
LR_MAX_FEATURES = 50
LR_MAX_ROWS     = 15000


def count_gosdt_tree_nodes(node):
    if hasattr(node, "left_child"):
        l_n, l_l = count_gosdt_tree_nodes(node.left_child)
        r_n, r_l = count_gosdt_tree_nodes(node.right_child)
        return 1 + l_n + r_n, l_l + r_l
    return 1, 1


def run_dataset(dataset_name: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'='*60}")

    out_dir = RESULTS_DIR / dataset_name
    out_dir.mkdir(exist_ok=True)

    tgb_dir = TGB_DIR / dataset_name
    if not tgb_dir.exists():
        print(f"  [SKIP] No TGB variables found at {tgb_dir}")
        return

    X_train = pd.read_csv(tgb_dir / "X_train_guessed.csv")
    X_test  = pd.read_csv(tgb_dir / "X_test_guessed.csv")
    y_train = pd.read_csv(tgb_dir / "y_train.csv").squeeze()
    y_test  = pd.read_csv(tgb_dir / "y_test.csv").squeeze()

    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"  Y dist:\n{y_train.value_counts()}")

    # Feature selection when features exceed threshold
    if X_train.shape[1] > FEATURE_SELECTION_THRESHOLD:
        print(f"  [Feature Selection] {X_train.shape[1]} features > {FEATURE_SELECTION_THRESHOLD} — "
              f"selecting top {TOP_K_FEATURES} via XGBoost importances...")
        _selector = XGBClassifier(
            max_depth=3, n_estimators=25, learning_rate=0.1,
            eval_metric="logloss", random_state=42,
        )
        _X_train_s = X_train.copy()
        _X_train_s.columns = (_X_train_s.columns
                              .str.replace("[", "{", regex=False)
                              .str.replace("]", "}", regex=False)
                              .str.replace("<", "lt", regex=False))
        _selector.fit(_X_train_s, y_train)
        _top_cols = pd.Series(_selector.feature_importances_, index=X_train.columns).nlargest(TOP_K_FEATURES).index.tolist()
        X_train = X_train[_top_cols]
        X_test  = X_test[_top_cols]
        print(f"  [Feature Selection] Reduced to {X_train.shape[1]} features.")

    # ── 1) GOSDT ──────────────────────────────────────────────────────────────
    if (out_dir / "gosdt_results.txt").exists():
        print(f"\n  [GOSDT] Skipping {dataset_name} — results already exist.")
    else:
        print(f"\n  [GOSDT] Training...")
        X_train_gosdt, y_train_gosdt = X_train.copy(), y_train.copy()

        if len(X_train_gosdt) > GOSDT_MAX_ROWS:
            idx = np.random.default_rng(42).choice(len(X_train_gosdt), GOSDT_MAX_ROWS, replace=False)
            X_train_gosdt = X_train_gosdt.iloc[idx]
            y_train_gosdt = y_train_gosdt.iloc[idx]
            print(f"  [GOSDT] Subsampled to {GOSDT_MAX_ROWS} rows.")

        if X_train_gosdt.shape[1] > GOSDT_MAX_FEATURES:
            _imp = pd.Series(
                XGBClassifier(max_depth=3, n_estimators=25, learning_rate=0.1,
                              eval_metric="logloss", random_state=42)
                .fit(X_train_gosdt.rename(columns=lambda c: c.replace("[", "{").replace("]", "}").replace("<", "lt")),
                     y_train_gosdt)
                .feature_importances_,
                index=X_train_gosdt.columns,
            )
            X_train_gosdt = X_train_gosdt[_imp.nlargest(GOSDT_MAX_FEATURES).index.tolist()]
            print(f"  [GOSDT] Reduced to {GOSDT_MAX_FEATURES} features.")

        clf = GOSDTClassifier(
            regularization=GOSDT_REG,
            similar_support=GOSDT_SIM_SUPPORT,
            time_limit=GOSDT_TIME_LIMIT,
            depth_budget=GOSDT_DEPTH,
            verbose=True,
        )
        clf.fit(X_train_gosdt, y_train_gosdt)
        y_pred_gosdt = clf.predict(X_test[X_train_gosdt.columns])

        try:
            n_nodes, n_leaves = count_gosdt_tree_nodes(clf.trees_[0].tree)
            gosdt_tree_size = {"n_leaves": n_leaves, "n_nodes": n_nodes}
        except Exception as e:
            gosdt_tree_size = {"error": str(e)}

        with open(out_dir / "gosdt_tree_size.json", "w") as f:
            json.dump(gosdt_tree_size, f)

        with open(out_dir / "gosdt_first_tree.txt", "w", encoding="utf-8") as fh:
            try:
                fh.write(str(clf.trees_[0]))
            except Exception as e:
                fh.write(f"Tree unavailable: {e}\n")

        with open(out_dir / "gosdt_results.txt", "w") as f:
            f.write(f"Accuracy: {accuracy_score(y_test, y_pred_gosdt)}")
            f.write(f"\nTraining Accuracy: {clf.score(X_train_gosdt, y_train_gosdt)}")
            f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_gosdt)}")
            f.write(f"\nClassification Report:\n{classification_report(y_test, y_pred_gosdt)}")
            f.write(f"\nGOSDT completed in {clf.result_.time:.2f} seconds")
            if "error" not in gosdt_tree_size:
                f.write(f"\nTree Size: {gosdt_tree_size['n_leaves']} leaves, {gosdt_tree_size['n_nodes']} total nodes")
            else:
                f.write(f"\nTree Size: Error - {gosdt_tree_size['error']}")

        print(f"  [GOSDT] Accuracy: {accuracy_score(y_test, y_pred_gosdt):.4f} | "
              f"Time: {clf.result_.time:.2f}s")

    # ── 2) XGBoost ────────────────────────────────────────────────────────────
    if (out_dir / "xgboost_results.txt").exists():
        print(f"\n  [XGBoost] Skipping {dataset_name} — results already exist.")
    else:
        print(f"\n  [XGBoost] Training...")
        X_train_xgb = X_train.copy()
        X_test_xgb  = X_test.copy()
        X_train_xgb.columns = (X_train_xgb.columns
                               .str.replace("[", "{", regex=False)
                               .str.replace("]", "}", regex=False)
                               .str.replace("<", "lt", regex=False))
        X_test_xgb.columns = X_train_xgb.columns

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

        importance_df = pd.DataFrame({
            "Feature":    X_train_xgb.columns,
            "Importance": xgb.feature_importances_,
        }).sort_values("Importance", ascending=False)

        with open(out_dir / "xgboost_tree_size.json", "w") as f:
            json.dump(xgb_tree_size, f)

        with open(out_dir / "xgboost_results.txt", "w") as f:
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

        print(f"  [XGBoost] Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f} | Time: {xgb_duration:.2f}s")

    # ── 3) LicketyRESPLIT ────────────────────────────────────────────────────
    if (out_dir / "licketyresplit_results.txt").exists():
        print(f"\n  [LicketyRESPLIT] Skipping {dataset_name} — results already exist.")
    else:
        print(f"\n  [LicketyRESPLIT] Training...")
        X_train_lr, y_train_lr = X_train.copy(), y_train.copy()

        if len(X_train_lr) > LR_MAX_ROWS:
            idx = np.random.default_rng(42).choice(len(X_train_lr), LR_MAX_ROWS, replace=False)
            X_train_lr = X_train_lr.iloc[idx]
            y_train_lr = y_train_lr.iloc[idx]
            print(f"  [LicketyRESPLIT] Subsampled to {LR_MAX_ROWS} rows.")

        if X_train_lr.shape[1] > LR_MAX_FEATURES:
            _imp = pd.Series(
                XGBClassifier(max_depth=3, n_estimators=25, learning_rate=0.1,
                              eval_metric="logloss", random_state=42)
                .fit(X_train_lr.rename(columns=lambda c: c.replace("[", "{").replace("]", "}").replace("<", "lt")),
                     y_train_lr)
                .feature_importances_,
                index=X_train_lr.columns,
            )
            X_train_lr = X_train_lr[_imp.nlargest(LR_MAX_FEATURES).index.tolist()]
            print(f"  [LicketyRESPLIT] Reduced to {LR_MAX_FEATURES} features.")

        lr_model = LicketyRESPLIT()
        start = time.perf_counter()
        lr_model.fit(
            X_train_lr, y_train_lr,
            lambda_reg=LR_LAMBDA,
            depth_budget=LR_DEPTH,
            rashomon_mult=LR_RASHOMON,
            multiplicative_slack=0,
            key_mode="hash",
            trie_cache_enabled=False,
            lookahead_k=1,
        )
        lr_duration = time.perf_counter() - start

        test_preds_lr = lr_model.get_predictions(0, X_test[X_train_lr.columns])
        n_trees = lr_model.count_trees()

        votes = np.zeros(X_test.shape[0], dtype=np.int32)
        for idx in range(n_trees):
            votes += lr_model.get_predictions(idx, X_test[X_train_lr.columns])
        ensemble_preds = (votes >= (n_trees / 2)).astype(int)
        ensemble_acc = accuracy_score(y_test, ensemble_preds)

        try:
            _paths, _ = lr_model.get_tree_paths(0)
            lr_tree_size = {"n_leaves": len(_paths), "n_nodes": 2 * len(_paths) - 1, "n_trees_in_set": n_trees}
        except Exception as e:
            lr_tree_size = {"error": str(e)}

        with open(out_dir / "licketyresplit_tree_size.json", "w") as f:
            json.dump(lr_tree_size, f)

        with open(out_dir / "licketyresplit_first_tree.txt", "w", encoding="utf-8") as fh:
            try:
                paths, labels = lr_model.get_tree_paths(0)
                cols = list(X_train_lr.columns)

                def _insert(node, path, label):
                    if not path:
                        node["prediction"] = int(label)
                        return
                    idx = path[0]
                    feat = abs(idx) - 1  # 1-indexed -> 0-indexed
                    node.setdefault("feature", feat)
                    child = "right" if idx > 0 else "left"
                    node.setdefault(child, {})
                    _insert(node[child], path[1:], label)

                def _fmt(node):
                    if "prediction" in node:
                        return f"{{ prediction: {node['prediction']}, loss: 0.0 }}"
                    feat = node["feature"]
                    left  = _fmt(node.get("left",  {"prediction": 0}))
                    right = _fmt(node.get("right", {"prediction": 1}))
                    return (f"{{ feature: {feat}, orig feature: {feat}, "
                            f"[ left child: {left}, right child: {right}] }}")

                root = {}
                for path, label in zip(paths, labels):
                    _insert(root, list(path), label)

                fh.write(f"{_fmt(root)}, Index({cols}, dtype='object')\n")
            except Exception as e:
                fh.write(f"Tree unavailable: {e}\n")

        with open(out_dir / "licketyresplit_results.txt", "w") as f:
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


for dataset_name in DATASETS:
    run_dataset(dataset_name)

print(f"\n{'='*60}")
print(f"All results saved to: {RESULTS_DIR}")
print(f"{'='*60}")
