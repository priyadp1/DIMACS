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

from pathlib import Path
from sklearn.model_selection import train_test_split
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

TMP_DIR     = BASEDIR / "tmp_splits_no_tgb"
RESULTS_DIR = BASEDIR / "benchmarks_no_TGB_results"
RESULTS_DIR.mkdir(exist_ok=True)
TMP_DIR.mkdir(exist_ok=True)
print(f"Base dir : {BASEDIR}")
print(f"Tmp dir  : {TMP_DIR}")
print(f"Results  : {RESULTS_DIR}")

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
    "breast_cancer": {
        "path": BASEDIR / "datasets" / "Mine" / "breast_cancer_data.csv",
        "target_col": "diagnosis",
        "drop_cols": ["id", "diagnosis"],
        "label_map": {"M": 1, "B": 0},
    }
}

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

# ── Feature selection (applied when features exceed threshold) ────────────────
FEATURE_SELECTION_THRESHOLD = 500
TOP_K_FEATURES              = 200

# ── LicketyRESPLIT raw-data limits ───────────────────────────────────────────
LR_MAX_FEATURES = 50
LR_MAX_ROWS     = 15000


def count_gosdt_tree_nodes(node):
    if hasattr(node, "left_child"):
        l_n, l_l = count_gosdt_tree_nodes(node.left_child)
        r_n, r_l = count_gosdt_tree_nodes(node.right_child)
        return 1 + l_n + r_n, l_l + r_l
    return 1, 1


for dataset_name, cfg in DATASETS.items():
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'='*60}")

    df = pd.read_csv(cfg["path"]).dropna(axis=1, how="all")

    if cfg["label_map"]:
        df[cfg["target_col"]] = df[cfg["target_col"]].map(cfg["label_map"])

    X = df.drop(columns=cfg["drop_cols"])
    Y = df[cfg["target_col"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    # Encode any categorical columns
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=False)
        X_test  = pd.get_dummies(X_test,  columns=cat_cols, drop_first=False)
        X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    # ── Feature selection ─────────────────────────────────────────────────────
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
        _importances = pd.Series(_selector.feature_importances_, index=X_train.columns)
        _top_cols = _importances.nlargest(TOP_K_FEATURES).index.tolist()
        X_train = X_train[_top_cols]
        X_test  = X_test[_top_cols]
        print(f"  [Feature Selection] Reduced to {X_train.shape[1]} features.")

    out_dir = RESULTS_DIR / dataset_name
    out_dir.mkdir(exist_ok=True)

    _expected_results = [
        "gosdt_results.txt",
        "split_results.txt",
        "xgboost_binarized_results.txt",
        "lightgbm_results.txt",
        "catboost_results.txt",
        "licketysplit_results.txt",
        "licketyresplit_binarized_results.txt",
        "split_first_tree.txt",
        "gosdt_first_tree.txt",
        "licketysplit_first_tree.txt",
        "licketyresplit_binarized_first_tree.txt",
        "catboost_ensemble.json",
        "xgboost_ensemble.txt",
        "lightgbm_ensemble.txt",
    ]
    if all((out_dir / f).exists() for f in _expected_results):
        print(f"  [SKIP] Results already exist for {dataset_name}")
        continue

    # Save splits for GOSDT subprocess
    tmp_dir = TMP_DIR / dataset_name
    tmp_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(tmp_dir / "X_train.csv", index=False)
    X_test.to_csv(tmp_dir / "X_test.csv", index=False)
    y_train.to_csv(tmp_dir / "y_train.csv", index=False)
    y_test.to_csv(tmp_dir / "y_test.csv", index=False)

    # ── GOSDT (subprocess to avoid _libgosdt pybind11 conflict with SPLIT) ──────
    print(f"\n  [GOSDT] Training on {dataset_name}...")
    _gosdt_script = Path(__file__).parent / "run_gosdt_no_tgb.py"
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
    with open(out_dir / "split_first_tree.txt", "w", encoding="utf-8") as fh:
        try:
            fh.write(str(model.clf.trees_[0]))
        except Exception as e:
            fh.write(f"Tree unavailable: {e}\n")

    print(f"  [SPLIT] Accuracy: {accuracy_score(y_test, y_pred_split):.4f} | "f"Time: {split_duration:.2f}s")

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

    with open(out_dir / "xgboost_ensemble.txt", "w", encoding="utf-8") as fh:
        try:
            fh.write("\n".join(xgb.get_booster().get_dump()))
        except Exception as e:
            fh.write(f"Ensemble unavailable: {e}\n")

    print(f"  [XGBoost]  Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f} | Time: {xgb_duration:.2f}s")


    # ── LightGBM ──────────────────────────────────────────────────────────────
    print(f"\n  [LightGBM] Training on {dataset_name}...")

    # LightGBM forbids special JSON characters in feature names — sanitize all
    X_train_lgb = X_train.copy()
    X_test_lgb  = X_test.copy()
    _lgb_cols = X_train_lgb.columns.str.replace(r'[^A-Za-z0-9_]', '_', regex=True)
    # Deduplicate: append _1, _2, ... to any colliding names
    _seen = {}
    _deduped = []
    for col in _lgb_cols:
        if col in _seen:
            _seen[col] += 1
            _deduped.append(f"{col}_{_seen[col]}")
        else:
            _seen[col] = 0
            _deduped.append(col)
    X_train_lgb.columns = _deduped
    X_test_lgb.columns  = _deduped

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

    with open(out_dir / "lightgbm_ensemble.txt", "w", encoding="utf-8") as fh:
        try:
            fh.write(lgbm.booster_.model_to_string())
        except Exception as e:
            fh.write(f"Ensemble unavailable: {e}\n")

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

    try:
        cb.save_model(str(out_dir / "catboost_ensemble.json"), format="json")
    except Exception as e:
        with open(out_dir / "catboost_ensemble.txt", "w", encoding="utf-8") as fh:
            fh.write(f"Ensemble unavailable: {e}\n")

    print(f"  [CatBoost] Accuracy: {accuracy_score(y_test, y_pred_cb):.4f} | Time: {cb_duration:.2f}s")

    # ── LicketySPLIT ──────────────────────────────────────────────────────────
    print(f"\n  [LicketySPLIT] Training on {dataset_name}...")
    try:
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

        with open(out_dir / "licketysplit_first_tree.txt", "w", encoding="utf-8") as fh:
            try:
                fh.write(str(ls_model.clf.trees_[0]))
            except Exception as e:
                fh.write(f"Tree unavailable: {e}\n")

        print(f"  [LicketySPLIT] Accuracy: {accuracy_score(y_test, y_pred_ls):.4f} | Time: {ls_duration:.2f}s")
    except Exception as ls_err:
        ls_duration = time.perf_counter() - start
        print(f"  [LicketySPLIT] ERROR: {ls_err}")
        with open(out_dir / "licketysplit_results.txt", "w") as f:
            f.write(f"LicketySPLIT failed: {ls_err}\n")
            f.write(f"LicketySPLIT failed after {ls_duration:.2f} seconds")

    # ── LicketyRESPLIT (binarized) ────────────────────────────────────────────
    print(f"\n  [LicketyRESPLIT] Training on {dataset_name}...")
    X_train_lr, y_train_lr = X_train.copy(), y_train.copy()

    # Subsample rows if needed
    if len(X_train_lr) > LR_MAX_ROWS:
        idx = np.random.default_rng(42).choice(len(X_train_lr), LR_MAX_ROWS, replace=False)
        X_train_lr = X_train_lr.iloc[idx]
        y_train_lr = y_train_lr.iloc[idx]
        print(f"  [LicketyRESPLIT] Subsampled to {LR_MAX_ROWS} rows.")

    # Reduce features if needed
    if X_train_lr.shape[1] > LR_MAX_FEATURES:
        _imp = pd.Series(
            XGBClassifier(max_depth=3, n_estimators=25, learning_rate=0.1,
                          eval_metric="logloss", random_state=42)
            .fit(X_train_lr.rename(columns=lambda c: c.replace("[", "{").replace("]", "}").replace("<", "lt")),
                 y_train_lr)
            .feature_importances_,
            index=X_train_lr.columns,
        )
        _top = _imp.nlargest(LR_MAX_FEATURES).index.tolist()
        X_train_lr = X_train_lr[_top]
        print(f"  [LicketyRESPLIT] Reduced to {LR_MAX_FEATURES} features.")

    model = LicketyRESPLIT()
    start = time.perf_counter()
    model.fit(
        X_train_lr,
        y_train_lr,
        lambda_reg=LR_LAMBDA,
        depth_budget=LR_DEPTH,
        rashomon_mult=LR_RASHOMON,
        multiplicative_slack=0,
        key_mode="hash",
        trie_cache_enabled=False,
        lookahead_k=1,
    )
    lr_duration = time.perf_counter() - start

    test_preds_lr = model.get_predictions(0, X_test[X_train_lr.columns])

    n_trees = model.count_trees()
    votes = np.zeros(X_test.shape[0], dtype=np.int32)
    for idx in range(n_trees):
        votes += model.get_predictions(idx, X_test[X_train_lr.columns])
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

    with open(out_dir / "licketyresplit_binarized_first_tree.txt", "w", encoding="utf-8") as fh:
        try:
            paths, labels = model.get_tree_paths(0)
            cols = list(X_train_lr.columns)

            def _insert(node, path, label):
                if not path:
                    node["prediction"] = int(label)
                    return
                idx = path[0]
                feat = abs(idx) - 1
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

    print(f"  [LicketyRESPLIT] Accuracy: {accuracy_score(y_test, test_preds_lr):.4f} | "
          f"Ensemble: {ensemble_acc:.4f} | Trees: {n_trees} | Time: {lr_duration:.2f}s")




print(f"\n{'='*60}")
print(f"All results saved to: {RESULTS_DIR}")
print(f"{'='*60}")
