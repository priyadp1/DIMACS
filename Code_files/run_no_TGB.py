import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from gosdt import GOSDTClassifier
from licketyresplit import LicketyRESPLIT
from xgboost import XGBClassifier

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent
BASEDIR = current

RESULTS_DIR = BASEDIR / "model_results_no_tgb"
RESULTS_DIR.mkdir(exist_ok=True)

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
    },
    "diabetes": {
        "path": BASEDIR / "datasets" / "Mine" / "diabetic_data.csv",
        "target_col": 'readmitted',
        "drop_cols": ['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty', 'max_glu_serum', 'A1Cresult', 'readmitted'],
        "label_map": {">30": 1, "<30": 1, "NO": 0},
    },
    "diabetes_smote": {
        "path": BASEDIR / "datasets" / "Mine" / "diabetes_smote.csv",
        "target_col": 'readmitted',
        "drop_cols": ['readmitted'],
        "label_map": None,
    },
    "creditcard_fraud_smote": {
        "path": BASEDIR / "datasets" / "Mine" / "creditcard_fraud_detection_smote.csv",
        "target_col": "Class",
        "drop_cols": ["Class"],
        "label_map": None,
    },
    "creditcard_fraud": {
        "path": BASEDIR / "datasets" / "Mine" / "creditcard_fraud_detection_test.csv",
        "target_col": "Class",
        "drop_cols": ["Class"],
        "label_map": None,
    },
}

# ── GOSDT parameters ──────────────────────────────────────────────────────────
GOSDT_REG         = 0.001
GOSDT_DEPTH       = 6
GOSDT_TIME_LIMIT  = 60
GOSDT_SIM_SUPPORT = False
GOSDT_MAX_FEATURES = 50    # GOSDT-specific feature cap
GOSDT_MAX_ROWS     = 10000 # GOSDT-specific row cap

# ── Feature selection (applied to all models when features exceed threshold) ──
FEATURE_SELECTION_THRESHOLD = 500
TOP_K_FEATURES              = 200

# ── LicketyRESPLIT parameters ─────────────────────────────────────────────────
LR_DEPTH        = 3
LR_LAMBDA       = 0.003
LR_RASHOMON     = 0.05
LR_MAX_FEATURES = 50
LR_MAX_ROWS     = 15000

# ── XGBoost parameters ────────────────────────────────────────────────────────
XGB_MAX_DEPTH    = 3
XGB_N_ESTIMATORS = 25


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

    # ── GOSDT ─────────────────────────────────────────────────────────────────
    if (out_dir / "gosdt_results.txt").exists():
        print(f"\n  [GOSDT] Skipping {dataset_name} — results already exist.")
    else:
        print(f"\n  [GOSDT] Training on {dataset_name}...")
        X_train_gosdt, y_train_gosdt = X_train, y_train

        # Subsample rows if needed
        if len(X_train_gosdt) > GOSDT_MAX_ROWS:
            idx = np.random.default_rng(42).choice(len(X_train_gosdt), GOSDT_MAX_ROWS, replace=False)
            X_train_gosdt = X_train_gosdt.iloc[idx]
            y_train_gosdt = y_train_gosdt.iloc[idx]
            print(f"  [GOSDT] Subsampled to {GOSDT_MAX_ROWS} rows.")

        # Reduce features further if needed
        if X_train_gosdt.shape[1] > GOSDT_MAX_FEATURES:
            _imp = pd.Series(
                XGBClassifier(max_depth=3, n_estimators=25, learning_rate=0.1,
                              eval_metric="logloss", random_state=42)
                .fit(X_train_gosdt.rename(columns=lambda c: c.replace("[","{").replace("]","}").replace("<","lt")),
                     y_train_gosdt)
                .feature_importances_,
                index=X_train_gosdt.columns,
            )
            _top = _imp.nlargest(GOSDT_MAX_FEATURES).index.tolist()
            X_train_gosdt = X_train_gosdt[_top]
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

    # ── XGBoost ───────────────────────────────────────────────────────────────
    if (out_dir / "xgboost_results.txt").exists():
        print(f"\n  [XGBoost] Skipping {dataset_name} — results already exist.")
    else:
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

    # ── LicketyRESPLIT ────────────────────────────────────────────────────────
    if (out_dir / "licketyresplit_results.txt").exists():
        print(f"\n  [LicketyRESPLIT] Skipping {dataset_name} — results already exist.")
    else:
        print(f"\n  [LicketyRESPLIT] Training on {dataset_name}...")
        X_train_lr, y_train_lr = X_train, y_train

        # Subsample rows if needed
        if len(X_train_lr) > LR_MAX_ROWS:
            idx = np.random.default_rng(42).choice(len(X_train_lr), LR_MAX_ROWS, replace=False)
            X_train_lr = X_train_lr.iloc[idx]
            y_train_lr = y_train_lr.iloc[idx]
            print(f"  [LicketyRESPLIT] Subsampled to {LR_MAX_ROWS} rows.")

        # Reduce features further if needed
        if X_train_lr.shape[1] > LR_MAX_FEATURES:
            _imp = pd.Series(
                XGBClassifier(max_depth=3, n_estimators=25, learning_rate=0.1,
                              eval_metric="logloss", random_state=42)
                .fit(X_train_lr.rename(columns=lambda c: c.replace("[","{").replace("]","}").replace("<","lt")),
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

        with open(out_dir / "licketyresplit_tree_size.json", "w") as f:
            json.dump(lr_tree_size, f)

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


print(f"\n{'='*60}")
print(f"All results saved to: {RESULTS_DIR}")
print(f"{'='*60}")
