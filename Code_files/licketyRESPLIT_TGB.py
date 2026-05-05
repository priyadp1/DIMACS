import os
import time
import json
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from licketyresplit import LicketyRESPLIT
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from gosdt import ThresholdGuessBinarizer

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent

BASEDIR = current

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

DEPTH_BUDGETS    = [2, 3, 4]
LAMBDA_REGS      = [0.001, 0.003, 0.01]
RASHOMON_MULTS   = [0.01, 0.05, 0.1]

total_combos = len(DEPTH_BUDGETS) * len(LAMBDA_REGS) * len(RASHOMON_MULTS)
total_runs = len(DATASETS) * total_combos
run_num = 0
overall_start = time.perf_counter()

print(f"Starting sweep: {len(DATASETS)} datasets x {total_combos} param combos = {total_runs} total runs")

for dataset_name, cfg in DATASETS.items():
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    print(f"  Loading data from {cfg['path']}...")
    df = pd.read_csv(cfg["path"]).dropna(axis=1, how="all")
    print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} cols")

    if cfg["label_map"] is not None:
        df[cfg["target_col"]] = df[cfg["target_col"]].map(cfg["label_map"])
        print(f"  Applied label map: {cfg['label_map']}")

    string_cols = [c for c in df.select_dtypes(include=["object", "str"]).columns if c != cfg["target_col"]]
    df = pd.get_dummies(df, columns=string_cols, drop_first=True)

    Y = df[cfg["target_col"]]
    X = df.drop(columns=cfg["drop_cols"])
    print(f"  Features: {X.shape[1]}, Samples: {X.shape[0]}, Class balance: {Y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )
    print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    print(f"  Running ThresholdGuessBinarizer...")
    enc = ThresholdGuessBinarizer(n_estimators=50, max_depth=2, random_state=42)
    enc.set_output(transform="pandas")
    X_train_guessed = enc.fit_transform(X_train, y_train)
    X_test_guessed = enc.transform(X_test)
    print(f"  After binarizing: {X_train_guessed.shape[1]} features")

    for depth_budget, lambda_reg, rashomon_mult in itertools.product(DEPTH_BUDGETS, LAMBDA_REGS, RASHOMON_MULTS):
        run_num += 1
        param_str = f"depth{depth_budget}_lambda{lambda_reg}_rashomon{rashomon_mult}"
        elapsed = time.perf_counter() - overall_start
        print(f"\n  [{run_num}/{total_runs}] {dataset_name} | {param_str}  (elapsed: {elapsed:.1f}s)")

        results_dir = BASEDIR / "LicketyRESPLIT_EXP_ThresholdBinarizer" / dataset_name / param_str
        os.makedirs(results_dir, exist_ok=True)

        model = LicketyRESPLIT()

        print(f"    Fitting model...", flush=True)
        start = time.perf_counter()
        model.fit(
            X_train_guessed,
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
        print(f"    Done in {duration:.2f}s | min obj: {model.get_min_objective():.4f} | Rashomon set: {model.count_trees()} trees")

        tree_idx = 0
        test_preds = model.get_predictions(tree_idx, X_test_guessed)
        acc = accuracy_score(y_test, test_preds)

        n_samples = X_test_guessed.shape[0]
        votes = np.zeros(n_samples, dtype=np.int32)
        n_trees = model.count_trees()
        for ti in range(n_trees):
            votes += model.get_predictions(ti, X_test_guessed)
        majority_vote = (votes >= (n_trees / 2)).astype(int)
        ensemble_acc = accuracy_score(y_test, majority_vote)
        print(f"    Test Acc: {acc:.4f} | Ensemble Acc: {ensemble_acc:.4f}")

        try:
            _paths, _ = model.get_tree_paths(0)
            _n_leaves = len(_paths)
            _n_nodes = 2 * _n_leaves - 1
            tree_size = {"n_leaves": _n_leaves, "n_nodes": _n_nodes, "n_trees_in_set": n_trees}
        except Exception as e:
            tree_size = {"error": str(e)}
            print(f"    Warning: could not get tree paths: {e}")

        with open(results_dir / "licketyresplit_binarized_tree_size.json", "w") as f:
            json.dump(tree_size, f)

        with open(results_dir / "licketyresplit_binarized_results.txt", "w") as f:
            f.write(f"\nAccuracy: {acc}")
            f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, test_preds)}")
            f.write(f"\nClassification Report:\n{classification_report(y_test, test_preds)}")
            f.write(f"\nEnsemble Accuracy: {ensemble_acc}")
            f.write(f"\nLicketyRESPLIT completed in {duration:.2f} seconds with {n_trees} trees")
            if "error" not in tree_size:
                f.write(f"\nTree Size (tree 0): {tree_size['n_leaves']} leaves, {tree_size['n_nodes']} total nodes")

total_elapsed = time.perf_counter() - overall_start
print(f"\n{'='*60}")
print(f"Sweep complete. {total_runs} runs finished in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
print(f"{'='*60}")
