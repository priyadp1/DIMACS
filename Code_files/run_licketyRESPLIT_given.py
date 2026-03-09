import os
import time
import json
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
results_dir = BASEDIR / "model_results"
os.makedirs(results_dir, exist_ok=True)

_cfg_file = BASEDIR / "Code_files" / "_run_config.json"
if _cfg_file.exists():
    with open(_cfg_file) as _f:
        _cfg = json.load(_f)
    results_dir = Path(_cfg['results_dir'])
    os.makedirs(results_dir, exist_ok=True)
    _target_col = _cfg['target_column']
    _drop_cols  = _cfg['drop_columns']
    _label_map  = _cfg.get('label_map')
    if 'train_path' in _cfg:
        train_df = pd.read_csv(_cfg['train_path']).dropna(axis=1, how="all")
        test_df  = pd.read_csv(_cfg['test_path']).dropna(axis=1, how="all")
        if _label_map:
            train_df[_target_col] = train_df[_target_col].map(_label_map)
            test_df[_target_col]  = test_df[_target_col].map(_label_map)
        X_train = train_df.drop(columns=_drop_cols)
        y_train = train_df[_target_col]
        X_test  = test_df.drop(columns=_drop_cols)
        y_test  = test_df[_target_col]
    else:
        df = pd.read_csv(Path(_cfg['dataset_path'])).dropna(axis=1, how="all")
        if _label_map:
            df[_target_col] = df[_target_col].map(_label_map)
        X = df.drop(columns=_drop_cols)
        Y = df[_target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42, stratify=Y
        )
else:
    _target_col = 'class'
    _drop_cols  = ['class']
    _label_map  = None
    df = pd.read_csv(BASEDIR / "datasets" / "Mine" / "spambase.csv").dropna(axis=1, how="all")
    X = df.drop(columns=_drop_cols)
    Y = df[_target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

depth_budget = 3
lambda_reg = 0.003
rashomon_mult = 0.05

# Step 1: Guess Thresholds
enc = ThresholdGuessBinarizer(n_estimators=50, max_depth=2, random_state=42)
enc.set_output(transform="pandas")
X_train_guessed = enc.fit_transform(X_train, y_train)
X_test_guessed = enc.transform(X_test)
print(f"After guessing, X train shape:{X_train_guessed.shape}, X test shape:{X_test_guessed.shape}")
print(f"train set column names == test set column names: {list(X_train_guessed.columns)==list(X_test_guessed.columns)}")

model = LicketyRESPLIT()

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

print("Done training.")
print("Minimum objective:", model.get_min_objective())
print("Rashomon set size:", model.count_trees())

tree_idx = 0
test_preds = model.get_predictions(tree_idx, X_test_guessed)
print("Test Accuracy:", accuracy_score(y_test, test_preds))
print(classification_report(y_test, test_preds))

n_samples = X_test_guessed.shape[0]
votes = np.zeros(n_samples, dtype=np.int32)
n_trees = model.count_trees()
for tree_idx in range(n_trees):
    preds = model.get_predictions(tree_idx, X_test_guessed)
    votes += preds
majority_vote = (votes >= (n_trees / 2)).astype(int)
ensemble_acc = accuracy_score(y_test, majority_vote)
print("Ensemble Accuracy:", ensemble_acc)

try:
    tree_idx = 0
    _paths, _ = model.get_tree_paths(tree_idx)
    _n_leaves = len(_paths)
    _n_nodes = 2 * _n_leaves - 1
    tree_size = {"n_leaves": _n_leaves, "n_nodes": _n_nodes, "n_trees_in_set": model.count_trees()}
except Exception as e:
    tree_size = {"error": str(e)}

with open(results_dir / "licketyresplit_binarized_tree_size.json", "w") as f:
    json.dump(tree_size, f)

with open(results_dir / "licketyresplit_binarized_results.txt", "w") as f:
    f.write(f"\nAccuracy: {accuracy_score(y_test, test_preds)}")
    f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, test_preds)}")
    f.write(f"\nClassification Report:\n{classification_report(y_test, test_preds)}")
    f.write(f"\nEnsemble Accuracy: {ensemble_acc}")
    f.write(f"\nLicketyRESPLIT completed in {duration:.2f} seconds with {model.count_trees()} trees")
    if "error" not in tree_size:
        f.write(f"\nTree Size (tree 0): {tree_size['n_leaves']} leaves, {tree_size['n_nodes']} total nodes")
