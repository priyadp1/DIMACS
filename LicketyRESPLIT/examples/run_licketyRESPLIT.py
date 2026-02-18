import os
import time
import numpy as np
import pandas as pd
from licketyresplit import LicketyRESPLIT
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from pathlib import Path
current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent

BASEDIR = current
DATAPATH = BASEDIR / "datasets" / "Mine" / "breast_cancer_data.csv"
path = DATAPATH
results_dir = BASEDIR / "model_results"
os.makedirs(results_dir, exist_ok=True)
import json as _json
_cfg_file = BASEDIR / "_run_config.json"
if _cfg_file.exists():
    with open(_cfg_file) as _f:
        _cfg = _json.load(_f)
    DATAPATH    = Path(_cfg['dataset_path'])
    path        = DATAPATH
    results_dir = Path(_cfg['results_dir'])
    os.makedirs(results_dir, exist_ok=True)
    _target_col = _cfg['target_column']
    _drop_cols  = _cfg['drop_columns']
    _label_map  = _cfg.get('label_map')
else:
    _target_col = 'diagnosis'
    _drop_cols  = ['id', 'diagnosis']
    _label_map  = {'M': 1, 'B': 0}
df = pd.read_csv(path)
df = df.dropna(axis=1, how="all")
if _label_map:
    df[_target_col] = df[_target_col].map(_label_map)
X = df.drop(columns=_drop_cols)
Y = df[_target_col]


X = X.values
Y = Y.values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# Train model
model = LicketyRESPLIT()

start = time.perf_counter()
model.fit(
    X_train,
    y_train,
    lambda_reg=0.01,
    depth_budget=5,
    rashomon_mult=0.05,
    multiplicative_slack=0,
    key_mode="hash",
    trie_cache_enabled=False,
    lookahead_k=1,
)
duration = time.perf_counter() - start

print("Done training.")
print("Minimum objective:", model.get_min_objective())
print("Rashomon set size:", model.count_trees())

# Single best tree
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
    _n_leaves_lr = len(_paths)
    _n_nodes_lr = 2 * _n_leaves_lr - 1  # binary tree: internal nodes = leaves - 1
    _tree_size_lr = {"n_leaves": _n_leaves_lr, "n_nodes": _n_nodes_lr, "n_trees_in_set": model.count_trees()}
except Exception as _e:
    _tree_size_lr = {"error": str(_e)}
with open(results_dir / "licketyresplit_tree_size.json", "w") as f:
    _json.dump(_tree_size_lr, f)

with open(results_dir / "licketyresplit_results.txt", "w") as f:
    f.write(f"\nAccuracy: {accuracy_score(y_test, test_preds)}")
    f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, test_preds)}")
    f.write(f"\nClassification Report:\n{classification_report(y_test, test_preds)}")
    f.write(f"\nEnsemble Accuracy: {ensemble_acc}")
    f.write(f"\nLicketyRESPLIT completed in {duration:.2f} seconds with {model.count_trees()} trees")
    if "error" not in _tree_size_lr:
        f.write(f"\nTree Size (tree 0): {_tree_size_lr['n_leaves']} leaves, {_tree_size_lr['n_nodes']} total nodes")

