import os
import time
import json as _json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from resplit.model.treefarms import TREEFARMS

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent

BASEDIR = current
DATAPATH = BASEDIR / "datasets" / "Mine" / "breast_cancer_data.csv"
results_dir = BASEDIR / "model_results"
os.makedirs(results_dir, exist_ok=True)

_cfg_file = BASEDIR / "_run_config.json"
if _cfg_file.exists():
    with open(_cfg_file) as _f:
        _cfg = _json.load(_f)
    DATAPATH    = Path(_cfg['dataset_path'])
    results_dir = Path(_cfg['results_dir'])
    os.makedirs(results_dir, exist_ok=True)
    _target_col = _cfg['target_column']
    _drop_cols  = _cfg['drop_columns']
    _label_map  = _cfg.get('label_map')
else:
    _target_col = 'diagnosis'
    _drop_cols  = ['diagnosis']
    _label_map  = {'M': 1, 'B': 0}

# Parameters
REGULARIZATION = 0.01
RASHOMON_MULT  = 0.05
DEPTH_BUDGET   = 5

print("Loading dataset...")
df = pd.read_csv(DATAPATH)
df = df.dropna(axis=1, how="all")

if _label_map:
    df[_target_col] = df[_target_col].map(_label_map)

X = df.drop(columns=_drop_cols)
Y = df[_target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

config = {
    "regularization": REGULARIZATION,
    "rashomon_bound_multiplier": RASHOMON_MULT,
    "depth_budget": DEPTH_BUDGET,
    "verbose": False,
}

print("\nTraining TREEFARMS...")
model = TREEFARMS(config)
start = time.perf_counter()
model.fit(X_train, y_train)
duration = time.perf_counter() - start

tree = model[0]
y_pred = tree.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {acc}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
print(f"\nTREEFARMS completed in {duration:.2f} seconds with {model.get_tree_count()} trees in Rashomon set")

def _count_dict_tree(source):
    if "prediction" in source:
        return 1, 1
    l_n, l_l = _count_dict_tree(source["true"])
    r_n, r_l = _count_dict_tree(source["false"])
    return 1 + l_n + r_n, l_l + r_l

try:
    _n_nodes, _n_leaves = _count_dict_tree(vars(model[0])['source'])
    _tree_size = {"n_leaves": _n_leaves, "n_nodes": _n_nodes, "n_trees_in_set": model.get_tree_count()}
except Exception as _e:
    _tree_size = {"error": str(_e)}

with open(results_dir / "treefarms_tree_size.json", "w") as f:
    _json.dump(_tree_size, f)

with open(results_dir / "treefarms_results.txt", "w") as f:
    f.write(f"\nAccuracy: {acc}")
    f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    f.write(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    f.write(f"\nTREEFARMS completed in {duration:.2f} seconds with {model.get_tree_count()} trees")
    if "error" not in _tree_size:
        f.write(f"\nTree Size (tree 0): {_tree_size['n_leaves']} leaves, {_tree_size['n_nodes']} total nodes")
    else:
        f.write(f"\nTree Size: Error - {_tree_size['error']}")
