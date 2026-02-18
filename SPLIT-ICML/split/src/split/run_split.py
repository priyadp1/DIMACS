import os
import time
from split import SPLIT
import pandas as pd
from sklearn.metrics import accuracy_score , classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from pathlib import Path

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent

BASEDIR = current
DATAPATH = BASEDIR / "datasets" / "Mine" / "breast_cancer_data.csv"
results_dir = BASEDIR / "model_results"
os.makedirs(results_dir, exist_ok=True)
import json as _json
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
    _drop_cols  = ['id', 'diagnosis']
    _label_map  = {'M': 1, 'B': 0}

print("Loading dataset...")
lookahead_depth = 2
depth_buget = 5
regularization = 0.01

dataset = pd.read_csv(DATAPATH) 
dataset = dataset.dropna(axis=1, how="all")

print("Mapping diagnosis to binary...")
if _label_map:
    dataset[_target_col] = dataset[_target_col].map(_label_map)
print("Preparing features and labels...")

X = dataset.drop(columns=_drop_cols)
Y = dataset[_target_col]
print("X shape:" , X.shape)
print("Y dist:\n" , Y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X,Y,test_size=0.2, random_state=42, stratify=Y
)

# y should correspond to a binary class label. 
print("Initializing SPLIT...")
model = SPLIT(lookahead_depth_budget=lookahead_depth, reg=regularization, full_depth_budget=depth_buget, verbose=True, binarize=True,time_limit=100)
# set binarize = True if dataset is not binarized.
print("Starting training...")
start = time.perf_counter()
model.fit(X_train,y_train)
duration = time.perf_counter() - start

print("Done training")
print("Making predicitions...")
y_pred = model.predict(X_test)

print("\nAccuracy: " , accuracy_score(y_test,y_pred))
print("\nConfusion Matrix: " , confusion_matrix(y_test, y_pred))
print("\nClassification Report: " , classification_report(y_test, y_pred))
print("Tree structure:")
print(model.tree)
def _count_tree_nodes(node):
    """Return (total_nodes, n_leaves) for a gosdt/split Node/Leaf tree."""
    if hasattr(node, 'left_child'):
        l_n, l_l = _count_tree_nodes(node.left_child)
        r_n, r_l = _count_tree_nodes(node.right_child)
        return 1 + l_n + r_n, l_l + r_l
    return 1, 1

try:
    _root = model.tree if model.tree is not None else model.clf.trees_[0].tree
    _n_nodes, _n_leaves = _count_tree_nodes(_root)
    _tree_size = {"n_leaves": _n_leaves, "n_nodes": _n_nodes}
except Exception as _e:
    _tree_size = {"error": str(_e)}
with open(results_dir / "split_tree_size.json", "w") as f:
    _json.dump(_tree_size, f)

with open(results_dir / "split_results.txt", "w") as f:
    f.write(f"\nAccuracy: {accuracy_score(y_test, y_pred)}")
    f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    f.write(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    f.write(f"\nSPLIT completed in {duration:.2f} seconds")
    if "error" not in _tree_size:
        f.write(f"\nTree Size: {_tree_size['n_leaves']} leaves, {_tree_size['n_nodes']} total nodes")
    else:
        f.write(f"\nTree Size: Error - {_tree_size['error']}")