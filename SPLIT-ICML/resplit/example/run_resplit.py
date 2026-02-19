import time
import pandas as pd
from resplit.model.treefarms import TREEFARMS
from resplit import RESPLIT
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report , confusion_matrix
from pathlib import Path
import os
current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent

BASEDIR = current
DATAPATH = BASEDIR / "datasets" / "Mine" / "breast_cancer_data.csv"
path = DATAPATH
results_dir = BASEDIR/"model_results"
os.makedirs(results_dir , exist_ok=True)
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
print(f"Loading dataset from: {path}")
df = pd.read_csv(path)
print("Mapping diagnosis to binary...")
if _label_map:
    df[_target_col] = df[_target_col].map(_label_map)

print("Preparing features and labels...")
df = df.dropna(axis=1, how="all")
X = df.drop(columns=_drop_cols)
Y = df[_target_col]
print("X shape:" , X.shape)
print("Y dist:\n" , Y.value_counts())
 
X_train, X_test, y_train, y_test = train_test_split(
    X,Y,test_size=0.2, random_state=42, stratify=Y
)
regularization = 0.01
rashomon_bound_multiplier = 0.01
depth_budget = 5
cart_lookahead_depth = 3
print("\n Running RESPLIT...")
config = {
        "regularization": regularization,
        "rashomon_bound_multiplier": rashomon_bound_multiplier,
        "depth_budget": depth_budget,
        "cart_lookahead_depth": cart_lookahead_depth,
        "verbose": True
    }
model = RESPLIT(config, fill_tree='treefarms')
start = time.perf_counter()
model.fit(X_train, y_train)
duration = time.perf_counter() - start
print("Done training RESPLIT")
print("Making predicitions...")
y_pred = model.predict(X_test, idx=0)
def _count_tree_nodes(node):
    """Return (total_nodes, n_leaves) for a gosdt/split Node/Leaf tree."""
    if hasattr(node, 'left_child'):
        l_n, l_l = _count_tree_nodes(node.left_child)
        r_n, r_l = _count_tree_nodes(node.right_child)
        return 1 + l_n + r_n, l_l + r_l
    return 1, 1

try:
    _n_nodes, _n_leaves = _count_tree_nodes(model[0])
    _tree_size_resplit = {"n_leaves": _n_leaves, "n_nodes": _n_nodes, "n_trees_in_set": len(model)}
except Exception as _e:
    _tree_size_resplit = {"error": str(_e)}
with open(results_dir / "resplit_tree_size.json", "w") as f:
    _json.dump(_tree_size_resplit, f)

with open(results_dir / f"{depth_budget}_{regularization}_{rashomon_bound_multiplier}" / "resplit_results.txt", "w") as f:
    f.write(f"\nAccuracy: {accuracy_score(y_test, y_pred)}")
    f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    f.write(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    f.write(f"\nRESPLIT completed in {duration:.2f} seconds with {len(model)} trees")
    if "error" not in _tree_size_resplit:
        f.write(f"\nTree Size (tree 0): {_tree_size_resplit['n_leaves']} leaves, {_tree_size_resplit['n_nodes']} total nodes")
print("\nAccuracy: " , accuracy_score(y_test,y_pred))
print("\nConfusion Matrix: " , confusion_matrix(y_test, y_pred))
print("\nClassification Report: " , classification_report(y_test, y_pred))
print(f" RESPLIT completed in {duration:.2f} seconds with {len(model)} trees")


print("\n Running TREEFARMS...")
reg_treefarms = 0.01
rashomon_mult_treefarms = 0.01
depth_budget_treefarms = 3
config = {
        "regularization": reg_treefarms,
        "rashomon_bound_multiplier": rashomon_mult_treefarms,
        "depth_budget": depth_budget_treefarms,
        "verbose": True
    }
model = TREEFARMS(config)
start = time.perf_counter()
model.fit(X_train, y_train)
duration = time.perf_counter() - start
print("Done training Treefarms")
print("Making predicitions...")
tree = model[0]
y_pred = tree.predict(X_test)
print("\nAccuracy: " , accuracy_score(y_test,y_pred))
print("\nConfusion Matrix: " , confusion_matrix(y_test, y_pred))
print("\nClassification Report: " , classification_report(y_test, y_pred))
print(f" TREEFARMS completed in {duration:.2f} seconds with {model.get_tree_count()} trees")
def _count_dict_tree(source):
    """Return (total_nodes, n_leaves) for a TreeClassifier dict tree (keys: 'true'/'false')."""
    if "prediction" in source:
        return 1, 1
    l_n, l_l = _count_dict_tree(source["true"])
    r_n, r_l = _count_dict_tree(source["false"])
    return 1 + l_n + r_n, l_l + r_l

try:
    _n_nodes_tf, _n_leaves_tf = _count_dict_tree(vars(model[0])['source'])
    _tree_size_tf = {"n_leaves": _n_leaves_tf, "n_nodes": _n_nodes_tf, "n_trees_in_set": model.get_tree_count()}
except Exception as _e:
    _tree_size_tf = {"error": str(_e)}
with open(results_dir / f"{depth_budget_treefarms}_{reg_treefarms}_{rashomon_mult_treefarms}" / "treefarms_tree_size.json", "w") as f:
    _json.dump(_tree_size_tf, f)

with open(results_dir / f"{depth_budget_treefarms}_{reg_treefarms}_{rashomon_mult_treefarms}" / "treefarms_results.txt", "w") as f:
    f.write(f"\nAccuracy: {accuracy_score(y_test, y_pred)}")
    f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    f.write(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    f.write(f"\nTREEFARMS completed in {duration:.2f} seconds with {model.get_tree_count()} trees")
    if "error" not in _tree_size_tf:
        f.write(f"\nTree Size (tree 0): {_tree_size_tf['n_leaves']} leaves, {_tree_size_tf['n_nodes']} total nodes")


