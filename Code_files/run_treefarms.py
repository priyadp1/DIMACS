import sys
import time
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from resplit.model.treefarms import TREEFARMS

REGULARIZATION = 0.01
RASHOMON_MULT  = 0.05
DEPTH_BUDGET   = 5

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent
BASEDIR = current

TGB_DIR     = BASEDIR / "TGB_Variables_Feature_Importance"
RESULTS_DIR = BASEDIR / "benchmarks_TGB_results_all"

dataset_name = sys.argv[1]
param_tag    = sys.argv[2]
tgb_dir = TGB_DIR / dataset_name / param_tag
out_dir = RESULTS_DIR / dataset_name / param_tag
out_dir.mkdir(parents=True, exist_ok=True)

if (out_dir / "treefarms_results.txt").exists():
    print(f"  [TREEFARMS] Skipping {dataset_name}/{param_tag} — results already exist.")
    sys.exit(0)

X_train = pd.read_csv(tgb_dir / "X_train_guessed.csv")
X_test  = pd.read_csv(tgb_dir / "X_test_guessed.csv")
y_train = pd.read_csv(tgb_dir / "y_train.csv").squeeze()
y_test  = pd.read_csv(tgb_dir / "y_test.csv").squeeze()

config = {
    "regularization": REGULARIZATION,
    "rashomon_bound_multiplier": RASHOMON_MULT,
    "depth_budget": DEPTH_BUDGET,
    "verbose": False,
}

print(f"  [TREEFARMS] Training on {dataset_name}/{param_tag}...")
model = TREEFARMS(config)
start = time.perf_counter()
model.fit(X_train, y_train)
duration = time.perf_counter() - start

tree = model[0]
y_pred = tree.predict(X_test)
acc = accuracy_score(y_test, y_pred)

def _count_dict_tree(source):
    if "prediction" in source:
        return 1, 1
    l_n, l_l = _count_dict_tree(source["true"])
    r_n, r_l = _count_dict_tree(source["false"])
    return 1 + l_n + r_n, l_l + r_l

try:
    _n_nodes, _n_leaves = _count_dict_tree(vars(model[0])['source'])
    tree_size = {"n_leaves": _n_leaves, "n_nodes": _n_nodes, "n_trees_in_set": model.get_tree_count()}
except Exception as e:
    tree_size = {"error": str(e)}

with open(out_dir / "treefarms_tree_size.json", "w") as f:
    json.dump(tree_size, f)

with open(out_dir / "treefarms_results.txt", "w") as f:
    f.write(f"Accuracy: {acc}")
    f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    f.write(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    f.write(f"\nTREEFARMS completed in {duration:.2f} seconds")
    f.write(f"\nRashomon set size: {model.get_tree_count()}")
    if "error" not in tree_size:
        f.write(f"\nTree Size (tree 0): {tree_size['n_leaves']} leaves, {tree_size['n_nodes']} total nodes")
    else:
        f.write(f"\nTree Size: Error - {tree_size['error']}")

print(f"  [TREEFARMS] Accuracy: {acc:.4f} | Time: {duration:.2f}s | Trees: {model.get_tree_count()}")
