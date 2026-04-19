import sys
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from gosdt import GOSDTClassifier

GOSDT_REG         = 0.001
GOSDT_DEPTH       = 6
GOSDT_TIME_LIMIT  = 60
GOSDT_SIM_SUPPORT = False

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent
BASEDIR = current

TGB_DIR     = BASEDIR / "TGB_Variables"
RESULTS_DIR = BASEDIR / "benchmarks_TGB_results"

dataset_name = sys.argv[1]
param_tag    = sys.argv[2]
tgb_dir = TGB_DIR / dataset_name / param_tag
out_dir = RESULTS_DIR / dataset_name / param_tag
out_dir.mkdir(parents=True, exist_ok=True)

if (out_dir / "gosdt_results.txt").exists():
    print(f"  [GOSDT] Skipping {dataset_name} — results already exist.")
    sys.exit(0)

X_train     = pd.read_csv(tgb_dir / "X_train_guessed.csv")
X_test      = pd.read_csv(tgb_dir / "X_test_guessed.csv")
y_train     = pd.read_csv(tgb_dir / "y_train.csv").squeeze()
y_test      = pd.read_csv(tgb_dir / "y_test.csv").squeeze()
warm_labels = pd.read_csv(tgb_dir / "warm_labels.csv").squeeze().to_numpy()

clf = GOSDTClassifier(
    regularization=GOSDT_REG,
    similar_support=GOSDT_SIM_SUPPORT,
    time_limit=GOSDT_TIME_LIMIT,
    depth_budget=GOSDT_DEPTH,
    verbose=True,
)
warm_classes = set(pd.Series(warm_labels).unique())
y_classes    = set(y_train.unique())
if warm_classes == y_classes:
    clf.fit(X_train, y_train, y_ref=warm_labels)
else:
    print(f"  [GOSDT] Warning: warm_labels classes {sorted(warm_classes)} != y classes {sorted(y_classes)}, skipping y_ref")
    clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

def count_nodes(node):
    if hasattr(node, "left_child"):
        l_n, l_l = count_nodes(node.left_child)
        r_n, r_l = count_nodes(node.right_child)
        return 1 + l_n + r_n, l_l + r_l
    return 1, 1

try:
    n_nodes, n_leaves = count_nodes(clf.trees_[0].tree)
    tree_size = {"n_leaves": n_leaves, "n_nodes": n_nodes}
except Exception as e:
    tree_size = {"error": str(e)}

with open(out_dir / "gosdt_tree_size.json", "w") as f:
    json.dump(tree_size, f)

with open(out_dir / "gosdt_results.txt", "w") as f:
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    f.write(f"\nTraining Accuracy: {clf.score(X_train, y_train)}")
    f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    f.write(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    f.write(f"\nGOSDT completed in {clf.result_.time:.2f} seconds")
    if "error" not in tree_size:
        f.write(f"\nTree Size: {tree_size['n_leaves']} leaves, {tree_size['n_nodes']} total nodes")
    else:
        f.write(f"\nTree Size: Error - {tree_size['error']}")

print(f"  [GOSDT] Accuracy: {accuracy_score(y_test, y_pred):.4f} | Time: {clf.result_.time:.2f}s")
