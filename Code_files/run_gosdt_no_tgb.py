import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from gosdt import GOSDTClassifier

GOSDT_REG         = 0.001
GOSDT_DEPTH       = 6
GOSDT_TIME_LIMIT  = 60
GOSDT_SIM_SUPPORT = False
GOSDT_MAX_FEATURES = 50
GOSDT_MAX_ROWS     = 10000

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent
BASEDIR = current

TMP_DIR     = BASEDIR / "tmp_splits_no_tgb"
RESULTS_DIR = BASEDIR / "benchmarks_no_TGB_results"

dataset_name = sys.argv[1]
tmp_dir = TMP_DIR / dataset_name
out_dir = RESULTS_DIR / dataset_name
out_dir.mkdir(parents=True, exist_ok=True)

if (out_dir / "gosdt_results.txt").exists():
    print(f"  [GOSDT] Skipping {dataset_name} — results already exist.")
    sys.exit(0)

X_train = pd.read_csv(tmp_dir / "X_train.csv")
X_test  = pd.read_csv(tmp_dir / "X_test.csv")
y_train = pd.read_csv(tmp_dir / "y_train.csv").squeeze()
y_test  = pd.read_csv(tmp_dir / "y_test.csv").squeeze()

X_train_gosdt = X_train.copy()
y_train_gosdt = y_train.copy()

# Subsample rows if needed
if len(X_train_gosdt) > GOSDT_MAX_ROWS:
    idx = np.random.default_rng(42).choice(len(X_train_gosdt), GOSDT_MAX_ROWS, replace=False)
    X_train_gosdt = X_train_gosdt.iloc[idx]
    y_train_gosdt = y_train_gosdt.iloc[idx]
    print(f"  [GOSDT] Subsampled to {GOSDT_MAX_ROWS} rows.")

# Reduce features if needed
if X_train_gosdt.shape[1] > GOSDT_MAX_FEATURES:
    _imp = pd.Series(
        XGBClassifier(max_depth=3, n_estimators=25, learning_rate=0.1,
                      eval_metric="logloss", random_state=42)
        .fit(X_train_gosdt.rename(columns=lambda c: c.replace("[", "{").replace("]", "}").replace("<", "lt")),
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
y_pred = clf.predict(X_test[X_train_gosdt.columns])


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

with open(out_dir / "gosdt_first_tree.txt", "w", encoding="utf-8") as fh:
    try:
        fh.write(str(clf.trees_[0]))
    except Exception as e:
        fh.write(f"Tree unavailable: {e}\n")

with open(out_dir / "gosdt_results.txt", "w") as f:
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    f.write(f"\nTraining Accuracy: {clf.score(X_train_gosdt, y_train_gosdt)}")
    f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    f.write(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    f.write(f"\nGOSDT completed in {clf.result_.time:.2f} seconds")
    if "error" not in tree_size:
        f.write(f"\nTree Size: {tree_size['n_leaves']} leaves, {tree_size['n_nodes']} total nodes")
    else:
        f.write(f"\nTree Size: Error - {tree_size['error']}")

print(f"  [GOSDT] Accuracy: {accuracy_score(y_test, y_pred):.4f} | Time: {clf.result_.time:.2f}s")
