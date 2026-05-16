import sys
import time
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from resplit import RESPLIT

resplit_regularization         = 0.01
resplit_rashomon_bound_multiplier = 0.05
resplit_depth_budget           = 5
resplit_cart_lookahead_depth   = 3

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

if (out_dir / "resplit_results.txt").exists():
    print(f"  [RESPLIT] Skipping {dataset_name} — results already exist.")
    sys.exit(0)

X_train = pd.read_csv(tgb_dir / "X_train_guessed.csv")
X_test  = pd.read_csv(tgb_dir / "X_test_guessed.csv")
y_train = pd.read_csv(tgb_dir / "y_train.csv").squeeze()
y_test  = pd.read_csv(tgb_dir / "y_test.csv").squeeze()

model = RESPLIT({
    "regularization": resplit_regularization,
    "rashomon_bound_multiplier": resplit_rashomon_bound_multiplier,
    "depth_budget": resplit_depth_budget,
    "cart_lookahead_depth": resplit_cart_lookahead_depth,
    "verbose": True,
}, fill_tree='treefarms')

start = time.perf_counter()
model.fit(X_train, y_train)
resplit_duration = time.perf_counter() - start

n_trees = len(model)
if n_trees == 0:
    print(f"  [RESPLIT] WARNING: Empty Rashomon set — no trees found.")
    with open(out_dir / "resplit_results.txt", "w") as f:
        f.write(f"\nRESPLIT completed in {resplit_duration:.2f} seconds with 0 trees (empty Rashomon set)")
    sys.exit(0)

y_pred = model.predict(X_test, idx=0)

with open(out_dir / "resplit_first_tree.txt", "w", encoding="utf-8") as fh:
    try:
        fh.write(str(model[0]))
    except Exception as e:
        fh.write(f"Tree unavailable: {e}\n")

with open(out_dir / "resplit_results.txt", "w") as f:
    f.write(f"\nAccuracy: {accuracy_score(y_test, y_pred)}")
    f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    f.write(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    f.write(f"\nRESPLIT completed in {resplit_duration:.2f} seconds with {n_trees} trees")

print(f"  [RESPLIT] Accuracy: {accuracy_score(y_test, y_pred):.4f} | Time: {resplit_duration:.2f}s | Trees: {n_trees}")
