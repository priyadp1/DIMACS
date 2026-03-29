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
results_dir = BASEDIR / "model_results" / "diabetic_data"
os.makedirs(results_dir, exist_ok=True)
_target_col = 'readmitted'
_drop_cols  = ['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty', 'max_glu_serum', 'A1Cresult']
_label_map  = None
df = pd.read_csv(BASEDIR / "datasets" / "Mine" / "diabetic_data.csv").dropna(axis=1, how="all")
# Drop unwanted columns first, then encode remaining string columns
df = df.drop(columns=_drop_cols)
string_cols = [c for c in df.select_dtypes(include="object").columns if c != _target_col]
df = pd.get_dummies(df, columns=string_cols, drop_first=True)
Y = df[_target_col]
X = df.drop(columns=[_target_col])
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
