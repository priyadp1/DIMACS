import numpy as np
import pandas as pd
from licketyresplit import LicketyRESPLIT
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from pathlib import Path
current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent

BASEDIR = current
DATAPATH = BASEDIR / "datasets" / "Mine" / "breast_cancer_data.csv"
path = DATAPATH
df = pd.read_csv(path)
df = df.dropna(axis=1, how="all")
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
X = df.drop(columns=["id", "diagnosis"])
Y = df["diagnosis"]


X = X.values
Y = Y.values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# Train model
model = LicketyRESPLIT()

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

