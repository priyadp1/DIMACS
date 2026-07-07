"""
Isolated worker that fits one GOSDT model and writes its result to JSON.

Run in a subprocess by run_feature_ablation.py so that an out-of-memory
kill during GOSDT's branch-and-bound search only takes down this worker,
not the whole ablation sweep.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from gosdt import GOSDTClassifier
from gosdt._tree import Leaf


def raw_feature_of(col_name):
    return col_name.split(" <= ")[0]


def main():
    in_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    depth_budget = int(sys.argv[3])
    reg = float(sys.argv[4])
    time_limit = int(sys.argv[5])
    max_rows = int(sys.argv[6])

    X_train = pd.read_csv(in_dir / "X_train.csv")
    X_test = pd.read_csv(in_dir / "X_test.csv")
    y_train = pd.read_csv(in_dir / "y_train.csv").squeeze()
    y_test = pd.read_csv(in_dir / "y_test.csv").squeeze()
    warm_labels = pd.read_csv(in_dir / "warm_labels.csv").squeeze().to_numpy()

    if len(X_train) > max_rows:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train), max_rows, replace=False)
        X_train = X_train.iloc[idx].reset_index(drop=True)
        y_train = y_train.iloc[idx].reset_index(drop=True)
        warm_labels = warm_labels[idx]

    clf = GOSDTClassifier(
        regularization=reg,
        similar_support=False,
        time_limit=time_limit,
        depth_budget=depth_budget,
        verbose=False,
    )
    warm_classes = set(pd.Series(warm_labels).unique())
    y_classes = set(y_train.unique())
    if warm_classes == y_classes:
        clf.fit(X_train, y_train, y_ref=warm_labels)
    else:
        clf.fit(X_train, y_train)

    test_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, test_pred)

    tree = clf.trees_[0]
    split_features = set()

    def walk(node):
        if isinstance(node, Leaf):
            return
        split_features.add(raw_feature_of(tree.features[node.feature]))
        walk(node.left_child)
        walk(node.right_child)

    walk(tree.tree)

    result = {
        "accuracy": acc,
        "test_pred": [int(p) for p in test_pred],
        "split_features": sorted(split_features),
    }
    out_path.write_text(json.dumps(result))


if __name__ == "__main__":
    main()
