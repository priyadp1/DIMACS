import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent
BASEDIR = current

TGB_DIR = BASEDIR / "TGB_Variables_Feature_Importance"
RESULTS_DIR = BASEDIR / "benchmarks_TGB_results_all"

# Fixed seed for the max_trees sub-sample drawn from the cached (already
# uniformly sampled) praxis_sampled_trees.json trees, so results stay
# reproducible across runs.
SUBSAMPLE_SEED = 0

# module/model/lib/tree_classifier.py only depends on numpy/sklearn/scipy, so we can
# reuse rashomon-framework's sklearn-compatible tree class without needing PRAXIS,
# treefarms, or gosdt importable.
sys.path.insert(0, str(BASEDIR / "rashomon-framework"))
from module.model.lib.tree_classifier import TreeClassifier  # noqa: E402


def discover_all_datasets():
    """List every dataset with at least one cached PRAXIS Rashomon set."""
    if not TGB_DIR.exists():
        return []
    return sorted(p.name for p in TGB_DIR.iterdir() if p.is_dir() and discover_param_tags(p.name))


def discover_param_tags(dataset_name):
    """List param_tag subdirectories that have a cached praxis_sampled_trees.json."""
    dataset_tgb_dir = TGB_DIR / dataset_name
    if not dataset_tgb_dir.exists():
        return []
    tags = []
    for tgb_dir in sorted(p for p in dataset_tgb_dir.iterdir() if p.is_dir()):
        if (RESULTS_DIR / dataset_name / tgb_dir.name / "praxis_sampled_trees.json").exists():
            tags.append(tgb_dir.name)
    return tags


def load_test_data(dataset_name, param_tag):
    """Load the TGB-guessed test split for a dataset/param_tag combo.

    Returns:
        X_test (np.ndarray[uint8]), y_test (np.ndarray[int]), feature_names (list[str])
    """
    tgb_dir = TGB_DIR / dataset_name / param_tag
    X_test = pd.read_csv(tgb_dir / "X_test_guessed.csv")
    y_test = pd.read_csv(tgb_dir / "y_test.csv").squeeze().to_numpy().astype(int)
    return X_test.to_numpy().astype(np.uint8), y_test, list(X_test.columns)


def load_train_data(dataset_name, param_tag):
    """Load the TGB-guessed train split for a dataset/param_tag combo.

    Returns:
        X_train (np.ndarray[uint8]), y_train (np.ndarray[int]), feature_names (list[str])
    """
    tgb_dir = TGB_DIR / dataset_name / param_tag
    X_train = pd.read_csv(tgb_dir / "X_train_guessed.csv")
    y_train = pd.read_csv(tgb_dir / "y_train.csv").squeeze().to_numpy().astype(int)
    return X_train.to_numpy().astype(np.uint8), y_train, list(X_train.columns)


def load_sampled_trees(dataset_name, param_tag, n_features, max_trees=None, classes=(0, 1)):
    """Load praxis_sampled_trees.json and rebuild each tree as a TreeClassifier.

    Args:
        n_features (int): Number of columns in the corresponding X_test_guessed.csv.
            Must be passed explicitly (rather than inferred from the tree) since a
            tree may not split on the highest-indexed feature.

    Returns:
        dict: tree_index -> TreeClassifier
        dict: metadata (n_total_trees, n_sampled_trees)
    """
    path = RESULTS_DIR / dataset_name / param_tag / "praxis_sampled_trees.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    trees = data["trees"]
    if max_trees is not None and len(trees) > max_trees:
        trees = random.Random(SUBSAMPLE_SEED).sample(trees, max_trees)

    tree_classifiers = {}
    for entry in trees:
        idx = entry["tree_index"]
        tree_classifiers[idx] = build_tree_classifier_from_path_strs(
            entry["paths"], entry["predictions"], n_features=n_features, classes=classes
        )

    meta = {"n_total_trees": data.get("n_total_trees"), "n_sampled_trees": data.get("n_sampled_trees")}
    return tree_classifiers, meta


def load_sampled_tree_leaves(dataset_name, param_tag, max_trees=None):
    """Load praxis_sampled_trees.json as raw per-leaf literal lists (no TreeClassifier).

    Used for the Hamming/L0 robustness attack, which only needs each leaf's path
    literals (feature, required_bit) and predicted class -- reconstructing a full
    tree object isn't necessary.

    Returns:
        dict: tree_index -> list of {"literals": [(feature, bit), ...], "pred": int}
        dict: metadata (n_total_trees, n_sampled_trees)
    """
    path = RESULTS_DIR / dataset_name / param_tag / "praxis_sampled_trees.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    trees = data["trees"]
    if max_trees is not None and len(trees) > max_trees:
        trees = random.Random(SUBSAMPLE_SEED).sample(trees, max_trees)

    tree_leaves = {}
    for entry in trees:
        idx = entry["tree_index"]
        tree_leaves[idx] = [
            {"literals": _parse_path_str(p), "pred": int(pred)}
            for p, pred in zip(entry["paths"], entry["predictions"])
        ]

    meta = {"n_total_trees": data.get("n_total_trees"), "n_sampled_trees": data.get("n_sampled_trees")}
    return tree_leaves, meta


def _parse_path_str(path_str):
    """Parse a PRAXIS path string like "[+56, -3]" into [(feature, bit), ...]."""
    body = path_str.strip()[1:-1].strip()
    if not body:
        return []
    literals = []
    for tok in body.split(","):
        tok = tok.strip()
        f = int(tok[1:])
        bit = 1 if tok[0] == "+" else 0
        literals.append((f, bit))
    return literals


def build_tree_classifier_from_path_strs(paths_str, preds, n_features, classes=(0, 1)) -> TreeClassifier:
    """Rebuild a sklearn-tree-compatible TreeClassifier from PRAXIS's leaf paths.

    Each entry in `paths_str` is a root-to-leaf conjunction of binary-feature
    literals (e.g. "[+56, -3]" means feature 56 == 1 AND feature 3 == 0 along
    this path). Since `paths_str`/`preds` enumerate every leaf of one tree,
    they fully determine a trie, which is rebuilt here into sklearn-style
    children/feature/threshold/value arrays (threshold fixed at 0.5 for every
    binary split: feature==0 -> left child, feature==1 -> right child).
    """
    root = {}
    for path_str, pred in zip(paths_str, preds):
        node = root
        for f, bit in _parse_path_str(path_str):
            node.setdefault("feature", f)
            node = node.setdefault(bit, {})
        node["leaf"] = True
        node["pred"] = int(pred)

    n_classes = len(classes)
    ch_left, ch_right, feat, thr, leaf_vecs = [], [], [], [], []

    def build(node) -> int:
        i = len(ch_left)
        ch_left.append(-1)
        ch_right.append(-1)
        feat.append(-2)
        thr.append(-2.0)
        leaf_vecs.append(None)

        if node.get("leaf"):
            v = np.zeros(n_classes, dtype=float)
            v[node["pred"]] = 1.0
            leaf_vecs[i] = v
        else:
            feat[i] = node["feature"]
            thr[i] = 0.5
            left_i = build(node[0])
            right_i = build(node[1])
            ch_left[i] = left_i
            ch_right[i] = right_i
        return i

    build(root)

    n = len(ch_left)
    ch_left = np.asarray(ch_left, dtype=np.int32)
    ch_right = np.asarray(ch_right, dtype=np.int32)
    feat = np.asarray(feat, dtype=np.int32)
    thr = np.asarray(thr, dtype=float)

    value = np.zeros((n, 1, n_classes), dtype=float)
    for i in range(n):
        if ch_left[i] == -1:
            value[i, 0, :] = leaf_vecs[i]
    for i in range(n - 1, -1, -1):
        if ch_left[i] != -1:
            value[i, 0, :] = value[ch_left[i], 0, :] + value[ch_right[i], 0, :]

    return TreeClassifier(
        children_left=ch_left,
        children_right=ch_right,
        feature=feat,
        threshold=thr,
        value=value,
        classes=classes,
        n_features_in=n_features,
    )
