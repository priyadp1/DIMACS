import functools
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent
BASEDIR = current

DATA_DIR = BASEDIR / "datasets" / "Mine"

# run_standard_arborenum.py and run_anytime_arborenum.py each write their raw results to
# their own top-level directory (no shared TGB-style param_tag subdirectory). "param_tag"
# below plays the same role praxis_rashomon_common.py's TGB tag does -- it selects which
# independently-run Rashomon set to load for a given dataset -- but its values are the
# keys of RESULTS_DIRS ("standard" / "anytime") rather than a TGB_Variables_Feature_Importance
# subdirectory name.
RESULTS_DIRS = {
    "standard": BASEDIR / "benchmarks_standard_arborenum_raw_results",
    "anytime": BASEDIR / "benchmarks_anytime_arborenum_raw_results",
}

# Must match the train_test_split seed in run_standard_arborenum.py / run_anytime_arborenum.py
# so the rebuilt test/train split lines up with the one the cached trees were fit on.
RANDOM_SEED = 42

# Fixed seed for the max_trees sub-sample drawn from the cached (already uniformly
# sampled) arborenum_sampled_trees.json trees, so results stay reproducible across runs.
SUBSAMPLE_SEED = 0

# ArborEnum.fit()'s default binary_unique_threshold, unmodified by either runner script:
# columns with at most this many unique values become ordinary binary features; columns
# with more are treated as continuous (see _arborenum_internal_features below).
BINARY_UNIQUE_THRESHOLD = 2

# module/model/lib/tree_classifier.py only depends on numpy/sklearn/scipy, so we can
# reuse rashomon-framework's sklearn-compatible tree class without needing PRAXIS,
# treefarms, or gosdt importable.
sys.path.insert(0, str(BASEDIR / "rashomon-framework"))
from module.model.lib.tree_classifier import TreeClassifier  # noqa: E402


# Mirrors the DATASETS dicts in run_standard_arborenum.py / run_anytime_arborenum.py (kept
# in sync manually -- update here whenever a dataset config changes there). Unlike PRAXIS's
# TGB pipeline, ArborEnum's benchmark scripts don't cache a guessed/binarized train-test
# split to disk, so this module rebuilds it itself from the raw CSVs in DATA_DIR.
DATASET_CONFIGS = {
    "bike": {
        "path": DATA_DIR / "bike.csv",
        "target_col": "cnt_binary",
        "drop_cols": ["instant", "cnt_binary"],
        "label_map": None,
    },
    "spambase": {
        "path": DATA_DIR / "spambase.csv",
        "target_col": "class",
        "drop_cols": ["class"],
        "label_map": None,
    },
    "compas": {
        "path": DATA_DIR / "compas.csv",
        "target_col": "two_year_recid",
        "drop_cols": ["two_year_recid"],
        "label_map": None,
    },
    "breast_cancer": {
        "path": DATA_DIR / "breast_cancer_data.csv",
        "target_col": "diagnosis",
        "drop_cols": ["id", "diagnosis"],
        "label_map": {"M": 1, "B": 0},
    },
    "heloc": {
        "path": DATA_DIR / "heloc_original.csv",
        "target_col": "RiskPerformance",
        "drop_cols": ["RiskPerformance"],
        "label_map": None,
    },
    "diabetes_smote": {
        "path": DATA_DIR / "diabetes_smote.csv",
        "target_col": "readmitted",
        "drop_cols": ["readmitted"],
        "label_map": None,
    },
    "german_credit": {
        "path": DATA_DIR / "german_credit_data.csv",
        "target_col": "Risk",
        "drop_cols": ["Risk"],
        "label_map": {"good": 0, "bad": 1},
        "na_strategy": "drop_columns",
        "cols_to_drop_for_na": ["Saving accounts", "Checking account"],
        "group_col": "Sex",
        "group_map": {"male": 0, "female": 1},
    },
    "german_credit_dropna": {
        "path": DATA_DIR / "german_credit_data.csv",
        "target_col": "Risk",
        "drop_cols": ["Risk"],
        "label_map": {"good": 0, "bad": 1},
        "na_strategy": "drop_rows",
        "cols_to_drop_for_na": None,
        "group_col": "Sex",
        "group_map": {"male": 0, "female": 1},
    },
}


def discover_all_datasets():
    """List every dataset with at least one cached ArborEnum Rashomon set."""
    names = set()
    for results_dir in RESULTS_DIRS.values():
        if results_dir.exists():
            names.update(p.name for p in results_dir.iterdir() if p.is_dir())
    return sorted(name for name in names if discover_param_tags(name))


def discover_param_tags(dataset_name):
    """List run tags ("standard"/"anytime") with a cached arborenum_sampled_trees.json."""
    return sorted(
        tag
        for tag, results_dir in RESULTS_DIRS.items()
        if (results_dir / dataset_name / "arborenum_sampled_trees.json").exists()
    )


@functools.lru_cache(maxsize=None)
def _load_dataset_split(dataset_name):
    """Rebuild X_train/X_test/y_train/y_test exactly as the ArborEnum benchmark scripts did.

    Reproduces run_standard_arborenum.py's/run_anytime_arborenum.py's load_dataset(): same
    CSV, same NA handling, same label_map, same get_dummies + column alignment, and the
    same train_test_split(test_size=0.2, random_state=RANDOM_SEED, stratify=y). The two
    runner scripts share this logic and RANDOM_SEED, so one reconstruction works for every
    param_tag.
    """
    cfg = DATASET_CONFIGS[dataset_name]
    df = pd.read_csv(cfg["path"])
    if cfg.get("na_strategy") == "drop_columns":
        df = df.drop(columns=cfg["cols_to_drop_for_na"])
    else:
        df = df.dropna(axis=1, how="all")
        df = df.dropna()

    if cfg["label_map"]:
        df[cfg["target_col"]] = df[cfg["target_col"]].map(cfg["label_map"])

    X = df.drop(columns=cfg["drop_cols"])
    y = df[cfg["target_col"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=False)
        X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=False)
        X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    return X_train, X_test, y_train.reset_index(drop=True), y_test.reset_index(drop=True)


def _arborenum_internal_features(X_fit, X_eval, binary_unique_threshold=BINARY_UNIQUE_THRESHOLD):
    """Reproduce ArborEnum's raw-to-internal feature expansion (its _split_binary_and_continuous
    + transform()) so cached tree paths -- which are indexed into this expanded space, not the
    raw columns -- can be evaluated against real data.

    ArborEnum.fit(X, y, ...) binarizes each raw column of X into one or more "internal feature"
    columns before searching for trees: a column with at most `binary_unique_threshold` unique
    values (in the training data) becomes one binary column per adjacent-value midpoint; a column
    with more unique values is treated as continuous and (with max_number_thresholds_per_feature=
    None, as both runner scripts use) exhaustively split at every adjacent-value midpoint too.
    Both are a deterministic function of X_fit's per-column unique values -- no ML, no randomness
    -- so they can be reproduced here without needing the compiled arborenum package installed.
    All binary-column features come before all continuous-column features in the output, exactly
    as in ArborEnum's own transform(): a tree's "[+f, -g]" path indexes into this same order.

    Args:
        X_fit: DataFrame used to derive cutpoints (X_train, matching what ArborEnum.fit() was
            called on).
        X_eval: DataFrame to transform into the internal feature space (X_train or X_test).

    Returns:
        np.ndarray[uint8] of shape (len(X_eval), n_internal_features)
        list[str] feature names, "<raw column> <= <cutpoint>", aligned with the tree feature
            indices (same naming convention PRAXIS's cached guessed CSVs use).
    """
    columns = list(X_fit.columns)
    X_fit_arr = X_fit.to_numpy(dtype=np.float64)
    X_eval_arr = X_eval.to_numpy(dtype=np.float64)

    binary_cols, binary_names = [], []
    continuous_cols, continuous_names = [], []

    for j, name in enumerate(columns):
        unique = np.unique(X_fit_arr[:, j])
        if unique.size <= 1:
            continue  # constant training column: ArborEnum drops it entirely

        midpoints = unique[:-1] + 0.5 * (unique[1:] - unique[:-1])
        eval_col = X_eval_arr[:, j]
        target_cols, target_names = (
            (binary_cols, binary_names)
            if unique.size <= binary_unique_threshold
            else (continuous_cols, continuous_names)
        )
        for cutpoint in midpoints:
            target_cols.append((eval_col <= cutpoint).astype(np.uint8))
            target_names.append(f"{name} <= {cutpoint}")

    pieces = binary_cols + continuous_cols
    feature_names = binary_names + continuous_names
    if not pieces:
        return np.empty((X_eval_arr.shape[0], 0), dtype=np.uint8), feature_names
    return np.column_stack(pieces).astype(np.uint8), feature_names


def load_test_data(dataset_name, param_tag=None):
    """Rebuild the internal-feature-space test split for an ArborEnum dataset.

    `param_tag` is accepted for call-signature parity with praxis_rashomon_common.load_test_data
    but ignored: the rebuilt train/test split and feature expansion depend only on the dataset's
    config and RANDOM_SEED, not on which run ("standard" vs "anytime") produced the cached trees.

    Returns:
        X_test (np.ndarray[uint8]), y_test (np.ndarray[int]), feature_names (list[str])
    """
    X_train, X_test, _, y_test = _load_dataset_split(dataset_name)
    X_test_internal, feature_names = _arborenum_internal_features(X_train, X_test)
    return X_test_internal, y_test.to_numpy().astype(int), feature_names


def load_train_data(dataset_name, param_tag=None):
    """Rebuild the internal-feature-space train split for an ArborEnum dataset.

    See load_test_data for why `param_tag` is accepted but ignored.

    Returns:
        X_train (np.ndarray[uint8]), y_train (np.ndarray[int]), feature_names (list[str])
    """
    X_train, _, y_train, _ = _load_dataset_split(dataset_name)
    X_train_internal, feature_names = _arborenum_internal_features(X_train, X_train)
    return X_train_internal, y_train.to_numpy().astype(int), feature_names


def load_sampled_trees(dataset_name, param_tag, n_features, max_trees=None, classes=(0, 1)):
    """Load arborenum_sampled_trees.json and rebuild each tree as a TreeClassifier.

    Args:
        n_features (int): Number of columns in the corresponding internal feature matrix
            (i.e. len(feature_names) from load_test_data/load_train_data). Must be passed
            explicitly (rather than inferred from the tree) since a tree may not split on
            the highest-indexed feature.

    Returns:
        dict: tree_index -> TreeClassifier
        dict: metadata (n_total_trees, n_sampled_trees)
    """
    path = RESULTS_DIRS[param_tag] / dataset_name / "arborenum_sampled_trees.json"
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
    """Load arborenum_sampled_trees.json as raw per-leaf literal lists (no TreeClassifier).

    Used for the Hamming/L0 robustness attack, which only needs each leaf's path
    literals (feature, required_bit) and predicted class -- reconstructing a full
    tree object isn't necessary.

    Returns:
        dict: tree_index -> list of {"literals": [(feature, bit), ...], "pred": int}
        dict: metadata (n_total_trees, n_sampled_trees)
    """
    path = RESULTS_DIRS[param_tag] / dataset_name / "arborenum_sampled_trees.json"
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
    """Parse an ArborEnum path string like "[+56, -3]" into [(feature, bit), ...]."""
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
    """Rebuild a sklearn-tree-compatible TreeClassifier from ArborEnum's leaf paths.

    Each entry in `paths_str` is a root-to-leaf conjunction of binary internal-feature
    literals (e.g. "[+56, -3]" means internal feature 56 == 1 AND internal feature 3 == 0
    along this path). Since `paths_str`/`preds` enumerate every leaf of one tree, they
    fully determine a trie, which is rebuilt here into sklearn-style children/feature/
    threshold/value arrays (threshold fixed at 0.5 for every binary split: feature==0 ->
    left child, feature==1 -> right child).
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
