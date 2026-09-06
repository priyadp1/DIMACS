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

# unbiased_binning_PRAXIS.py caches exactly one run per dataset (no TGB-style param_tag
# sweep, and no separate "standard"/"anytime" configs the way ArborEnum has) directly
# under RESULTS_DIR/dataset_name/.
RESULTS_DIR = BASEDIR / "benchmarks_unbiased_binning_PRAXIS_raw_results"

# Must match unbiased_fairness_binning.py's RANDOM_SEED (also unbiased_binning_PRAXIS.py's
# train_test_split seed) so the rebuilt split lines up with the one the cached trees were
# fit on. Hardcoded here rather than imported so this module stays a light dependency
# (numpy/pandas/sklearn only) -- unbiased_fairness_binning.py pulls in the actual fairness-
# binning solvers, which this module doesn't need: every column's chosen bin edges are
# already cached in praxis_binning_map.json, so we only ever replay them, never refit them.
RANDOM_SEED = 42

# Fixed seed for the max_trees sub-sample drawn from the cached (already uniformly
# sampled) praxis_sampled_trees.json trees, so results stay reproducible across runs.
SUBSAMPLE_SEED = 0

# module/model/lib/tree_classifier.py only depends on numpy/sklearn/scipy, so we can
# reuse rashomon-framework's sklearn-compatible tree class without needing PRAXIS,
# ArborEnum, or gosdt importable.
sys.path.insert(0, str(BASEDIR / "rashomon-framework"))
from module.model.lib.tree_classifier import TreeClassifier  # noqa: E402


# Mirrors the DATASETS dict in unbiased_binning_PRAXIS.py (kept in sync manually -- update
# here whenever that config changes). group_col/group_map feed the fairness-aware binning's
# sensitive-group labels there, but aren't needed here since the resulting bin edges are
# already cached in praxis_binning_map.json; they're kept in this config anyway so
# load_raw_split matches unbiased_binning_PRAXIS.py's load_raw_split exactly (group_map is
# applied to the raw dataframe before the train/test split).
DATASET_CONFIGS = {
    "compas": {
        "path": DATA_DIR / "compas.csv",
        "target_col": "two_year_recid",
        "drop_cols": ["two_year_recid"],
        "label_map": None,
        "na_strategy": "drop_rows",
        "cols_to_drop_for_na": None,
        "group_col": "sex=female",
        "group_map": None,
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
    "diabetes_smote": {
        "path": DATA_DIR / "diabetes_smote.csv",
        "target_col": "readmitted",
        "drop_cols": ["readmitted"],
        "label_map": None,
        "na_strategy": "drop_rows",
        "cols_to_drop_for_na": None,
        "group_col": "race",
        "group_map": None,
    },
}


def discover_all_datasets():
    """List every dataset with a cached unbiased-binning PRAXIS Rashomon set."""
    if not RESULTS_DIR.exists():
        return []
    return sorted(p.name for p in RESULTS_DIR.iterdir() if p.is_dir() and discover_param_tags(p.name))


# This backend has no param_tag sweep -- one run per dataset -- but discover_param_tags/
# load_test_data etc. still take a param_tag argument for call-signature parity with
# praxis_rashomon_common and arborenum_rashomon_common, so the same calculate_*_rashomon.py
# loop shape (`for param_tag in discover_param_tags(dataset_name)`) works unmodified
# across all three backends.
PARAM_TAG = "unbiased_binning"


def discover_param_tags(dataset_name):
    """[PARAM_TAG] if this dataset has a cached praxis_sampled_trees.json, else []."""
    return [PARAM_TAG] if (RESULTS_DIR / dataset_name / "praxis_sampled_trees.json").exists() else []


@functools.lru_cache(maxsize=None)
def _load_dataset_split(dataset_name):
    """Rebuild X_train_raw/X_test_raw/y_train/y_test exactly as unbiased_binning_PRAXIS.py's
    load_raw_split() did: same CSV, same NA handling, same label_map/group_map, same
    train_test_split(test_size=0.2, random_state=RANDOM_SEED, stratify=y). Unlike
    binarize_features (the fairness-binning step), this part has no cached state to
    replay -- it's a deterministic function of the raw CSV and RANDOM_SEED alone.
    """
    cfg = DATASET_CONFIGS[dataset_name]
    df = pd.read_csv(cfg["path"])
    if cfg["na_strategy"] == "drop_columns":
        df = df.drop(columns=cfg["cols_to_drop_for_na"])
    else:  # "drop_rows"
        df = df.dropna(axis=1, how="all")
        df = df.dropna()

    if cfg["label_map"]:
        df[cfg["target_col"]] = df[cfg["target_col"]].map(cfg["label_map"])
    if cfg["group_map"]:
        df[cfg["group_col"]] = df[cfg["group_col"]].map(cfg["group_map"])

    X = df.drop(columns=cfg["drop_cols"])
    y = df[cfg["target_col"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    return (
        X_train.reset_index(drop=True), X_test.reset_index(drop=True),
        y_train.reset_index(drop=True), y_test.reset_index(drop=True),
    )


def _load_per_feature_info(dataset_name):
    path = RESULTS_DIR / dataset_name / "praxis_binning_map.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["per_feature_info"]


def _reconstruct_binarized_features(X_train_raw, X_eval_raw, per_feature_info):
    """Replay unbiased_binning_PRAXIS.py's binarize_features() using its own cached
    decisions (praxis_binning_map.json's per_feature_info) instead of refitting them.

    Each raw column becomes zero or more binary output columns, in the exact same
    order/shape binarize_features produced originally (per_feature_info preserves
    insertion order = original raw-column order, and json.load preserves key order):
      - "constant"/"degenerate": dropped, no output columns.
      - "categorical": one-hot via pd.get_dummies(col.astype(str), prefix=col) --
        deterministic from the raw values alone, so no cached state needed beyond
        knowing this column was categorical.
      - "binary_passthrough": the column already had exactly 2 unique training values;
        map them to 0/1 by sorted order (deterministic from X_train_raw alone).
      - "fairness_binned" (covers every actual fairness-binning method AND the
        quantile fallback -- both are recorded under this one type): apply the
        cached `cut_values` thresholds directly. This is the one case that
        genuinely can't be re-derived without the cache, since it's the output of
        a fairness-constrained optimization (or DP-noised) binning algorithm.
    Column-name collisions (e.g. duplicate cut_values producing the same
    "col <= edge" name) silently overwrite, matching binarize_features' own
    dict-based accumulation -- so the reconstructed n_features always matches
    whatever the cached trees were actually built against.
    """
    train_cols, eval_cols = {}, {}

    for col, info in per_feature_info.items():
        kind = info.get("type")

        if kind in ("constant", "degenerate"):
            continue

        if kind == "categorical":
            dummies_train = pd.get_dummies(X_train_raw[col].astype(str), prefix=col)
            dummies_eval = pd.get_dummies(X_eval_raw[col].astype(str), prefix=col)
            dummies_eval = dummies_eval.reindex(columns=dummies_train.columns, fill_value=0)
            for dcol in dummies_train.columns:
                train_cols[dcol] = dummies_train[dcol].astype(int).to_numpy()
                eval_cols[dcol] = dummies_eval[dcol].astype(int).to_numpy()
            continue

        if kind == "binary_passthrough":
            uniq_sorted = sorted(X_train_raw[col].unique())
            mapping = {uniq_sorted[0]: 0, uniq_sorted[1]: 1}
            train_cols[col] = X_train_raw[col].map(mapping).astype(int).to_numpy()
            eval_cols[col] = X_eval_raw[col].map(mapping).fillna(0).astype(int).to_numpy()
            continue

        if kind == "fairness_binned":
            for edge in info["cut_values"]:
                cname = f"{col} <= {edge}"
                train_cols[cname] = (X_train_raw[col] <= edge).astype(int).to_numpy()
                eval_cols[cname] = (X_eval_raw[col] <= edge).astype(int).to_numpy()
            continue

        raise ValueError(f"Unknown per_feature_info type {kind!r} for column {col!r}")

    feature_names = list(train_cols.keys())
    n_train, n_eval = len(X_train_raw), len(X_eval_raw)
    if not feature_names:
        return (
            np.empty((n_train, 0), dtype=np.uint8),
            np.empty((n_eval, 0), dtype=np.uint8),
            feature_names,
        )

    X_train_bin = np.column_stack([train_cols[c] for c in feature_names]).astype(np.uint8)
    X_eval_bin = np.column_stack([eval_cols[c] for c in feature_names]).astype(np.uint8)
    return X_train_bin, X_eval_bin, feature_names


@functools.lru_cache(maxsize=None)
def _load_binarized_split(dataset_name):
    X_train_raw, X_test_raw, y_train, y_test = _load_dataset_split(dataset_name)
    per_feature_info = _load_per_feature_info(dataset_name)
    X_train_bin, X_test_bin, feature_names = _reconstruct_binarized_features(
        X_train_raw, X_test_raw, per_feature_info
    )
    return X_train_bin, X_test_bin, y_train.to_numpy().astype(int), y_test.to_numpy().astype(int), feature_names


def load_test_data(dataset_name, param_tag=None):
    """`param_tag` is accepted for call-signature parity with the other two backends but
    ignored -- this dataset has exactly one cached run, so there's nothing to select.

    Returns:
        X_test (np.ndarray[uint8]), y_test (np.ndarray[int]), feature_names (list[str])
    """
    _, X_test_bin, _, y_test, feature_names = _load_binarized_split(dataset_name)
    return X_test_bin, y_test, feature_names


def load_train_data(dataset_name, param_tag=None):
    """See load_test_data for why `param_tag` is accepted but ignored.

    Returns:
        X_train (np.ndarray[uint8]), y_train (np.ndarray[int]), feature_names (list[str])
    """
    X_train_bin, _, y_train, _, feature_names = _load_binarized_split(dataset_name)
    return X_train_bin, y_train, feature_names


def load_sampled_trees(dataset_name, param_tag, n_features, max_trees=None, classes=(0, 1)):
    """Load praxis_sampled_trees.json and rebuild each tree as a TreeClassifier.

    Args:
        n_features (int): Number of columns in the corresponding binarized feature matrix
            (i.e. len(feature_names) from load_test_data/load_train_data). Must be passed
            explicitly (rather than inferred from the tree) since a tree may not split on
            the highest-indexed feature.

    Returns:
        dict: tree_index -> TreeClassifier
        dict: metadata (n_total_trees, n_sampled_trees)
    """
    path = RESULTS_DIR / dataset_name / "praxis_sampled_trees.json"
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
    path = RESULTS_DIR / dataset_name / "praxis_sampled_trees.json"
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
