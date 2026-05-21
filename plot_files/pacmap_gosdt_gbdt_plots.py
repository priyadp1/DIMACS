import pacmap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import os
import pandas as pd
import json
import re

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent
BASEDIR = current
tgb_dir = BASEDIR / "benchmarks_TGB_results"
tgb_vars_dir = BASEDIR / "TGB_Variables_Feature_Importance"
pacmap_output_dir = BASEDIR / "pacmap_embeddings"
os.makedirs(pacmap_output_dir, exist_ok=True)

parameters = ["nest40_depth1", "nest40_depth2", "nest40_depth3",
              "nest100_depth1", "nest100_depth2", "nest100_depth3",
              "nest200_depth1", "nest200_depth2", "nest200_depth3"]
datasets = ["bike", "breast_cancer"]


# ── GOSDT ──────────────────────────────────────────────────────────────────────
# The gosdt_first_tree.txt format is a custom non-Python-literal format:
#   { feature: N, orig feature: N, [ left child: {...}, right child: {...}] }
# ast.literal_eval cannot handle it — we use a custom recursive parser instead.

def _split_top_level(s, sep=','):
    """Split s on sep only at depth 0 (not inside {} or [])."""
    parts, current, depth = [], [], 0
    for ch in s:
        if ch in '{[':
            depth += 1
            current.append(ch)
        elif ch in '}]':
            depth -= 1
            current.append(ch)
        elif ch == sep and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current))
    return parts


def _parse_gosdt_node(s):
    """Recursively parse one GOSDT node from string s."""
    s = s.strip()
    assert s.startswith('{') and s.endswith('}'), f"Expected {{...}}: {s[:60]}"
    content = s[1:-1].strip()
    node = {}
    for part in _split_top_level(content):
        part = part.strip()
        if not part:
            continue
        if part.startswith('['):
            # Children block: [ left child: {...}, right child: {...} ]
            inner = part[1:-1].strip()
            for child_part in _split_top_level(inner):
                child_part = child_part.strip()
                if not child_part:
                    continue
                colon = child_part.index(':')
                key = child_part[:colon].strip()
                node[key] = _parse_gosdt_node(child_part[colon + 1:].strip())
        else:
            colon = part.index(':')
            key = part[:colon].strip()
            val_str = part[colon + 1:].strip()
            try:
                val = int(val_str)
            except ValueError:
                val = float(val_str)
            node[key] = val
    return node


def load_gosdt_tree(filepath):
    with open(filepath) as f:
        text = f.read()
    start = text.find('{')
    end = text.rfind('}') + 1
    return _parse_gosdt_node(text[start:end])


def _collect_internal_nodes(node, node_list):
    if "prediction" in node:
        return
    node["node_id"] = len(node_list)
    node_list.append(node)
    _collect_internal_nodes(node["left child"], node_list)
    _collect_internal_nodes(node["right child"], node_list)


def _build_decision_path(node, x, vec):
    if "prediction" in node:
        return
    vec[node["node_id"]] = 1
    feature_idx = node["feature"]
    if x[feature_idx] == 1:
        _build_decision_path(node["left child"], x, vec)
    else:
        _build_decision_path(node["right child"], x, vec)


def gosdt_fingerprint(tree_dict, X):
    internal_nodes = []
    _collect_internal_nodes(tree_dict, internal_nodes)
    num_nodes = len(internal_nodes)
    paths = []
    for x in X:
        vec = np.zeros(num_nodes)
        _build_decision_path(tree_dict, x, vec)
        paths.append(vec)
    return np.array(paths)


# ── XGBoost ────────────────────────────────────────────────────────────────────

def parse_xgboost_ensemble(filepath):
    with open(filepath) as f:
        content = f.read()
    trees = []
    for block in content.split('\n\n'):
        block = block.strip()
        if not block:
            continue
        tree = {}
        for line in block.split('\n'):
            line = line.strip()
            if not line:
                continue
            colon = line.index(':')
            node_id = int(line[:colon])
            rest = line[colon + 1:]
            if rest.startswith('leaf='):
                tree[node_id] = {'leaf': float(rest[5:])}
            else:
                bracket_end = rest.index(']')
                feature_part = rest[1:bracket_end]
                # "hr lt= 6.5<1" → split on " lt= " → ["hr", "6.5<1"]
                parts = feature_part.split(' lt= ', 1)
                raw_name = parts[0]
                threshold = parts[1].split('<')[0]
                feature_name = f"{raw_name} <= {threshold}"
                after = rest[bracket_end + 2:]
                kv = dict(p.split('=') for p in after.split(','))
                tree[node_id] = {
                    'feature': feature_name,
                    'yes': int(kv['yes']),
                    'no': int(kv['no']),
                }
        if tree:
            trees.append(tree)
    return trees


def xgboost_get_leaf(tree, x, col_index):
    node = 0
    while True:
        n = tree[node]
        if 'leaf' in n:
            return node
        feat_idx = col_index[n['feature']]
        # binary feature == 0 → condition true (< 1) → yes child
        node = n['yes'] if x[feat_idx] == 0 else n['no']


def xgboost_fingerprint(filepath, X, col_names):
    col_index = {name: i for i, name in enumerate(col_names)}
    trees = parse_xgboost_ensemble(filepath)
    result = np.zeros((len(X), len(trees)), dtype=np.int32)
    for t, tree in enumerate(trees):
        for s, x in enumerate(X):
            result[s, t] = xgboost_get_leaf(tree, x, col_index)
    return result


# ── LightGBM ───────────────────────────────────────────────────────────────────

def parse_lightgbm_ensemble(filepath):
    with open(filepath) as f:
        content = f.read()
    # Split on Tree=N markers; first chunk is the header
    import re
    chunks = re.split(r'\nTree=\d+\n', content)
    trees = []
    for chunk in chunks[1:]:
        info = {}
        for line in chunk.split('\n'):
            if '=' in line:
                k, v = line.split('=', 1)
                info[k.strip()] = v.strip()
        if 'split_feature' not in info:
            continue
        trees.append({
            'split_feature': list(map(int, info['split_feature'].split())),
            'left_child':    list(map(int, info['left_child'].split())),
            'right_child':   list(map(int, info['right_child'].split())),
        })
    return trees


def lgbm_get_leaf(tree, x):
    # split_feature indices map directly to CSV column order
    node = 0
    while True:
        feat_idx = tree['split_feature'][node]
        # threshold ≈ 0: feature == 0 → left, feature == 1 → right
        child = tree['left_child'][node] if x[feat_idx] == 0 else tree['right_child'][node]
        if child < 0:
            return -(child + 1)
        node = child


def lgbm_fingerprint(filepath, X):
    trees = parse_lightgbm_ensemble(filepath)
    result = np.zeros((len(X), len(trees)), dtype=np.int32)
    for t, tree in enumerate(trees):
        for s, x in enumerate(X):
            result[s, t] = lgbm_get_leaf(tree, x)
    return result


# ── CatBoost ───────────────────────────────────────────────────────────────────

def catboost_fingerprint(filepath, X):
    with open(filepath) as f:
        d = json.load(f)
    trees = d['oblivious_trees']
    result = np.zeros((len(X), len(trees)), dtype=np.int32)
    for t, tree in enumerate(trees):
        splits = tree['splits']
        for s, x in enumerate(X):
            leaf_idx = 0
            for bit, split in enumerate(splits):
                # float_feature_index maps directly to CSV column index
                if x[split['float_feature_index']] > split['border']:
                    leaf_idx |= (1 << bit)
            result[s, t] = leaf_idx
    return result


# ── Encoding + PaCMAP ──────────────────────────────────────────────────────────

def onehot_leaves(fp):
    """Convert (n_samples, n_trees) leaf indices into one-hot leaf fingerprint."""
    n_samples, n_trees = fp.shape
    max_per_tree = fp.max(axis=0) + 1
    offsets = np.concatenate([[0], np.cumsum(max_per_tree)])
    total_dims = int(offsets[-1])
    result = np.zeros((n_samples, total_dims), dtype=np.float32)
    for t in range(n_trees):
        result[np.arange(n_samples), offsets[t] + fp[:, t]] = 1.0
    return result


def run_pacmap(fp):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(fp.astype(float))
    reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0)
    return reducer.fit_transform(X_scaled)


# ── Main loop ──────────────────────────────────────────────────────────────────

for dataset in datasets:
    for params in parameters:
        bench_dir = tgb_dir / dataset / params
        vars_dir  = tgb_vars_dir / dataset / params

        gosdt_file  = bench_dir / "gosdt_first_tree.txt"
        xgb_file    = bench_dir / "xgboost_ensemble.txt"
        lgbm_file   = bench_dir / "lightgbm_ensemble.txt"
        cb_file     = bench_dir / "catboost_ensemble.json"
        X_train_f   = vars_dir / "X_train_guessed.csv"
        X_test_f    = vars_dir / "X_test_guessed.csv"
        y_train_f   = vars_dir / "y_train.csv"
        y_test_f    = vars_dir / "y_test.csv"

        missing = [p for p in [gosdt_file, xgb_file, lgbm_file, cb_file,
                                X_train_f, X_test_f, y_train_f, y_test_f]
                   if not p.exists()]
        if missing:
            print(f"Skipping {dataset}/{params}: missing {[p.name for p in missing]}")
            continue

        X_train_df = pd.read_csv(X_train_f)
        X_test_df  = pd.read_csv(X_test_f)
        col_names  = list(X_train_df.columns)
        X_all      = np.vstack([X_train_df.values, X_test_df.values])
        y_all      = np.concatenate([
            pd.read_csv(y_train_f).values.ravel(),
            pd.read_csv(y_test_f).values.ravel(),
        ])

        # GOSDT — decision-path binary vector through first tree
        tree_dict = load_gosdt_tree(gosdt_file)
        gosdt_fp = gosdt_fingerprint(tree_dict, X_all)

        # GBDTs — one-hot leaf fingerprint across the full ensemble
        xgb_fp  = onehot_leaves(xgboost_fingerprint(xgb_file, X_all, col_names))
        lgbm_fp = onehot_leaves(lgbm_fingerprint(lgbm_file, X_all))
        cb_fp   = onehot_leaves(catboost_fingerprint(cb_file, X_all))

        models = {
            "GOSDT (first tree, decision path)": gosdt_fp,
            "XGBoost (full ensemble)":           xgb_fp,
            "LightGBM (full ensemble)":          lgbm_fp,
            "CatBoost (full ensemble)":          cb_fp,
        }

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"PaCMAP — Learned Representations: {dataset} / {params}", fontsize=13)

        for ax, (title, fp) in zip(axes.ravel(), models.items()):
            emb = run_pacmap(fp)
            for label, color in [(0, "steelblue"), (1, "tomato")]:
                mask = y_all == label
                ax.scatter(emb[mask, 0], emb[mask, 1],
                           c=color, label=f"class {label}", alpha=0.5, s=8)
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=8)
            ax.axis("off")

        plt.tight_layout()
        out_path = pacmap_output_dir / f"{dataset}_{params}_comparison_pacmap.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")
