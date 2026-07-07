import pacmap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import os
import pandas as pd
import re

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent
BASEDIR = current
tgb_dir = BASEDIR / "benchmarks_TGB_results_all"
no_tgb_dir = BASEDIR / "benchmarks_no_TGB_results_all"
no_tgb_splits_dir = BASEDIR / "tmp_splits_no_tgb"
tgb_vars_dir = BASEDIR / "TGB_Variables_Feature_Importance"
pacmap_output_dir = BASEDIR / "pacmap_plots"
os.makedirs(pacmap_output_dir, exist_ok=True)
parameters = ["nest5_depth1", "nest5_depth2", "nest5_depth3", "nest10_depth1", "nest10_depth2", "nest10_depth3", "nest15_depth1", "nest15_depth2", "nest15_depth3", "nest20_depth1", "nest20_depth2", "nest20_depth3" , "nest25_depth1", "nest25_depth2", "nest25_depth3" , "nest30_depth1", "nest30_depth2", "nest30_depth3", "nest35_depth1", "nest35_depth2", "nest35_depth3", "nest40_depth1", "nest40_depth2", "nest40_depth3", "nest100_depth1", "nest100_depth2", "nest100_depth3", "nest200_depth1", "nest200_depth2", "nest200_depth3"]
datasets = ["bike" , "breast_cancer" , "creditcard_fraud" , "diabetes" , "heloc_original" ] 
no_tgb_datasets = ["bike" , "breast_cancer" , "creditcard_fraud" , "diabetes" , "heloc_original" ]


# ── GOSDT ──────────────────────────────────────────────────────────────────────

def _split_top_level(s, sep=','):
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
    s = s.strip()
    assert s.startswith('{') and s.endswith('}')
    content = s[1:-1].strip()
    node = {}
    for part in _split_top_level(content):
        part = part.strip()
        if not part:
            continue
        if part.startswith('['):
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


def gosdt_predict(node, x):
    if "prediction" in node:
        return node["prediction"]
    feature_idx = node["feature"]
    if x[feature_idx] == 1:
        return gosdt_predict(node["left child"], x)
    else:
        return gosdt_predict(node["right child"], x)


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
                raw_name, threshold_part = feature_part.split(' lt= ', 1)
                if threshold_part.endswith('<1'):
                    threshold_part = threshold_part[:-2]
                feature_name = f"{raw_name} <= {threshold_part}"
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


def xgboost_predict(filepath, X, col_names):
    trees = parse_xgboost_ensemble(filepath)
    col_index = {name: i for i, name in enumerate(col_names)}
    preds = []
    for x in X:
        score = 0.0
        for tree in trees:
            node = 0
            while True:
                n = tree[node]
                if "leaf" in n:
                    score += n["leaf"]
                    break
                feat_idx = col_index[n["feature"]]
                node = n["yes"] if x[feat_idx] == 0 else n["no"]
        preds.append(1 if score > 0 else 0)
    return np.array(preds)


# ── XGBoost (raw / no-TGB format: [feature<threshold]) ────────────────────────

def parse_xgboost_ensemble_raw(filepath):
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
                lt_pos = feature_part.index('<')
                feature_name = feature_part[:lt_pos]
                threshold = float(feature_part[lt_pos + 1:])
                after = rest[bracket_end + 2:]
                kv = dict(p.split('=') for p in after.split(','))
                tree[node_id] = {
                    'feature': feature_name,
                    'threshold': threshold,
                    'yes': int(kv['yes']),
                    'no': int(kv['no']),
                }
        if tree:
            trees.append(tree)
    return trees


def xgboost_predict_raw(filepath, X, col_names):
    trees = parse_xgboost_ensemble_raw(filepath)
    col_index = {name: i for i, name in enumerate(col_names)}
    preds = []
    for x in X:
        score = 0.0
        for tree in trees:
            node = 0
            while True:
                n = tree[node]
                if 'leaf' in n:
                    score += n['leaf']
                    break
                feat_idx = col_index[n['feature']]
                node = n['yes'] if x[feat_idx] < n['threshold'] else n['no']
        preds.append(1 if score > 0 else 0)
    return np.array(preds)


# ── PaCMAP ─────────────────────────────────────────────────────────────────────

def run_pacmap(X, distance="euclidean"):
    if distance == "hamming":
        reducer = pacmap.PaCMAP(
            n_components=2,
            n_neighbors=6,
            MN_ratio=0.5,
            FP_ratio=2.0,
            distance="hamming",
            knn_backend="annoy",
        )
        return reducer.fit_transform(X.astype(float))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.astype(float))
    reducer = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=10,
        MN_ratio=0.5,
        FP_ratio=2.0,
    )
    return reducer.fit_transform(X_scaled)


def save_pacmap_plot(emb, preds, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(title, fontsize=11)
    for label, color in [(0, "steelblue"), (1, "tomato")]:
        mask = preds == label
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            c=color, label=f"class {label}",
            alpha=0.5, s=8,
        )
    ax.legend(fontsize=8)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


# ── Main loop ──────────────────────────────────────────────────────────────────

for dataset in datasets:
    for params in parameters:
        bench_dir = tgb_dir / dataset / params
        vars_dir = tgb_vars_dir / dataset / params
        gosdt_file = bench_dir / "gosdt_first_tree.txt"
        xgb_file = bench_dir / "xgboost_ensemble.txt"
        X_train_f = vars_dir / "X_train_guessed.csv"
        X_test_f = vars_dir / "X_test_guessed.csv"

        missing = [
            p for p in [gosdt_file, xgb_file, X_train_f, X_test_f]
            if not p.exists()
        ]
        if missing:
            print(f"Skipping {dataset}/{params}: missing {[p.name for p in missing]}")
            continue

        X_train_df = pd.read_csv(X_train_f)
        X_test_df = pd.read_csv(X_test_f)
        col_names = list(X_train_df.columns)
        X_all = np.vstack([X_train_df.values, X_test_df.values])

        if X_all.shape[1] < 2:
            print(f"Skipping {dataset}/{params}: only {X_all.shape[1]} feature(s), need ≥2 for PaCMAP")
            continue

        # PaCMAP on binarized features only — no labels passed in
        emb = run_pacmap(X_all, distance="hamming")

        # Predictions used only for coloring
        tree_dict = load_gosdt_tree(gosdt_file)
        gosdt_preds = np.array([gosdt_predict(tree_dict, x) for x in X_all])
        xgb_preds = xgboost_predict(xgb_file, X_all, col_names)

        plot_dir = pacmap_output_dir / dataset / params
        os.makedirs(plot_dir, exist_ok=True)

        save_pacmap_plot(
            emb, gosdt_preds,
            f"PaCMAP — GOSDT predictions | {dataset} / {params}",
            plot_dir / "gosdt_pacmap.png",
        )
        save_pacmap_plot(
            emb, xgb_preds,
            f"PaCMAP — XGBoost predictions | {dataset} / {params}",
            plot_dir / "xgb_pacmap.png",
        )

# ── No-TGB loop ────────────────────────────────────────────────────────────────

for dataset in no_tgb_datasets:
    bench_dir = no_tgb_dir / dataset
    splits_dir = no_tgb_splits_dir / dataset
    gosdt_file = bench_dir / "gosdt_first_tree.txt"
    xgb_file = bench_dir / "xgboost_ensemble.txt"
    X_train_f = splits_dir / "X_train.csv"
    X_test_f = splits_dir / "X_test.csv"

    missing = [p for p in [gosdt_file, xgb_file, X_train_f, X_test_f] if not p.exists()]
    if missing:
        print(f"Skipping no_tgb/{dataset}: missing {[p.name for p in missing]}")
        continue

    X_train_df = pd.read_csv(X_train_f)
    X_test_df = pd.read_csv(X_test_f)
    col_names = list(X_train_df.columns)
    X_all = np.vstack([X_train_df.values, X_test_df.values])

    if X_all.shape[1] < 2:
        print(f"Skipping no_tgb/{dataset}: only {X_all.shape[1]} feature(s), need ≥2 for PaCMAP")
        continue

    emb = run_pacmap(X_all)

    tree_dict = load_gosdt_tree(gosdt_file)
    gosdt_preds = np.array([gosdt_predict(tree_dict, x) for x in X_all])
    xgb_preds = xgboost_predict_raw(xgb_file, X_all, col_names)

    plot_dir = pacmap_output_dir / "no_tgb" / dataset
    os.makedirs(plot_dir, exist_ok=True)

    save_pacmap_plot(
        emb, gosdt_preds,
        f"PaCMAP — GOSDT predictions | no_tgb / {dataset}",
        plot_dir / "gosdt_pacmap.png",
    )
    save_pacmap_plot(
        emb, xgb_preds,
        f"PaCMAP — XGBoost predictions | no_tgb / {dataset}",
        plot_dir / "xgb_pacmap.png",
    )
