import json
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from praxis import PRAXIS

MAX_TREE_SAMPLE = 1000

LAMBDA_REG    = 0.01
DEPTH_BUDGET  = 5
RASHOMON_MULT = 0.03
LOOKAHEAD_K   = 1
RID_N_BOOT    = 10
RID_SEED      = 0

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent
BASEDIR = current

from unbiased_fairness_binning import (
    run_unbiased_binning,
    run_ebias_binning,
    run_ebias_dnc,
    N_BINS,
    EPSILON,
    RANDOM_SEED,
    MAX_QUADRATIC_N,
)
random.seed(RANDOM_SEED)

DATA_DIR    = BASEDIR / "datasets" / "Mine"
RESULTS_DIR = BASEDIR / "benchmarks_unbiased_binning_PRAXIS_raw_results"

DATASETS = {
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


def load_raw_split(cfg):
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


def fit_fairness_bins(x_train_int, group_train, k):
    """Fits fairness-aware cut values for one integer-valued column against
    group_train, cascading unbiased_binning -> ebias_binning -> ebias_dnc
    (same preference order used to build the headline results in
    benchmarks_unbiased_fairness_binning_raw_results). Returns
    (cut_values, method_name) or (None, None) if every algorithm is
    infeasible for this column."""
    D = np.column_stack((x_train_int, group_train))
    n = len(D)

    res = run_unbiased_binning(D, k=k)
    if res["feasible"]:
        return res["boundary_values"], "unbiased_binning"

    D_dp = D
    if n > MAX_QUADRATIC_N:
        rng = np.random.default_rng(RANDOM_SEED)
        sample_idx = rng.choice(n, size=MAX_QUADRATIC_N, replace=False)
        D_dp = D[sample_idx]
    res = run_ebias_binning(D_dp, k=k, eps=EPSILON)
    if res["feasible"]:
        return res["boundary_values"], "ebias_binning"

    res = run_ebias_dnc(D, k=k, eps=EPSILON)
    if res["feasible"]:
        sorted_x = np.sort(D[:, 0])
        edges = []
        for cut in res["boundaries"][1:-1]:
            lo, hi = sorted_x[cut - 1], sorted_x[cut]
            edges.append(float(lo) if lo == hi else (float(lo) + float(hi)) / 2.0)
        return edges, "ebias_dnc"

    return None, None


def binarize_features(X_train, X_test, group_train, k=N_BINS):
    train_cols = {}
    test_cols = {}
    col_to_origin = {}
    per_feature_info = {}

    for col in X_train.columns:
        s_train = X_train[col]
        s_test = X_test[col]

        if not pd.api.types.is_numeric_dtype(s_train):
            dummies_train = pd.get_dummies(s_train.astype(str), prefix=col)
            dummies_test = pd.get_dummies(s_test.astype(str), prefix=col)
            dummies_test = dummies_test.reindex(columns=dummies_train.columns, fill_value=0)
            for dcol in dummies_train.columns:
                train_cols[dcol] = dummies_train[dcol].astype(int).to_numpy()
                test_cols[dcol] = dummies_test[dcol].astype(int).to_numpy()
                col_to_origin[dcol] = col
            per_feature_info[col] = {"type": "categorical", "n_columns": int(len(dummies_train.columns))}
            continue

        n_unique = int(s_train.nunique(dropna=True))
        if n_unique <= 1:
            per_feature_info[col] = {"type": "constant", "dropped": True}
            continue

        if n_unique == 2:
            uniq_sorted = sorted(s_train.unique())
            mapping = {uniq_sorted[0]: 0, uniq_sorted[1]: 1}
            train_cols[col] = s_train.map(mapping).astype(int).to_numpy()
            test_cols[col] = s_test.map(mapping).fillna(0).astype(int).to_numpy()
            col_to_origin[col] = col
            per_feature_info[col] = {"type": "binary_passthrough"}
            continue

        x_train_vals = s_train.to_numpy()
        scale = 1
        if not np.issubdtype(x_train_vals.dtype, np.integer):
            x_train_vals = (x_train_vals * 100).astype(int)
            scale = 100

        cut_values, method = fit_fairness_bins(x_train_vals, group_train, k=min(k, n_unique))
        if cut_values is None:
            edges = pd.qcut(s_train, q=min(k, n_unique), retbins=True, duplicates="drop")[1]
            cut_values = [float(e) for e in edges[1:-1]]
            method = "quantile_fallback"
        elif scale != 1:
            cut_values = [c / scale for c in cut_values]

        if not cut_values:
            per_feature_info[col] = {"type": "degenerate", "dropped": True}
            continue

        for edge in cut_values:
            cname = f"{col} <= {edge}"
            train_cols[cname] = (s_train <= edge).astype(int).to_numpy()
            test_cols[cname] = (s_test <= edge).astype(int).to_numpy()
            col_to_origin[cname] = col
        per_feature_info[col] = {
            "type": "fairness_binned",
            "method": method,
            "n_bins": len(cut_values) + 1,
            "cut_values": cut_values,
        }

    X_train_bin = pd.DataFrame(train_cols)
    X_test_bin = pd.DataFrame(test_cols).reindex(columns=X_train_bin.columns, fill_value=0)

    origin_to_indices = {}
    for idx, cname in enumerate(X_train_bin.columns):
        origin_to_indices.setdefault(col_to_origin[cname], []).append(idx)

    binning_map = {}
    group_names = []
    for key, (origin, idxs) in enumerate(origin_to_indices.items()):
        binning_map[key] = idxs
        group_names.append(origin)

    return X_train_bin, X_test_bin, binning_map, group_names, per_feature_info


def run_dataset(dataset_name, cfg):
    out_dir = RESULTS_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    _expected_outputs = [
        "praxis_results.txt",
        "praxis_first_tree.txt",
        "praxis_tree_size.json",
        "praxis_sampled_trees.json",
        "praxis_first_tree_paths.txt",
        "praxis_rid.json",
        "praxis_binning_map.json",
    ]
    if all((out_dir / f).exists() for f in _expected_outputs):
        print(f"  [PRAXIS] Skipping {dataset_name} — results already exist.")
        return

    X_train_raw, X_test_raw, y_train, y_test = load_raw_split(cfg)
    group_train = X_train_raw[cfg["group_col"]].to_numpy().astype(int)

    print(f"  [PRAXIS] Fairness-binarizing {dataset_name} "
          f"({X_train_raw.shape[1]} raw features, group='{cfg['group_col']}')...")
    X_train, X_test, binning_map, group_names, per_feature_info = binarize_features(
        X_train_raw, X_test_raw, group_train, k=N_BINS,
    )

    with open(out_dir / "praxis_binning_map.json", "w") as f:
        json.dump({
            "group_names": group_names,
            "binning_map": {str(k): v for k, v in binning_map.items()},
            "per_feature_info": per_feature_info,
        }, f, indent=2)

    model = PRAXIS()

    print(f"  [PRAXIS] Training on {dataset_name}...")
    start = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        lambda_reg=LAMBDA_REG,
        depth_budget=DEPTH_BUDGET,
        rashomon_mult=RASHOMON_MULT,
        lookahead_k=LOOKAHEAD_K,
    )
    duration = time.perf_counter() - start

    n_trees = model.count_trees()
    if n_trees <= MAX_TREE_SAMPLE:
        sampled_tree_indices = list(range(n_trees))
    else:
        sampled_tree_indices = sorted(
            random.sample(range(n_trees), MAX_TREE_SAMPLE)
        )

    tree_idx = 0   # Keep for evaluation and visualization
    test_preds = model.get_predictions(tree_idx, X_test)
    acc = accuracy_score(y_test, test_preds)

    try:
        paths0, _ = model.get_tree_paths(tree_idx)
        n_leaves = len(paths0)
        n_nodes = 2 * n_leaves - 1
        tree_size = {
            "n_leaves": n_leaves,
            "n_nodes": n_nodes,
            "n_trees_in_set": n_trees,
            "n_sampled_trees": len(sampled_tree_indices),
        }
    except Exception as e:
        tree_size = {"error": str(e)}

    with open(out_dir / "praxis_tree_size.json", "w") as f:
        json.dump(tree_size, f, indent=2)

    all_tree_paths = []

    for idx in sampled_tree_indices:
        try:
            paths_str, leaf_preds = model.get_tree_paths_str(idx)

            all_tree_paths.append({
                "tree_index": idx,
                "paths": paths_str,
                "predictions": [int(p) for p in leaf_preds],
            })

        except Exception as e:
            print(f"Failed on tree {idx}: {e}")

    metadata = {
        "n_total_trees": n_trees,
        "n_sampled_trees": len(sampled_tree_indices),
        "trees": all_tree_paths,
    }

    with open(out_dir / "praxis_sampled_trees.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(out_dir / "praxis_first_tree.txt", "w", encoding="utf-8") as fh:
        try:
            paths_str, tree_preds = model.get_tree_paths_str(tree_idx)
            fh.write("\n".join(f"{p} -> {pred}" for p, pred in zip(paths_str, tree_preds)))
        except Exception as e:
            fh.write(f"Tree unavailable: {e}\n")

    print(f"  [PRAXIS] Computing Rashomon Importance Distribution...")
    try:
        rid_start = time.perf_counter()
        rid_out = model.compute_rid(
            X_train,
            y_train,
            n_boot=RID_N_BOOT,
            lambda_reg=LAMBDA_REG,
            depth_budget=DEPTH_BUDGET,
            rashomon_mult=RASHOMON_MULT,
            lookahead_k=LOOKAHEAD_K,
            seed=RID_SEED,
            memory_efficient=False,
            binning_map=binning_map,
        )
        rid_duration = time.perf_counter() - rid_start

        with open(out_dir / "praxis_rid.json", "w") as f:
            json.dump(
                {
                    "feature_names": group_names,
                    "mean_sub_mr": [float(v) for v in rid_out["mean_sub_mr"]],
                },
                f,
            )

        # fig, _ = model.rid_plot_mean(feature_names=group_names, show=False)
        # fig.savefig(out_dir / "praxis_rid_mean.png", dpi=150, bbox_inches="tight")
        # plt.close(fig)

        # fig, _ = model.rid_plot_violin(feature_names=group_names, show=False)
        # fig.savefig(out_dir / "praxis_rid_violin.png", dpi=150, bbox_inches="tight")
        # plt.close(fig)

        # fig, _ = model.rid_plot_cdfs(feature_names=group_names, show=False)
        # fig.savefig(out_dir / "praxis_rid_cdfs.png", dpi=150, bbox_inches="tight")
        # plt.close(fig)

        rid_error = None
    except Exception as e:
        rid_duration = 0.0
        rid_error = str(e)


    print("Get Tree Paths:")
    try:
        paths_str, leaf_preds = model.get_tree_paths_str(tree_idx)

        with open(out_dir / "praxis_first_tree_paths.txt", "w", encoding="utf-8") as fh:
            for path, pred in zip(paths_str, leaf_preds):
                print(f"Path: {path} -> Prediction: {pred}")
                fh.write(f"{path} -> {pred}\n")
    except Exception as e:
        print(f"Error retrieving tree paths: {e}")

    with open(out_dir / "praxis_results.txt", "w") as f:
        f.write(f"Accuracy: {acc}")
        f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, test_preds)}")
        f.write(f"\nClassification Report:\n{classification_report(y_test, test_preds)}")
        f.write(f"\nPRAXIS completed in {duration:.2f} seconds")
        f.write(f"\nRashomon set size: {n_trees}")
        if "error" not in tree_size:
            f.write(f"\nTree Size (tree 0): {tree_size['n_leaves']} leaves, {tree_size['n_nodes']} total nodes")
        else:
            f.write(f"\nTree Size: Error - {tree_size['error']}")
        if rid_error is None:
            f.write(f"\nRID computed in {rid_duration:.2f} seconds (see praxis_rid.json)")
        else:
            f.write(f"\nRID: Error - {rid_error}")

    print(f"  [PRAXIS] {dataset_name} — Accuracy: {acc:.4f} | Time: {duration:.2f}s | Trees: {n_trees}")


if __name__ == "__main__":
    for dataset_name, cfg in DATASETS.items():
        print(f"\n{'='*60}\nDataset: {dataset_name}\n{'='*60}")
        try:
            run_dataset(dataset_name, cfg)
        except Exception as e:
            print(f"  [PRAXIS] {dataset_name} failed: {e}")
