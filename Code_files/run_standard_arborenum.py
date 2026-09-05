import json
import time
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from arborenum import ArborEnum
import random

MAX_TREE_SAMPLE = 1000
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

LAMBDA_REG    = 0.01
DEPTH_BUDGET  = 5
RASHOMON_MULT = 0.02
LOOKAHEAD_K   = 1
RID_N_BOOT    = 10
RID_SEED      = 0

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent
BASEDIR = current

DATA_DIR    = BASEDIR / "datasets" / "Mine"
RESULTS_DIR = BASEDIR / "benchmarks_standard_arborenum_raw_results"
DATASETS = {
    # "bike": {
    #     "path": DATA_DIR / "bike.csv",
    #     "target_col": "cnt_binary",
    #     "drop_cols": ["instant", "cnt_binary"],
    #     "label_map": None,
    # },
    # "spambase": {
    #     "path": DATA_DIR / "spambase.csv",
    #     "target_col": "class",
    #     "drop_cols": ["class"],
    #     "label_map": None,
    # },
    "compas": {
        "path": DATA_DIR / "compas.csv",
        "target_col": "two_year_recid",
        "drop_cols": ["two_year_recid"],
        "label_map": None,
    },
    # "breast_cancer": {
    #     "path": DATA_DIR / "breast_cancer_data.csv",
    #     "target_col": "diagnosis",
    #     "drop_cols": ["id", "diagnosis"],
    #     "label_map": {"M": 1, "B": 0},
    # },
    # "heloc": {
    #     "path": DATA_DIR / "heloc_original.csv",
    #     "target_col": "RiskPerformance",
    #     "drop_cols": ["RiskPerformance"],
    #     "label_map": None,
    # },
    # "diabetes_smote": {
    #     "path": DATA_DIR / "diabetes_smote.csv",
    #     "target_col": "readmitted",
    #     "drop_cols": ["readmitted"],
    #     "label_map": None,
    # },
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
        } 
}


def load_dataset(cfg):
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


def run_dataset(dataset_name, cfg):
    out_dir = RESULTS_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    _expected_outputs = [
        "arborenum_results.txt",
        "arborenum_first_tree.txt",
        "arborenum_tree_size.json",
        "arborenum_sampled_trees.json",
        "arborenum_first_tree_paths.txt",
        "arborenum_rid.json",
    ]
    if all((out_dir / f).exists() for f in _expected_outputs):
        print(f"  [ARBORENUM] Skipping {dataset_name} — results already exist.")
        return

    X_train, X_test, y_train, y_test = load_dataset(cfg)

    model = ArborEnum()

    print(f"  [ARBORENUM] Training on {dataset_name}...")
    start = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        proxy_mode = "hybrid",
        lambda_reg = LAMBDA_REG,
        depth_budget = DEPTH_BUDGET,
        rashomon_mult= RASHOMON_MULT,
        lookahead_k = LOOKAHEAD_K,
        max_number_thresholds_per_feature=None,
        key_mode="hash"
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

    with open(out_dir / "arborenum_tree_size.json", "w") as f:
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

    with open(out_dir / "arborenum_sampled_trees.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(out_dir / "arborenum_first_tree.txt", "w", encoding="utf-8") as fh:
        try:
            paths_str, tree_preds = model.get_tree_paths_str(tree_idx)
            fh.write("\n".join(f"{p} -> {pred}" for p, pred in zip(paths_str, tree_preds)))
        except Exception as e:
            fh.write(f"Tree unavailable: {e}\n")

    feature_names = list(X_train.columns)

    print(f"  [ARBORENUM] Computing Rashomon Importance Distribution...")
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
        )
        rid_duration = time.perf_counter() - rid_start

        with open(out_dir / "arborenum_rid.json", "w") as f:
            json.dump(
                {
                    "feature_names": feature_names,
                    "mean_sub_mr": [float(v) for v in rid_out["mean_sub_mr"]],
                },
                f,
            )

        # fig, _ = model.rid_plot_mean(feature_names=feature_names, show=False)
        # fig.savefig(out_dir / "praxis_rid_mean.png", dpi=150, bbox_inches="tight")
        # plt.close(fig)

        # fig, _ = model.rid_plot_violin(feature_names=feature_names, show=False)
        # fig.savefig(out_dir / "praxis_rid_violin.png", dpi=150, bbox_inches="tight")
        # plt.close(fig)

        # fig, _ = model.rid_plot_cdfs(feature_names=feature_names, show=False)
        # fig.savefig(out_dir / "praxis_rid_cdfs.png", dpi=150, bbox_inches="tight")
        # plt.close(fig)

        rid_error = None
    except Exception as e:
        rid_duration = 0.0
        rid_error = str(e)


    print("Get Tree Paths:")
    try:
        paths_str, leaf_preds = model.get_tree_paths_str(tree_idx)

        with open(out_dir / "arborenum_first_tree_paths.txt", "w", encoding="utf-8") as fh:
            for path, pred in zip(paths_str, leaf_preds):
                print(f"Path: {path} -> Prediction: {pred}")
                fh.write(f"{path} -> {pred}\n")
    except Exception as e:
        print(f"Error retrieving tree paths: {e}")

    with open(out_dir / "arborenum_results.txt", "w") as f:
        f.write(f"Accuracy: {acc}")
        f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, test_preds)}")
        f.write(f"\nClassification Report:\n{classification_report(y_test, test_preds)}")
        f.write(f"\nARBORENUM completed in {duration:.2f} seconds")
        f.write(f"\nRashomon set size: {n_trees}")
        if "error" not in tree_size:
            f.write(f"\nTree Size (tree 0): {tree_size['n_leaves']} leaves, {tree_size['n_nodes']} total nodes")
        else:
            f.write(f"\nTree Size: Error - {tree_size['error']}")
        if rid_error is None:
            f.write(f"\nRID computed in {rid_duration:.2f} seconds (see praxis_rid.json)")
        else:
            f.write(f"\nRID: Error - {rid_error}")

    print(f"  [ARBORENUM] Accuracy: {acc:.4f} | Time: {duration:.2f}s | Trees: {n_trees}")


if __name__ == "__main__":
    for dataset_name, cfg in DATASETS.items():
        print(f"\n{'='*60}\nDataset: {dataset_name}\n{'='*60}")
        try:
            run_dataset(dataset_name, cfg)
        except Exception as e:
            print(f"  [ARBORENUM] {dataset_name} failed: {e}")
