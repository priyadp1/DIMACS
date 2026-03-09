"""
Cross-dataset comparison: for each metric and parameter setting,
plot No Binarizer, ThresholdBinarizer, GOSDT, and XGBoost with datasets on the x-axis.
"""
import re
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent
BASEDIR = current

EXP_DIR     = BASEDIR / "LicketyRESPLIT_EXP"
BINAR_DIR   = BASEDIR / "LicketyRESPLIT_EXP_ThresholdBinarizer"
RESULTS_DIR = BASEDIR / "model_results"
PLOTS_DIR   = BASEDIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def parse_results_file(path):
    text = path.read_text()
    acc      = re.search(r"^Accuracy:\s*([\d.]+)", text, re.MULTILINE)
    ens_acc  = re.search(r"Ensemble Accuracy:\s*([\d.]+)", text)
    duration = re.search(r"completed in ([\d.]+) seconds", text)
    return {
        "accuracy":          float(acc.group(1))      if acc      else None,
        "ensemble_accuracy": float(ens_acc.group(1))  if ens_acc  else None,
        "duration_sec":      float(duration.group(1)) if duration else None,
    }


def parse_tree_size_file(path):
    try:
        d = json.loads(path.read_text())
        return {
            "n_leaves":       d.get("n_leaves"),
            "n_nodes":        d.get("n_nodes"),
            "n_trees_in_set": d.get("n_trees_in_set"),
        }
    except Exception:
        return {"n_leaves": None, "n_nodes": None, "n_trees_in_set": None}


def load_folder(folder):
    rows = []
    for param_dir in sorted(folder.iterdir()):
        if not param_dir.is_dir():
            continue
        parts = param_dir.name.split("_")
        if len(parts) < 3:
            continue
        depth, lam, rash = parts[0], parts[1], parts[2]
        param_label = f"d={depth} λ={lam} ε={rash}"

        for res_file in sorted(param_dir.glob("*_results.txt")):
            dataset = res_file.stem.replace("_results", "")
            size_file = param_dir / f"{dataset}_tree_size.json"

            row = {
                "dataset":     dataset,
                "depth":       int(depth),
                "lambda_reg":  float(lam),
                "rashomon":    float(rash),
                "param_label": param_label,
            }
            row.update(parse_results_file(res_file))
            if size_file.exists():
                row.update(parse_tree_size_file(size_file))
            rows.append(row)
    return pd.DataFrame(rows)


def parse_gosdt_results(path):
    text = path.read_text()
    acc      = re.search(r"^Accuracy:\s*([\d.]+)", text, re.MULTILINE)
    duration = re.search(r"completed in ([\d.]+) seconds", text)
    n_leaves = re.search(r"Tree Size:\s*(\d+) leaves", text)
    return {
        "accuracy":     float(acc.group(1))      if acc      else None,
        "duration_sec": float(duration.group(1)) if duration else None,
        "n_leaves":     int(n_leaves.group(1))   if n_leaves else None,
    }


def parse_xgboost_results(path):
    text = path.read_text()
    acc      = re.search(r"^Accuracy:\s*([\d.]+)", text, re.MULTILINE)
    duration = re.search(r"completed in ([\d.]+) seconds", text)
    n_leaves = re.search(r"Tree Size:.*?(\d+) total leaves", text)
    return {
        "accuracy":     float(acc.group(1))      if acc      else None,
        "duration_sec": float(duration.group(1)) if duration else None,
        "n_leaves":     int(n_leaves.group(1))   if n_leaves else None,
    }


def load_baseline_results(results_dir):
    """Load per-dataset GOSDT and XGBoost results from model_results/<dataset>/."""
    gosdt_rows, xgb_rows = {}, {}
    for dataset_dir in sorted(results_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        gosdt_file = dataset_dir / "gosdt_results.txt"
        xgb_file   = dataset_dir / "xgboost_results.txt"
        if gosdt_file.exists():
            gosdt_rows[dataset_dir.name] = parse_gosdt_results(gosdt_file)
        if xgb_file.exists():
            xgb_rows[dataset_dir.name] = parse_xgboost_results(xgb_file)
    return gosdt_rows, xgb_rows


df_exp   = load_folder(EXP_DIR)
df_binar = load_folder(BINAR_DIR)
gosdt_results, xgb_results = load_baseline_results(RESULTS_DIR)

# Only keep (dataset, param_label) pairs present in both LicketyRESPLIT runs
common_keys = set(zip(df_exp["dataset"], df_exp["param_label"])) & \
              set(zip(df_binar["dataset"], df_binar["param_label"]))

df_exp   = df_exp[df_exp.apply(lambda r: (r["dataset"], r["param_label"]) in common_keys, axis=1)].reset_index(drop=True)
df_binar = df_binar[df_binar.apply(lambda r: (r["dataset"], r["param_label"]) in common_keys, axis=1)].reset_index(drop=True)

param_labels = sorted(df_exp["param_label"].unique())

# All datasets: union of LicketyRESPLIT and baseline datasets
lickety_datasets  = set(df_exp["dataset"].unique())
baseline_datasets = set(gosdt_results.keys()) | set(xgb_results.keys())
datasets = sorted(lickety_datasets | baseline_datasets)

METRICS = [
    ("accuracy",          "Test Accuracy"),
    ("ensemble_accuracy", "Ensemble Accuracy"),
    ("n_leaves",          "Tree Leaves (tree 0)"),
    ("n_trees_in_set",    "Rashomon Set Size"),
    ("duration_sec",      "Training Time (s)"),
]

# Metrics supported by GOSDT/XGBoost baselines
BASELINE_METRICS = {"accuracy", "duration_sec", "n_leaves"}

COLORS = {
    "No Binarizer":       "#4C72B0",
    "ThresholdBinarizer": "#DD8452",
    "GOSDT":              "#2ca02c",
    "XGBoost":            "#d62728",
}
BAR_WIDTH = 0.2

for metric_col, metric_label in METRICS:
    if metric_col not in df_exp.columns and metric_col not in BASELINE_METRICS:
        continue

    fig, axes = plt.subplots(1, len(param_labels), figsize=(5 * len(param_labels), 5), sharey=False)
    if len(param_labels) == 1:
        axes = [axes]

    for ax, param_label in zip(axes, param_labels):
        sub_exp   = df_exp[df_exp["param_label"] == param_label].set_index("dataset")
        sub_binar = df_binar[df_binar["param_label"] == param_label].set_index("dataset")

        x = np.arange(len(datasets))

        # LicketyRESPLIT bars (only for metric_col in df columns)
        if metric_col in df_exp.columns:
            nobin_vals = [sub_exp.loc[d, metric_col] if d in sub_exp.index else float("nan") for d in datasets]
            bin_vals   = [sub_binar.loc[d, metric_col] if d in sub_binar.index else float("nan") for d in datasets]
            ax.bar(x - 1.5 * BAR_WIDTH, nobin_vals, width=BAR_WIDTH,
                   label="No Binarizer",       color=COLORS["No Binarizer"],      alpha=0.85)
            ax.bar(x - 0.5 * BAR_WIDTH, bin_vals,   width=BAR_WIDTH,
                   label="ThresholdBinarizer", color=COLORS["ThresholdBinarizer"], alpha=0.85)

        # GOSDT and XGBoost bars (only for supported metrics)
        if metric_col in BASELINE_METRICS:
            gosdt_vals = [gosdt_results.get(d, {}).get(metric_col, float("nan")) for d in datasets]
            xgb_vals   = [xgb_results.get(d, {}).get(metric_col, float("nan"))   for d in datasets]
            ax.bar(x + 0.5 * BAR_WIDTH, gosdt_vals, width=BAR_WIDTH,
                   label="GOSDT",   color=COLORS["GOSDT"],   alpha=0.85)
            ax.bar(x + 1.5 * BAR_WIDTH, xgb_vals,   width=BAR_WIDTH,
                   label="XGBoost", color=COLORS["XGBoost"], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=8)
        ax.set_title(param_label, fontsize=10, fontweight="bold")
        ax.set_ylabel(metric_label if ax == axes[0] else "")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
        ax.legend(fontsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle(f"{metric_label}: Cross-Dataset Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = PLOTS_DIR / f"cross_{metric_col}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

print(f"\nAll plots saved to: {PLOTS_DIR}")
