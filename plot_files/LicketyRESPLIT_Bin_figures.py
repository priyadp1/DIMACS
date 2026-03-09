import re
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent
BASEDIR = current

BINAR_DIR = BASEDIR / "LicketyRESPLIT_EXP_ThresholdBinarizer"
PLOTS_DIR = BASEDIR / "plots"
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


df = load_folder(BINAR_DIR)
datasets = sorted(df["dataset"].unique())

METRICS = [
    ("accuracy",          "Test Accuracy"),
    ("ensemble_accuracy", "Ensemble Accuracy"),
    ("n_leaves",          "Tree Leaves (tree 0)"),
    ("n_trees_in_set",    "Rashomon Set Size"),
    ("duration_sec",      "Training Time (s)"),
]

for metric_col, metric_label in METRICS:
    if metric_col not in df.columns:
        continue

    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 5), sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        sub = df[df["dataset"] == dataset].sort_values("param_label")
        x = range(len(sub))
        labels = sub["param_label"].tolist()

        ax.bar(list(x), sub[metric_col], width=0.5, color="#DD8452", alpha=0.85)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(dataset, fontsize=11, fontweight="bold")
        ax.set_ylabel(metric_label if ax == axes[0] else "")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle(f"{metric_label}: LicketyRESPLIT (ThresholdBinarizer)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = PLOTS_DIR / f"bin_{metric_col}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

print(f"\nAll plots saved to: {PLOTS_DIR}")
