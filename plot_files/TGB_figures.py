"""
Plots results from run_from_TGB.py experiments.
Compares GOSDT, LicketyRESPLIT+TGB (binarized), and XGBoost
across bike, breast_cancer, compas, and spambase datasets.
Saves figures to analysis_figures/.
"""
import re
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent
BASEDIR = current

RESULTS_DIR = BASEDIR / "model_results"
PLOTS_DIR   = BASEDIR / "analysis_figures"
PLOTS_DIR.mkdir(exist_ok=True)

DATASETS = ["bike", "breast_cancer", "compas", "spambase"]

MODELS = ["GOSDT", "LicketyRESPLIT+TGB", "XGBoost"]

COLORS = {
    "GOSDT":              "#2ca02c",
    "LicketyRESPLIT+TGB": "#4C72B0",
    "XGBoost":            "#d62728",
}


# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_licketyresplit_bin(path):
    text = path.read_text()
    acc      = re.search(r"^Accuracy:\s*([\d.]+)", text, re.MULTILINE)
    ens_acc  = re.search(r"Ensemble Accuracy:\s*([\d.]+)", text)
    duration = re.search(r"completed in ([\d.]+) seconds", text)
    return {
        "accuracy":          float(acc.group(1))      if acc      else None,
        "ensemble_accuracy": float(ens_acc.group(1))  if ens_acc  else None,
        "duration_sec":      float(duration.group(1)) if duration else None,
    }


def parse_gosdt(path):
    text = path.read_text()
    acc       = re.search(r"^Accuracy:\s*([\d.]+)", text, re.MULTILINE)
    train_acc = re.search(r"Training Accuracy:\s*([\d.]+)", text)
    duration  = re.search(r"completed in ([\d.]+) seconds", text)
    return {
        "accuracy":       float(acc.group(1))       if acc       else None,
        "train_accuracy": float(train_acc.group(1)) if train_acc else None,
        "duration_sec":   float(duration.group(1))  if duration  else None,
    }


def parse_xgboost(path):
    text = path.read_text()
    acc      = re.search(r"^Accuracy:\s*([\d.]+)", text, re.MULTILINE)
    duration = re.search(r"completed in ([\d.]+) seconds", text)
    return {
        "accuracy":     float(acc.group(1))      if acc      else None,
        "duration_sec": float(duration.group(1)) if duration else None,
    }


def parse_tree_size(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


# ── Load all results ──────────────────────────────────────────────────────────

data = {}   # data[dataset][model] = dict of metrics

for dataset in DATASETS:
    d = {}
    ds_dir = RESULTS_DIR / dataset

    files = {
        "GOSDT":              ("gosdt_results.txt",                    "gosdt_tree_size.json",                    parse_gosdt),
        "LicketyRESPLIT+TGB": ("licketyresplit_binarized_results.txt", "licketyresplit_binarized_tree_size.json", parse_licketyresplit_bin),
        "XGBoost":            ("xgboost_results.txt",                  "xgboost_tree_size.json",                  parse_xgboost),
    }

    for model, (res_f, sz_f, parser) in files.items():
        res_path = ds_dir / res_f
        sz_path  = ds_dir / sz_f
        if res_path.exists():
            d[model] = {**parser(res_path), **(parse_tree_size(sz_path) if sz_path.exists() else {})}
        else:
            d[model] = {}

    data[dataset] = d


# ── Helpers ───────────────────────────────────────────────────────────────────

x     = np.arange(len(DATASETS))
bar_w = 0.25
offsets = np.array([i * bar_w for i in range(len(MODELS))]) - bar_w * (len(MODELS) - 1) / 2


def get_vals(metric):
    return {m: [data[ds].get(m, {}).get(metric, float("nan")) for ds in DATASETS]
            for m in MODELS}


def bar_chart(ax, vals_by_model, ylabel):
    for offset, (model, vals) in zip(offsets, vals_by_model.items()):
        ax.bar(x + offset, vals, width=bar_w, label=model,
               color=COLORS[model], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)


# ── Figure 1: Test Accuracy ───────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 4))
bar_chart(ax, get_vals("accuracy"), "Test Accuracy")
ax.set_title("Test Accuracy: GOSDT vs LicketyRESPLIT+TGB vs XGBoost",
             fontweight="bold")
plt.tight_layout()
out = PLOTS_DIR / "tgb_accuracy.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# ── Figure 2: Ensemble Accuracy (LicketyRESPLIT) vs Test Accuracy (others) ───

fig, ax = plt.subplots(figsize=(9, 4))
ens_vals = {}
for model in MODELS:
    metric = "ensemble_accuracy" if model == "LicketyRESPLIT+TGB" else "accuracy"
    ens_vals[model] = [data[ds].get(model, {}).get(metric, float("nan")) for ds in DATASETS]
bar_chart(ax, ens_vals, "Accuracy")
ax.set_title("Ensemble Accuracy (LicketyRESPLIT+TGB) vs Test Accuracy (GOSDT, XGBoost)",
             fontweight="bold")
plt.tight_layout()
out = PLOTS_DIR / "tgb_ensemble_accuracy.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# ── Figure 3: Training Time ───────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 4))
bar_chart(ax, get_vals("duration_sec"), "Training Time (s)")
ax.set_title("Training Time: GOSDT vs LicketyRESPLIT+TGB vs XGBoost",
             fontweight="bold")
plt.tight_layout()
out = PLOTS_DIR / "tgb_duration.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# ── Figure 4: Tree Size (leaves) ──────────────────────────────────────────────
# GOSDT/LicketyRESPLIT+TGB → n_leaves, XGBoost → total_leaves

fig, ax = plt.subplots(figsize=(9, 4))
leaf_vals = {}
for model in MODELS:
    metric = "total_leaves" if model == "XGBoost" else "n_leaves"
    leaf_vals[model] = [data[ds].get(model, {}).get(metric, float("nan")) for ds in DATASETS]
bar_chart(ax, leaf_vals, "Number of Leaves")
ax.set_title("Tree Size (Leaves): GOSDT vs LicketyRESPLIT+TGB vs XGBoost",
             fontweight="bold")
plt.tight_layout()
out = PLOTS_DIR / "tgb_tree_size.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# ── Figure 5: Rashomon Set Size (LicketyRESPLIT+TGB only) ────────────────────

fig, ax = plt.subplots(figsize=(6, 4))
rashomon_vals = [data[ds].get("LicketyRESPLIT+TGB", {}).get("n_trees_in_set", float("nan"))
                 for ds in DATASETS]
ax.bar(x, rashomon_vals, width=0.5, color=COLORS["LicketyRESPLIT+TGB"], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(DATASETS, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Rashomon Set Size")
ax.set_title("LicketyRESPLIT+TGB: Rashomon Set Size", fontweight="bold")
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
out = PLOTS_DIR / "tgb_rashomon_set_size.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

print(f"\nAll TGB plots saved to: {PLOTS_DIR}")
