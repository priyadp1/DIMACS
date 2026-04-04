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

DATASETS = ["bike", "breast_cancer", "compas", "spambase", "diabetes", "diabetes_smote",
            "creditcard_fraud", "creditcard_fraud_smote"]

MODELS = ["GOSDT", "LicketyRESPLIT+TGB", "XGBoost"]

COLORS = {
    "GOSDT":              "#2ca02c",
    "LicketyRESPLIT+TGB": "#4C72B0",
    "XGBoost":           "#d62728",
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

def parse_gradient_boosting_acc(path):
    text = path.read_text()
    acc = re.search(r"GBDT warm-label ensemble accuracy on test set:\s*([\d.]+)", text)
    return float(acc.group(1)) if acc else None


# ── Load all results ──────────────────────────────────────────────────────────

data = {}   # data[dataset][model] = dict of metrics

for dataset in DATASETS:
    d = {}
    ds_dir = RESULTS_DIR / dataset

    files = {
        "GOSDT":              ("gosdt_results.txt",                    "gosdt_tree_size.json",                    parse_gosdt),
        "LicketyRESPLIT+TGB": ("licketyresplit_binarized_results.txt", "licketyresplit_binarized_tree_size.json", parse_licketyresplit_bin),
        "XGBoost":            ("xgboost_binarized_results.txt",          "xgboost_tree_size_binarized.json",        parse_xgboost),
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
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    ax.grid(axis="y", linestyle="--", alpha=0.5)


# ── Figure 1: Test Accuracy ───────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 6))
bar_chart(ax, get_vals("accuracy"), "Test Accuracy")
ax.set_title("Test Accuracy: GOSDT vs LicketyRESPLIT+TGB vs XGBoost",
             fontweight="bold")
plt.tight_layout()
out = PLOTS_DIR / "tgb_accuracy.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")


# ── Figure 2: Ensemble Accuracy (LicketyRESPLIT) vs Test Accuracy (others) ───

fig, ax = plt.subplots(figsize=(12, 6))
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

fig, ax = plt.subplots(figsize=(12, 6))
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

fig, ax = plt.subplots(figsize=(12, 6))
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

fig, ax = plt.subplots(figsize=(12, 6))
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

# ── XGBoost: with TGB vs without TGB ─────────────────────────────────────────

XGB_COLORS = {"XGBoost (raw)": "#ff7f0e", "XGBoost+TGB": "#d62728"}
xgb_offsets = np.array([-bar_w / 2, bar_w / 2])

xgb_raw = {}
for dataset in DATASETS:
    res_path = RESULTS_DIR / dataset / "xgboost_results.txt"
    sz_path  = RESULTS_DIR / dataset / "xgboost_tree_size.json"
    if res_path.exists():
        xgb_raw[dataset] = {**parse_xgboost(res_path), **(parse_tree_size(sz_path) if sz_path.exists() else {})}
    else:
        xgb_raw[dataset] = {}

xgb_tgb = {ds: data[ds].get("XGBoost", {}) for ds in DATASETS}


def xgb_bar_chart(ax, raw_vals, tgb_vals, ylabel):
    ax.bar(x + xgb_offsets[0], raw_vals, width=bar_w,
           label="XGBoost (raw)", color=XGB_COLORS["XGBoost (raw)"], alpha=0.85)
    ax.bar(x + xgb_offsets[1], tgb_vals, width=bar_w,
           label="XGBoost+TGB",  color=XGB_COLORS["XGBoost+TGB"],  alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    ax.grid(axis="y", linestyle="--", alpha=0.5)


# Figure 6: XGBoost accuracy — raw vs TGB
fig, ax = plt.subplots(figsize=(12, 6))
xgb_bar_chart(
    ax,
    [xgb_raw[ds].get("accuracy", float("nan")) for ds in DATASETS],
    [xgb_tgb[ds].get("accuracy", float("nan")) for ds in DATASETS],
    "Test Accuracy",
)
ax.set_title("XGBoost: Raw Features vs TGB Features — Test Accuracy", fontweight="bold")
plt.tight_layout()
out = PLOTS_DIR / "xgb_tgb_vs_raw_accuracy.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# Figure 7: XGBoost training time — raw vs TGB
fig, ax = plt.subplots(figsize=(12, 6))
xgb_bar_chart(
    ax,
    [xgb_raw[ds].get("duration_sec", float("nan")) for ds in DATASETS],
    [xgb_tgb[ds].get("duration_sec", float("nan")) for ds in DATASETS],
    "Training Time (s)",
)
ax.set_title("XGBoost: Raw Features vs TGB Features — Training Time", fontweight="bold")
plt.tight_layout()
out = PLOTS_DIR / "xgb_tgb_vs_raw_duration.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# Figure 8: XGBoost tree size (total leaves) — raw vs TGB
fig, ax = plt.subplots(figsize=(12, 6))
xgb_bar_chart(
    ax,
    [xgb_raw[ds].get("total_leaves", float("nan")) for ds in DATASETS],
    [xgb_tgb[ds].get("total_leaves", float("nan")) for ds in DATASETS],
    "Total Leaves",
)
ax.set_title("XGBoost: Raw Features vs TGB Features — Tree Size (Leaves)", fontweight="bold")
plt.tight_layout()
out = PLOTS_DIR / "xgb_tgb_vs_raw_tree_size.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")


#Figure 9: Gradient Boosting warm-label ensemble test accuracy on each dataset
TGB_DIR = BASEDIR / "TGB_Variables"
path1 = TGB_DIR / "bike" / "gbdt_warm_label_results.txt"
path2 = TGB_DIR / "breast_cancer" / "gbdt_warm_label_results.txt"
path3 = TGB_DIR / "compas" / "gbdt_warm_label_results.txt"
path4 = TGB_DIR / "spambase" / "gbdt_warm_label_results.txt"
path5 = TGB_DIR / "diabetes" / "gbdt_warm_label_results.txt"
path6 = TGB_DIR / "diabetes_smote" / "gbdt_warm_label_results.txt"
path7 = TGB_DIR / "creditcard_fraud" / "gbdt_warm_label_results.txt"
path8 = TGB_DIR / "creditcard_fraud_smote" / "gbdt_warm_label_results.txt"
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x, [parse_gradient_boosting_acc(path) for path in [path1, path2, path3, path4, path5, path6, path7, path8]], width=0.5, color="#ff7f0e", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(DATASETS, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Warm-label Ensemble Test Accuracy")
ax.set_title("Gradient Boosting: Warm-label Ensemble Test Accuracy", fontweight="bold")
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
out = PLOTS_DIR / "gradient_boosting_warm_label_ensemble_accuracy.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# Figure 10: Separate bar chart per dataset — GOSDT, LicketyRESPLIT+TGB,
# XGBoost+TGB, and GBDT warm-label ensemble accuracy side by side
gbdt_paths = [path1, path2, path3, path4, path5, path6, path7, path8]
fig10_models = ["GOSDT", "LicketyRESPLIT+TGB", "XGBoost", "GBDT Warm-label"]
fig10_colors = {**COLORS, "GBDT Warm-label": "#ff7f0e"}

for dataset, gbdt_path in zip(DATASETS, gbdt_paths):
    metric_map = {"GOSDT": "accuracy", "LicketyRESPLIT+TGB": "ensemble_accuracy", "XGBoost": "accuracy"}
    vals = [data[dataset].get(m, {}).get(metric_map.get(m, "accuracy"), float("nan"))
            for m in ["GOSDT", "LicketyRESPLIT+TGB", "XGBoost"]]
    vals.append(parse_gradient_boosting_acc(gbdt_path))

    fig, ax = plt.subplots(figsize=(12, 6))
    xi = np.arange(len(fig10_models))
    for i, (model, val) in enumerate(zip(fig10_models, vals)):
        ax.bar(xi[i], val, width=0.6, label=model, color=fig10_colors[model], alpha=0.85)

    ax.set_xticks(xi)
    ax.set_xticklabels(fig10_models, rotation=25, ha="right", fontsize=8)
    ax.set_title(f"{dataset}: GOSDT vs LicketyRESPLIT+TGB vs XGBoost+TGB vs GBDT Warm-label",
                 fontsize=10, fontweight="bold")
    ax.set_ylabel("Test Accuracy")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out = PLOTS_DIR / f"tgb_accuracy_comparison_{dataset}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

#Figure 11 SMOTE vs No SMOTE for diabetes dataset (Warm labels)
path_smote = TGB_DIR / "diabetes_smote" / "gbdt_warm_label_results.txt"
path_no_smote = TGB_DIR / "diabetes" / "gbdt_warm_label_results.txt"
print(f"\nGradient Boosting warm-label ensemble accuracy on diabetes with SMOTE: {parse_gradient_boosting_acc(path_smote):.4f}")
print(f"Gradient Boosting warm-label ensemble accuracy on diabetes without SMOTE: {parse_gradient_boosting_acc(path_no_smote):.4f}")
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(["Diabetes (SMOTE)", "Diabetes (No SMOTE)"],
       [parse_gradient_boosting_acc(path_smote), parse_gradient_boosting_acc(path_no_smote)],
       width=0.5, color=["#d62728", "#ff7f0e"], alpha=0.85)
ax.set_ylabel("Warm-label Ensemble Test Accuracy")
ax.set_title("Gradient Boosting: Warm-label Ensemble Test Accuracy on Diabetes with vs without SMOTE", fontweight="bold")
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
out = PLOTS_DIR / "gradient_boosting_diabetes_smote_vs_no_smote_accuracy.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

#Figure 12 SMOTE vs No SMOTE Rashamon set size for diabetes dataset (LicketyRESPLIT+TGB)
path_smote_sz = RESULTS_DIR / "diabetes_smote" / "licketyresplit_binarized_tree_size.json"
path_no_smote_sz = RESULTS_DIR / "diabetes" / "licketyresplit_binarized_tree_size.json"
smote_sz = parse_tree_size(path_smote_sz).get("n_trees_in_set", float("nan"))
no_smote_sz = parse_tree_size(path_no_smote_sz).get("n_trees_in_set", float("nan"))
print(f"\nLicketyRESPLIT+TGB Rashomon set size on diabetes with SMOTE: {smote_sz}")
print(f"LicketyRESPLIT+TGB Rashomon set size on diabetes without SMOTE: {no_smote_sz}")
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(["Diabetes (SMOTE)", "Diabetes (No SMOTE)"],
       [smote_sz, no_smote_sz],
       width=0.5, color=["#4C72B0", "#2ca02c"], alpha=0.85)
ax.set_ylabel("Rashomon Set Size")
ax.set_title("LicketyRESPLIT+TGB: Rashomon Set Size on Diabetes with vs without SMOTE", fontweight="bold")
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
out = PLOTS_DIR / "licketyresplit_diabetes_smote_vs_no_smote_rashomon_set_size.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

#Figure 13 SMOTE vs no SMOTE Rashmomon set size for creditcard fraud dataset (LicketyRESPLIT+TGB)
path_smote_sz_cc = RESULTS_DIR / "creditcard_fraud_smote" / "licketyresplit_binarized_tree_size.json"
path_no_smote_sz_cc = RESULTS_DIR / "creditcard_fraud" / "licketyresplit_binarized_tree_size.json"
smote_sz_cc = parse_tree_size(path_smote_sz_cc).get("n_trees_in_set", float("nan"))
no_smote_sz_cc = parse_tree_size(path_no_smote_sz_cc).get("n_trees_in_set", float("nan"))
print(f"\nLicketyRESPLIT+TGB Rashomon set size on creditcard fraud with SMOTE: {smote_sz_cc}") 
print(f"LicketyRESPLIT+TGB Rashomon set size on creditcard fraud without SMOTE: {no_smote_sz_cc}")
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(["Credit Card Fraud (SMOTE)", "Credit Card Fraud (No SMOTE)"],
       [smote_sz_cc, no_smote_sz_cc],
       width=0.5, color=["#4C72B0", "#2ca02c"], alpha=0.85)
ax.set_ylabel("Rashomon Set Size")
ax.set_title("LicketyRESPLIT+TGB: Rashomon Set Size on Credit Card Fraud with vs without SMOTE", fontweight="bold")
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
out = PLOTS_DIR / "licketyresplit_creditcard_fraud_smote_vs_no_smote_rashomon_set_size.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

