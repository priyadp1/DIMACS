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
NO_TGB_DATASETS = ["bike", "breast_cancer", "compas", "spambase"]

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
    n_leaves = re.search(r"Tree Size:\s*(\d+)\s*leaves", text)
    return {
        "accuracy":          float(acc.group(1))       if acc       else None,
        "ensemble_accuracy": float(ens_acc.group(1))   if ens_acc   else None,
        "duration_sec":      float(duration.group(1))  if duration  else None,
        "n_leaves":          int(n_leaves.group(1))    if n_leaves  else None,
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
    if not path.exists():
        return float("nan")
    text = path.read_text()
    acc = re.search(r"GBDT warm-label ensemble accuracy on test set:\s*([\d.]+)", text)
    return float(acc.group(1)) if acc else float("nan")


# ── Load all results ──────────────────────────────────────────────────────────

data = {}   # data[dataset][model] = dict of metrics

for dataset in DATASETS:
    d = {}
    ds_dir = RESULTS_DIR / dataset

    files = {
        "GOSDT":              ("gosdt_results.txt",                    "gosdt_tree_size.json",                    parse_gosdt),
        "LicketyRESPLIT+TGB": ("licketyresplit_binarized_results.txt", "licketyresplit_binarized_tree_size.json", parse_licketyresplit_bin),
        "LicketyRESPLIT":     ("licketyresplit_binarized_results.txt", "licketyresplit_binarized_tree_size.json", parse_licketyresplit_bin),
        "XGBoost":            ("xgboost_binarized_results.txt",        "xgboost_tree_size_binarized.json",        parse_xgboost),
    }

    for model, (res_f, sz_f, parser) in files.items():
        res_path = ds_dir / res_f
        sz_path  = ds_dir / sz_f
        if res_path.exists():
            d[model] = {**parser(res_path), **(parse_tree_size(sz_path) if sz_path.exists() else {})}
        else:
            d[model] = {}

    # Load benchmark models from benchmarks_TGB_results
    bench_dir = BASEDIR / "benchmarks_TGB_results" / dataset
    for model, res_f, sz_f in [
        ("LightGBM",    "lightgbm_results.txt",    "lightgbm_tree_size.json"),
        ("CatBoost",    "catboost_results.txt",    "catboost_tree_size.json"),
        ("LicketySPLIT", "licketysplit_results.txt", None),
        ("SPLIT",        "split_results.txt",        None),
    ]:
        res_path = bench_dir / res_f
        parsed = parse_licketyresplit_bin(res_path) if res_path.exists() else {}
        if sz_f:
            sz_path = bench_dir / sz_f
            parsed = {**parsed, **(parse_tree_size(sz_path) if sz_path.exists() else {})}
        d[model] = parsed

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

# ── Figure 14: XGBoost vs LightGBM vs CatBoost (all with TGB features) ─────────────────────────────────────────
benchmarks_dir = BASEDIR / "benchmarks_TGB_results"
models = ["XGBoost", "LightGBM", "CatBoost"]
model_colors = {"XGBoost": "#d62728", "LightGBM": "#2ca02c", "CatBoost": "#4C72B0"}
fig, ax = plt.subplots(figsize=(12, 6))
fig14_offsets = np.array([-1, 0, 1]) * bar_w
for m_idx, model in enumerate(models):
    vals = [data[ds].get(model, {}).get("accuracy", float("nan")) for ds in DATASETS]
    ax.bar(x + fig14_offsets[m_idx], vals, width=bar_w,
           label=model, color=model_colors[model], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(DATASETS, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Test Accuracy")
ax.set_title("XGBoost vs LightGBM vs CatBoost (all with TGB features) — Test Accuracy", fontweight="bold")
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
out = PLOTS_DIR / "xgboost_lightgbm_catboost_tgb_accuracy.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# Layout shared by Figures 15-18
x_dt       = np.arange(len(DATASETS)) * 1.8
bw_dt      = 0.32
dt_offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bw_dt

#Figure 15 Decision Tree Accuracies Across Datasets (TGB Features)
opt_models = ["GOSDT", "LicketyRESPLIT", "LicketySPLIT", "SPLIT"]
model_colors = {"GOSDT": "#2ca02c", "LicketyRESPLIT": "#4C72B0", "LicketySPLIT": "#ff7f0e", "SPLIT": "#d62728"}
fig, ax = plt.subplots(figsize=(18, 7))
for m_idx, model in enumerate(opt_models):
    vals = [data[ds].get(model, {}).get("accuracy", float("nan")) for ds in DATASETS]
    ax.bar(x_dt + dt_offsets[m_idx], vals, width=bw_dt,
           label=model, color=model_colors[model], alpha=0.85)
ax.set_xticks(x_dt)
ax.set_xticklabels(DATASETS, rotation=20, ha="right", fontsize=10)
ax.set_ylabel("Test Accuracy", fontsize=11)
ax.set_title("Decision Tree Accuracies Across Datasets (TGB Features)", fontweight="bold", fontsize=12)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
ax.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
out = PLOTS_DIR / "decision_tree_accuracies_tgb.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

def dt_vs_bar(ax, boost_accs, boost_label, boost_color):
    for i, dataset in enumerate(DATASETS):
        gosdt_acc        = data[dataset].get("GOSDT",         {}).get("accuracy", float("nan"))
        licketyresplit_acc = data[dataset].get("LicketyRESPLIT", {}).get("accuracy", float("nan"))
        licketysplit_acc = data[dataset].get("LicketySPLIT",  {}).get("accuracy", float("nan"))
        split_acc        = data[dataset].get("SPLIT",         {}).get("accuracy", float("nan"))
        ax.bar(x_dt[i] + dt_offsets[0], gosdt_acc,          width=bw_dt, label="GOSDT"          if i == 0 else None, color="#2ca02c", alpha=0.85)
        ax.bar(x_dt[i] + dt_offsets[1], licketyresplit_acc, width=bw_dt, label="LicketyRESPLIT" if i == 0 else None, color="#4C72B0", alpha=0.85)
        ax.bar(x_dt[i] + dt_offsets[2], licketysplit_acc,   width=bw_dt, label="LicketySPLIT"   if i == 0 else None, color="#FF7F0E", alpha=0.85)
        ax.bar(x_dt[i] + dt_offsets[3], split_acc,          width=bw_dt, label="SPLIT"          if i == 0 else None, color="#9467bd", alpha=0.85)
    ax.plot(x_dt, boost_accs, label=boost_label, color=boost_color, marker="o", zorder=5)
    ax.set_xticks(x_dt)
    ax.set_xticklabels(DATASETS, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Test Accuracy", fontsize=11)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
    ax.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

#Figure 16: Decision Tree Accuracies vs XGBoost Accuracy (TGB features)
fig, ax = plt.subplots(figsize=(18, 7))
xgboost_accs = [data[ds].get("XGBoost", {}).get("accuracy", float("nan")) for ds in DATASETS]
dt_vs_bar(ax, xgboost_accs, "XGBoost (TGB)", "#d62728")
ax.set_title("Decision Tree Accuracies vs XGBoost Accuracy (TGB features)", fontweight="bold", fontsize=12)
plt.tight_layout()
out = PLOTS_DIR / "decision_tree_vs_xgboost_tgb_accuracy.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

#Figure 17 Decision Tree Accuracies vs LightGBM Accuracy (TGB features)
fig, ax = plt.subplots(figsize=(18, 7))
lightgbm_accs = [data[ds].get("LightGBM", {}).get("accuracy", float("nan")) for ds in DATASETS]
dt_vs_bar(ax, lightgbm_accs, "LightGBM (TGB)", "#d62728")
ax.set_title("Decision Tree Accuracies vs LightGBM Accuracy (TGB features)", fontweight="bold", fontsize=12)
plt.tight_layout()
out = PLOTS_DIR / "decision_tree_vs_lightgbm_tgb_accuracy.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

#Figure 18 Decision Tree Accuracies vs CatBoost Accuracy (TGB features)
fig, ax = plt.subplots(figsize=(18, 7))
catboost_accs = [data[ds].get("CatBoost", {}).get("accuracy", float("nan")) for ds in DATASETS]
dt_vs_bar(ax, catboost_accs, "CatBoost (TGB)", "#d62728")
ax.set_title("Decision Tree Accuracies vs CatBoost Accuracy (TGB features)", fontweight="bold", fontsize=12)
plt.tight_layout()
out = PLOTS_DIR / "decision_tree_vs_catboost_tgb_accuracy.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")


#Rashomon Set Sizes for GOSDT, LicketyRESPLIT, LicketySPLIT, and SPLIT (TGB features)
fig, ax = plt.subplots(figsize=(12, 6))
for i, dataset in enumerate(DATASETS):
    gosdt_sz = data[dataset].get("GOSDT", {}).get("n_trees_in_set", float("nan"))
    licketyresplit_sz = data[dataset].get("LicketyRESPLIT", {}).get("n_trees_in_set", float("nan"))
    licketysplit_sz = data[dataset].get("LicketySPLIT", {}).get("n_trees_in_set", float("nan"))
    split_sz = data[dataset].get("SPLIT", {}).get("n_trees_in_set", float("nan"))
    ax.bar(i - 1.5 * bar_w, gosdt_sz, width=bar_w, label="GOSDT" if i == 0 else None, color="#2ca02c", alpha=0.85)
    ax.bar(i - 0.5 * bar_w, licketyresplit_sz, width=bar_w, label="LicketyRESPLIT+TGB" if i == 0 else None, color="#4C72B0", alpha=0.85)
    ax.bar(i + 0.5 * bar_w, licketysplit_sz, width=bar_w, label="LicketySPLIT+TGB" if i == 0 else None, color="#FF7F0E", alpha=0.85)
    ax.bar(i + 1.5 * bar_w, split_sz, width=bar_w, label="SPLIT" if i == 0 else None, color="#d62728", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(DATASETS, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Rashomon Set Size")
ax.set_title("Rashomon Set Sizes for GOSDT, LicketyRESPLIT, LicketySPLIT, and SPLIT (TGB features)", fontweight="bold")
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
out = PLOTS_DIR / "decision_tree_rashomon_set_sizes_tgb.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# Figure 19: Rashomon Set Size (total_leaves) — XGBoost vs LightGBM vs CatBoost (TGB features)
boost_models  = ["XGBoost", "LightGBM", "CatBoost"]
boost_colors  = {"XGBoost": "#d62728", "LightGBM": "#2ca02c", "CatBoost": "#4C72B0"}
boost_offsets = np.array([-1, 0, 1]) * bw_dt

fig, ax = plt.subplots(figsize=(18, 7))
for m_idx, model in enumerate(boost_models):
    vals = [data[ds].get(model, {}).get("total_leaves", float("nan")) for ds in DATASETS]
    ax.bar(x_dt + boost_offsets[m_idx], vals, width=bw_dt,
           label=model, color=boost_colors[model], alpha=0.85)
ax.set_xticks(x_dt)
ax.set_xticklabels(DATASETS, rotation=20, ha="right", fontsize=10)
ax.set_ylabel("Total Leaves", fontsize=11)
ax.set_title("Rashomon Set Size (Total Leaves): XGBoost vs LightGBM vs CatBoost (TGB features)",
             fontweight="bold", fontsize=12)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
ax.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
out = PLOTS_DIR / "xgb_lgbm_catboost_rashomon_set_size.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")


# ── Load No-TGB results ───────────────────────────────────────────────────────

NO_TGB_DIR = BASEDIR / "benchmarks_no_TGB_results"
NO_TGB_MODELS = ["XGBoost", "LightGBM", "CatBoost", "LicketyRESPLIT", "LicketySPLIT", "GOSDT", "SPLIT"]

no_tgb_data = {}
for dataset in NO_TGB_DATASETS:
    d2 = {}
    nd = NO_TGB_DIR / dataset
    for model, res_f, sz_f, parser in [
        ("XGBoost",       "xgboost_binarized_results.txt",         "xgboost_tree_size_binarized.json",        parse_xgboost),
        ("LightGBM",      "lightgbm_results.txt",                  "lightgbm_tree_size.json",                 parse_xgboost),
        ("CatBoost",      "catboost_results.txt",                  "catboost_tree_size.json",                 parse_xgboost),
        ("LicketyRESPLIT","licketyresplit_binarized_results.txt",  "licketyresplit_binarized_tree_size.json", parse_licketyresplit_bin),
        ("LicketySPLIT",  "licketysplit_results.txt",              None,                                      parse_licketyresplit_bin),
        ("GOSDT",         "gosdt_results.txt",                     "gosdt_tree_size.json",                    parse_gosdt),
        ("SPLIT",         "split_results.txt",                     None,                                      parse_licketyresplit_bin),
    ]:
        res_path = nd / res_f
        parsed = parser(res_path) if res_path.exists() else {}
        if sz_f:
            sz_path = nd / sz_f
            parsed = {**parsed, **(parse_tree_size(sz_path) if sz_path.exists() else {})}
        d2[model] = parsed
    no_tgb_data[dataset] = d2

# Shared layout for No-TGB figures (7 models, 4 datasets)
no_tgb_colors = {
    "XGBoost":        "#d62728",
    "LightGBM":       "#2ca02c",
    "CatBoost":       "#4C72B0",
    "LicketyRESPLIT": "#9467bd",
    "LicketySPLIT":   "#FF7F0E",
    "GOSDT":          "#8c564b",
    "SPLIT":          "#e377c2",
}
n_no_tgb   = len(NO_TGB_MODELS)
bw_nt      = 0.25
nt_offsets = (np.arange(n_no_tgb) - (n_no_tgb - 1) / 2) * bw_nt
x_nt       = np.arange(len(NO_TGB_DATASETS)) * 2.4


def no_tgb_bar(ax, metric, ylabel):
    for m_idx, model in enumerate(NO_TGB_MODELS):
        vals = [no_tgb_data[ds].get(model, {}).get(metric, float("nan")) for ds in NO_TGB_DATASETS]
        ax.bar(x_nt + nt_offsets[m_idx], vals, width=bw_nt,
               label=model, color=no_tgb_colors[model], alpha=0.85)
    ax.set_xticks(x_nt)
    ax.set_xticklabels(NO_TGB_DATASETS, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
    ax.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    ax.grid(axis="y", linestyle="--", alpha=0.5)


# Figure 20: Test Accuracy — all 7 models, No-TGB features
fig, ax = plt.subplots(figsize=(20, 7))
no_tgb_bar(ax, "accuracy", "Test Accuracy")
ax.set_title("Test Accuracy (No TGB): XGBoost vs LightGBM vs CatBoost vs LicketyRESPLIT vs LicketySPLIT vs GOSDT vs SPLIT",
             fontweight="bold", fontsize=11)
plt.tight_layout()
out = PLOTS_DIR / "no_tgb_accuracy_all.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# Figure 21: Training Time — all 7 models, No-TGB features
fig, ax = plt.subplots(figsize=(20, 7))
no_tgb_bar(ax, "duration_sec", "Training Time (s)")
ax.set_title("Training Time (No TGB): XGBoost vs LightGBM vs CatBoost vs LicketyRESPLIT vs LicketySPLIT vs GOSDT vs SPLIT",
             fontweight="bold", fontsize=11)
plt.tight_layout()
out = PLOTS_DIR / "no_tgb_duration_all.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# Figure 22: Total Leaves — all 7 models, No-TGB features
fig, ax = plt.subplots(figsize=(20, 7))
for m_idx, model in enumerate(NO_TGB_MODELS):
    metric = "n_leaves" if model in ("GOSDT", "LicketyRESPLIT") else "total_leaves"
    vals = [no_tgb_data[ds].get(model, {}).get(metric, float("nan")) for ds in NO_TGB_DATASETS]
    ax.bar(x_nt + nt_offsets[m_idx], vals, width=bw_nt,
           label=model, color=no_tgb_colors[model], alpha=0.85)
ax.set_xticks(x_nt)
ax.set_xticklabels(NO_TGB_DATASETS, rotation=20, ha="right", fontsize=10)
ax.set_ylabel("Number of Leaves", fontsize=11)
ax.set_title("Tree Size / Leaves (No TGB): XGBoost vs LightGBM vs CatBoost vs LicketyRESPLIT vs LicketySPLIT vs GOSDT vs SPLIT",
             fontweight="bold", fontsize=11)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
ax.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
out = PLOTS_DIR / "no_tgb_tree_size_all.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# Figure 23-25: Decision Trees vs boosting models, No-TGB
dt_no_tgb_models  = ["GOSDT", "LicketyRESPLIT", "LicketySPLIT", "SPLIT"]
dt_no_tgb_colors  = {m: no_tgb_colors[m] for m in dt_no_tgb_models}
dt_nt_offsets     = np.array([-1.5, -0.5, 0.5, 1.5]) * bw_dt
x_nt4             = np.arange(len(NO_TGB_DATASETS)) * 1.8

def dt_no_tgb_vs_bar(ax, boost_accs, boost_label):
    for i, dataset in enumerate(NO_TGB_DATASETS):
        for j, model in enumerate(dt_no_tgb_models):
            val = no_tgb_data[dataset].get(model, {}).get("accuracy", float("nan"))
            ax.bar(x_nt4[i] + dt_nt_offsets[j], val, width=bw_dt,
                   label=model if i == 0 else None,
                   color=dt_no_tgb_colors[model], alpha=0.85)
    ax.plot(x_nt4, boost_accs, label=boost_label, color="#d62728", marker="o", zorder=5)
    ax.set_xticks(x_nt4)
    ax.set_xticklabels(NO_TGB_DATASETS, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Test Accuracy", fontsize=11)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
    ax.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

fig, ax = plt.subplots(figsize=(18, 7))
xgb_nt_accs = [no_tgb_data[ds].get("XGBoost", {}).get("accuracy", float("nan")) for ds in NO_TGB_DATASETS]
dt_no_tgb_vs_bar(ax, xgb_nt_accs, "XGBoost (No TGB)")
ax.set_title("Decision Tree Accuracies vs XGBoost Accuracy (No TGB features)", fontweight="bold", fontsize=12)
plt.tight_layout()
out = PLOTS_DIR / "no_tgb_decision_tree_vs_xgboost.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# Figure 24: Decision Trees vs LightGBM accuracy, No-TGB
fig, ax = plt.subplots(figsize=(18, 7))
lgbm_nt_accs = [no_tgb_data[ds].get("LightGBM", {}).get("accuracy", float("nan")) for ds in NO_TGB_DATASETS]
dt_no_tgb_vs_bar(ax, lgbm_nt_accs, "LightGBM (No TGB)")
ax.set_title("Decision Tree Accuracies vs LightGBM Accuracy (No TGB features)", fontweight="bold", fontsize=12)
plt.tight_layout()
out = PLOTS_DIR / "no_tgb_decision_tree_vs_lightgbm.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# Figure 25: Decision Trees vs CatBoost accuracy, No-TGB
fig, ax = plt.subplots(figsize=(18, 7))
cb_nt_accs = [no_tgb_data[ds].get("CatBoost", {}).get("accuracy", float("nan")) for ds in NO_TGB_DATASETS]
dt_no_tgb_vs_bar(ax, cb_nt_accs, "CatBoost (No TGB)")
ax.set_title("Decision Tree Accuracies vs CatBoost Accuracy (No TGB features)", fontweight="bold", fontsize=12)
plt.tight_layout()
out = PLOTS_DIR / "no_tgb_decision_tree_vs_catboost.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# Figure 26: Rashomon Set Size (total_leaves) — XGBoost vs LightGBM vs CatBoost, No-TGB
x_nt3        = np.arange(len(NO_TGB_DATASETS)) * 1.8
nt3_offsets  = np.array([-1, 0, 1]) * bw_dt
fig, ax = plt.subplots(figsize=(18, 7))
for m_idx, model in enumerate(["XGBoost", "LightGBM", "CatBoost"]):
    vals = [no_tgb_data[ds].get(model, {}).get("total_leaves", float("nan")) for ds in NO_TGB_DATASETS]
    ax.bar(x_nt3 + nt3_offsets[m_idx], vals, width=bw_dt,
           label=model, color=no_tgb_colors[model], alpha=0.85)
ax.set_xticks(x_nt3)
ax.set_xticklabels(NO_TGB_DATASETS, rotation=20, ha="right", fontsize=10)
ax.set_ylabel("Total Leaves", fontsize=11)
ax.set_title("Rashomon Set Size (Total Leaves): XGBoost vs LightGBM vs CatBoost (No TGB features)",
             fontweight="bold", fontsize=12)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
ax.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
out = PLOTS_DIR / "no_tgb_xgb_lgbm_catboost_rashomon_set_size.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# Figure 27: Decision Tree Accuracies, No-TGB
fig, ax = plt.subplots(figsize=(18, 7))
for m_idx, model in enumerate(dt_no_tgb_models):
    vals = [no_tgb_data[ds].get(model, {}).get("accuracy", float("nan")) for ds in NO_TGB_DATASETS]
    ax.bar(x_nt4 + dt_nt_offsets[m_idx], vals, width=bw_dt,
           label=model, color=dt_no_tgb_colors[model], alpha=0.85)
ax.set_xticks(x_nt4)
ax.set_xticklabels(NO_TGB_DATASETS, rotation=20, ha="right", fontsize=10)
ax.set_ylabel("Test Accuracy", fontsize=11)
ax.set_title("Decision Tree Accuracies (No TGB features)", fontweight="bold", fontsize=12)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
ax.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
out = PLOTS_DIR / "no_tgb_decision_tree_accuracies.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# Figure 28: Decision Tree Rashomon Set Sizes, No-TGB
# (GOSDT and LicketyRESPLIT have n_trees_in_set; LicketySPLIT and SPLIT have no tree size JSON)
fig, ax = plt.subplots(figsize=(18, 7))
for m_idx, model in enumerate(dt_no_tgb_models):
    vals = [no_tgb_data[ds].get(model, {}).get("n_trees_in_set", float("nan")) for ds in NO_TGB_DATASETS]
    ax.bar(x_nt4 + dt_nt_offsets[m_idx], vals, width=bw_dt,
           label=model, color=dt_no_tgb_colors[model], alpha=0.85)
ax.set_xticks(x_nt4)
ax.set_xticklabels(NO_TGB_DATASETS, rotation=20, ha="right", fontsize=10)
ax.set_ylabel("Rashomon Set Size", fontsize=11)
ax.set_title("Decision Tree Rashomon Set Sizes (No TGB features)", fontweight="bold", fontsize=12)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
ax.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
out = PLOTS_DIR / "no_tgb_decision_tree_rashomon_set.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# Figure 29: XGBoost vs LightGBM vs CatBoost Accuracy, No-TGB
boost3_offsets = np.array([-1, 0, 1]) * bw_dt
fig, ax = plt.subplots(figsize=(18, 7))
for m_idx, model in enumerate(["XGBoost", "LightGBM", "CatBoost"]):
    vals = [no_tgb_data[ds].get(model, {}).get("accuracy", float("nan")) for ds in NO_TGB_DATASETS]
    ax.bar(x_nt4 + boost3_offsets[m_idx], vals, width=bw_dt,
           label=model, color=no_tgb_colors[model], alpha=0.85)
ax.set_xticks(x_nt4)
ax.set_xticklabels(NO_TGB_DATASETS, rotation=20, ha="right", fontsize=10)
ax.set_ylabel("Test Accuracy", fontsize=11)
ax.set_title("XGBoost vs LightGBM vs CatBoost Accuracy (No TGB features)", fontweight="bold", fontsize=12)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
ax.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
out = PLOTS_DIR / "no_tgb_xgb_lgbm_catboost_accuracy.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")


# ── Figures 30+: Bike dataset parameter sweeps ────────────────────────────────
# Grouped bar charts: one bar per model per x-tick.
# Accuracy: all 7 models.  Rashomon: n_trees for boosting, n_trees_in_set for
# LicketyRESPLIT, n_leaves for GOSDT/SPLIT/LicketySPLIT (single-tree models).

_bike_sweep_dir = BASEDIR / "benchmarks_TGB_results" / "bike"
_nest_values    = [40, 100, 200]
_depth_values   = [1, 2, 3]

_all_sweep_models = [
    "XGBoost", "LightGBM", "CatBoost",
    "GOSDT", "LicketyRESPLIT", "LicketySPLIT", "SPLIT",
]
_sweep_model_colors = {
    "XGBoost":       "#d62728",
    "LightGBM":      "#2ca02c",
    "CatBoost":      "#4C72B0",
    "GOSDT":         "#8c564b",
    "LicketyRESPLIT":"#9467bd",
    "LicketySPLIT":  "#FF7F0E",
    "SPLIT":         "#e377c2",
}

def _load_sweep_all(nest, depth):
    pdir = _bike_sweep_dir / f"nest{nest}_depth{depth}"
    row = {}
    specs = [
        ("XGBoost",       "xgboost_binarized_results.txt",        "xgboost_tree_size_binarized.json",        parse_xgboost,            "n_trees"),
        ("LightGBM",      "lightgbm_results.txt",                  "lightgbm_tree_size.json",                 parse_xgboost,            "n_trees"),
        ("CatBoost",      "catboost_results.txt",                  "catboost_tree_size.json",                 parse_xgboost,            "n_trees"),
        ("GOSDT",         "gosdt_results.txt",                     "gosdt_tree_size.json",                    parse_gosdt,              "n_leaves"),
        ("LicketyRESPLIT","licketyresplit_binarized_results.txt",  "licketyresplit_binarized_tree_size.json", parse_licketyresplit_bin, "n_trees_in_set"),
        ("LicketySPLIT",  "licketysplit_results.txt",              None,                                      parse_licketyresplit_bin, None),
        ("SPLIT",         "split_results.txt",                     None,                                      parse_licketyresplit_bin, None),
    ]
    for model, res_f, sz_f, parser, rashomon_key in specs:
        res_path = pdir / res_f
        parsed   = parser(res_path) if res_path.exists() else {}
        acc      = parsed.get("accuracy", float("nan"))
        if sz_f and rashomon_key:
            sz_path  = pdir / sz_f
            rashomon = parse_tree_size(sz_path).get(rashomon_key, float("nan")) if sz_path.exists() else float("nan")
        else:
            rashomon = float("nan")
        row[model] = {"accuracy": acc, "rashomon": rashomon}
    return row


def _sweep_bar_plot(x_vals, x_label, bike_data_dict, metric, ylabel, title, out_path):
    n_models  = len(_all_sweep_models)
    n_x       = len(x_vals)
    bw        = 0.10
    offsets   = (np.arange(n_models) - (n_models - 1) / 2) * bw
    xi        = np.arange(n_x)

    fig, ax = plt.subplots(figsize=(max(14, n_x * n_models * bw * 6), 6))
    for m_idx, model in enumerate(_all_sweep_models):
        vals = [bike_data_dict[(xv, model)][metric] for xv in x_vals]
        ax.bar(xi + offsets[m_idx], vals, width=bw,
               label=model, color=_sweep_model_colors[model], alpha=0.85)
    ax.set_xticks(xi)
    ax.set_xticklabels([str(v) for v in x_vals], fontsize=11)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
    ax.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# Pre-load all bike sweep data (all 7 models)
_bike_data_all = {}
for _n in _nest_values:
    for _d in _depth_values:
        _row = _load_sweep_all(_n, _d)
        for _m in _all_sweep_models:
            _bike_data_all[(_n, _d, _m)] = _row[_m]

# Sweep over n_estimators for each fixed depth
for _d in _depth_values:
    _series = {(_n, _m): _bike_data_all[(_n, _d, _m)]
               for _n in _nest_values for _m in _all_sweep_models}
    _sweep_bar_plot(
        x_vals=_nest_values, x_label="n_estimators",
        bike_data_dict=_series, metric="accuracy", ylabel="Test Accuracy",
        title=f"Bike (depth={_d}): Accuracy vs n_estimators",
        out_path=PLOTS_DIR / f"bike_sweep_depth{_d}_accuracy.png",
    )
    _sweep_bar_plot(
        x_vals=_nest_values, x_label="n_estimators",
        bike_data_dict=_series, metric="rashomon", ylabel="Rashomon Set Size",
        title=f"Bike (depth={_d}): Rashomon Set Size vs n_estimators",
        out_path=PLOTS_DIR / f"bike_sweep_depth{_d}_rashomon.png",
    )

# Sweep over depth for each fixed n_estimators
for _n in _nest_values:
    _series = {(_d, _m): _bike_data_all[(_n, _d, _m)]
               for _d in _depth_values for _m in _all_sweep_models}
    _sweep_bar_plot(
        x_vals=_depth_values, x_label="max_depth",
        bike_data_dict=_series, metric="accuracy", ylabel="Test Accuracy",
        title=f"Bike (n_estimators={_n}): Accuracy vs depth",
        out_path=PLOTS_DIR / f"bike_sweep_nest{_n}_accuracy.png",
    )
    _sweep_bar_plot(
        x_vals=_depth_values, x_label="max_depth",
        bike_data_dict=_series, metric="rashomon", ylabel="Rashomon Set Size",
        title=f"Bike (n_estimators={_n}): Rashomon Set Size vs depth",
        out_path=PLOTS_DIR / f"bike_sweep_nest{_n}_rashomon.png",
    )

