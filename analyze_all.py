#!/usr/bin/env python3
"""
Analyze parameter sweep results for XGBoost, Threshold Guessing (GOSDT),
and LicketyRESPLIT across all datasets in model_results/.

Scans every subfolder and top-level result file in each dataset directory,
parses all .txt and .json files, and generates per-model and cross-model plots.
"""

import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASEDIR       = Path(__file__).resolve().parent
MODEL_RESULTS = BASEDIR / "model_results"
ANALYSIS_DIR  = BASEDIR / "analysis_figures"

# Folder name patterns for parameter-sweep subdirectories
LR_PATTERN  = re.compile(r'^(\d+)_([\d.]+)_([\d.]+)$')      # depth_lambda_rashomon
GSD_PATTERN = re.compile(r'^gosdt_(\d+)_([\d.e+\-]+)$')      # gosdt_depth_reg
XGB_PATTERN = re.compile(r'^xgboost_(\d+)_(\d+)$')           # xgboost_maxdepth_nest

# ============================
# PARSING HELPERS
# ============================

def read_txt(path):
    if not path.exists():
        return ""
    with open(path) as f:
        return f.read()

def read_json(path):
    if not path.exists():
        return {}
    with open(path) as f:
        try:
            return json.load(f)
        except Exception:
            return {}

def parse_accuracy(content):
    # Match "Accuracy: X" only at the start of a line (excludes Training/Ensemble prefixes)
    m = re.search(r"^Accuracy:\s*([0-9.]+)", content, re.MULTILINE)
    return float(m.group(1)) if m else None

def parse_ensemble_acc(content):
    m = re.search(r"^Ensemble Accuracy:\s*([0-9.]+)", content, re.MULTILINE)
    return float(m.group(1)) if m else None

def parse_training_acc(content):
    m = re.search(r"^Training Accuracy:\s*([0-9.]+)", content, re.MULTILINE)
    return float(m.group(1)) if m else None

def parse_duration(content):
    m = re.search(r"completed in\s+([\d.]+)\s+seconds", content)
    return float(m.group(1)) if m else None

def parse_n_trees_text(content):
    m = re.search(r"completed in.*?seconds with\s+(\d+)\s+trees", content)
    return int(m.group(1)) if m else None

def parse_macro_f1(content):
    m = re.search(r"macro avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)", content)
    return float(m.group(1)) if m else None

# ============================
# RESULT COLLECTORS
# ============================

def collect_licketyresplit(folder, depth, lambda_reg, rashomon):
    txt = read_txt(folder / "licketyresplit_results.txt")
    sz  = read_json(folder / "licketyresplit_tree_size.json")
    if not txt and not sz:
        return None
    n_trees = parse_n_trees_text(txt)
    n_leaves = sz.get("n_leaves")      if "error" not in sz else None
    n_nodes  = sz.get("n_nodes")       if "error" not in sz else None
    if n_trees is None:
        n_trees = sz.get("n_trees_in_set") if "error" not in sz else None
    return {
        "model":            "LicketyRESPLIT",
        "depth_budget":     depth,
        "lambda_reg":       lambda_reg,
        "rashomon_mult":    rashomon,
        "accuracy":         parse_accuracy(txt),
        "macro_f1":         parse_macro_f1(txt),
        "ensemble_accuracy": parse_ensemble_acc(txt),
        "n_leaves":         n_leaves,
        "n_nodes":          n_nodes,
        "n_trees_in_set":   n_trees,
        "duration":         parse_duration(txt),
    }

def collect_gosdt(folder, depth, reg):
    txt = read_txt(folder / "gosdt_results.txt")
    sz  = read_json(folder / "gosdt_tree_size.json")
    if not txt and not sz:
        return None
    return {
        "model":            "Threshold Guessing",
        "depth_budget":     depth,
        "regularization":   reg,
        "accuracy":         parse_accuracy(txt),
        "macro_f1":         parse_macro_f1(txt),
        "training_accuracy": parse_training_acc(txt),
        "n_leaves":         sz.get("n_leaves") if "error" not in sz else None,
        "n_nodes":          sz.get("n_nodes")  if "error" not in sz else None,
        "duration":         parse_duration(txt),
    }

def collect_xgboost(folder, max_depth, n_est):
    txt = read_txt(folder / "xgboost_results.txt")
    sz  = read_json(folder / "xgboost_tree_size.json")
    if not txt and not sz:
        return None
    return {
        "model":              "XGBoost",
        "max_depth":          max_depth,
        "n_estimators":       n_est,
        "accuracy":           parse_accuracy(txt),
        "macro_f1":           parse_macro_f1(txt),
        "n_trees":            sz.get("n_trees")              if "error" not in sz else None,
        "total_leaves":       sz.get("total_leaves")         if "error" not in sz else None,
        "total_nodes":        sz.get("total_nodes")          if "error" not in sz else None,
        "avg_leaves_per_tree": sz.get("avg_leaves_per_tree") if "error" not in sz else None,
        "duration":           parse_duration(txt),
    }

# ============================
# PLOT HELPERS
# ============================

def depth_color_map(depths):
    colors = plt.cm.viridis(np.linspace(0, 1, max(len(depths), 1)))
    return {d: colors[i] for i, d in enumerate(sorted(depths))}

MODEL_COLORS = {
    "LicketyRESPLIT":    "#2196F3",
    "Threshold Guessing": "#FF9800",
    "XGBoost":            "#4CAF50",
}

# ============================
# MAIN LOOP OVER DATASETS
# ============================

# Accumulates one best-row per (model, dataset) for the cross-dataset summary
cross_dataset_rows = []

for dataset_dir in sorted(MODEL_RESULTS.iterdir()):
    if not dataset_dir.is_dir():
        continue

    dataset_name = dataset_dir.name
    out_dir = ANALYSIS_DIR / dataset_name
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_name}")
    print(f"{'='*60}")

    lr_rows  = []
    gsd_rows = []
    xgb_rows = []

    # ── Scan parameter-sweep subdirectories ───────────────────────
    for folder in sorted(dataset_dir.iterdir()):
        if not folder.is_dir():
            continue

        m = LR_PATTERN.match(folder.name)
        if m:
            row = collect_licketyresplit(folder, int(m.group(1)), float(m.group(2)), float(m.group(3)))
            if row:
                lr_rows.append(row)
            continue

        m = GSD_PATTERN.match(folder.name)
        if m:
            row = collect_gosdt(folder, int(m.group(1)), float(m.group(2)))
            if row:
                gsd_rows.append(row)
            continue

        m = XGB_PATTERN.match(folder.name)
        if m:
            row = collect_xgboost(folder, int(m.group(1)), int(m.group(2)))
            if row:
                xgb_rows.append(row)
            continue

    # ── Also read top-level result files (from run_all.py) ────────
    # Only add if that model has no sweep results (avoids duplicates)
    if not lr_rows:
        row = collect_licketyresplit(dataset_dir, depth=None, lambda_reg=None, rashomon=None)
        if row:
            lr_rows.append(row)

    if not gsd_rows:
        row = collect_gosdt(dataset_dir, depth=None, reg=None)
        if row:
            gsd_rows.append(row)

    if not xgb_rows:
        row = collect_xgboost(dataset_dir, max_depth=None, n_est=None)
        if row:
            xgb_rows.append(row)

    df_lr  = pd.DataFrame(lr_rows)
    df_gsd = pd.DataFrame(gsd_rows)
    df_xgb = pd.DataFrame(xgb_rows)

    # ── Print summary tables ──────────────────────────────────────
    for label, df in [("LicketyRESPLIT", df_lr),
                      ("Threshold Guessing", df_gsd),
                      ("XGBoost", df_xgb)]:
        if df.empty:
            print(f"\n{label}: no results found.")
        else:
            print(f"\n{label} ({len(df)} configs):")
            print(df.to_string(index=False))

    # ── Save raw CSVs ─────────────────────────────────────────────
    for fname, df in [("licketyresplit_sweep.csv", df_lr),
                      ("gosdt_sweep.csv",          df_gsd),
                      ("xgboost_sweep.csv",        df_xgb)]:
        if not df.empty:
            df.sort_values("accuracy", ascending=False).to_csv(out_dir / fname, index=False)
            print(f"Saved: {out_dir / fname}")

    # ============================
    # LICKETYRESPLIT PLOTS
    # ============================

    if not df_lr.empty:
        sweep_lr = df_lr.dropna(subset=["depth_budget", "lambda_reg", "rashomon_mult"])
        dcmap = depth_color_map(sweep_lr["depth_budget"].unique() if not sweep_lr.empty
                                else df_lr["depth_budget"].dropna().unique())

        # Accuracy vs Tree Size
        if not sweep_lr.empty and sweep_lr["n_leaves"].notna().any():
            plt.figure(figsize=(10, 6))
            for depth, sub in sweep_lr.groupby("depth_budget"):
                plt.scatter(sub["n_leaves"], sub["accuracy"],
                            c=[dcmap[depth]], s=100, alpha=0.7,
                            label=f"depth={depth}", edgecolors="black", linewidth=0.5)
            for _, row in sweep_lr.iterrows():
                plt.annotate(f"λ={row['lambda_reg']}\nρ={row['rashomon_mult']}",
                             (row["n_leaves"], row["accuracy"]),
                             fontsize=7, ha="center", alpha=0.7,
                             xytext=(0, -15), textcoords="offset points")
            plt.xlabel("Number of Leaves", fontsize=12)
            plt.ylabel("Test Accuracy", fontsize=12)
            plt.title(f"LicketyRESPLIT: Accuracy vs Complexity [{dataset_name}]",
                      fontsize=13, fontweight="bold")
            plt.legend(title="Depth Budget")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / "lr_accuracy_vs_complexity.png", dpi=300)
            plt.close()
            print(f"✓ Saved: {out_dir / 'lr_accuracy_vs_complexity.png'}")

        # Ensemble accuracy vs Tree Size
        if not sweep_lr.empty and sweep_lr["ensemble_accuracy"].notna().any():
            plt.figure(figsize=(10, 6))
            for depth, sub in sweep_lr.groupby("depth_budget"):
                plt.scatter(sub["n_leaves"], sub["ensemble_accuracy"],
                            c=[dcmap[depth]], s=100, alpha=0.7,
                            label=f"depth={depth}", edgecolors="black", linewidth=0.5)
            plt.xlabel("Number of Leaves", fontsize=12)
            plt.ylabel("Ensemble Accuracy", fontsize=12)
            plt.title(f"LicketyRESPLIT: Ensemble Accuracy vs Complexity [{dataset_name}]",
                      fontsize=13, fontweight="bold")
            plt.legend(title="Depth Budget")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / "lr_ensemble_vs_complexity.png", dpi=300)
            plt.close()
            print(f"✓ Saved: {out_dir / 'lr_ensemble_vs_complexity.png'}")

        # Accuracy & Rashomon-set heatmaps per depth
        for depth in sorted(sweep_lr["depth_budget"].dropna().unique()):
            sub = sweep_lr[sweep_lr["depth_budget"] == depth]
            for metric, cmap, label, suffix in [
                ("accuracy",       "viridis", "Accuracy",         "accuracy"),
                ("n_trees_in_set", "plasma",  "Rashomon Set Size", "rashomon_size"),
            ]:
                if sub[metric].isna().all():
                    continue
                try:
                    hmap = sub.pivot(index="lambda_reg", columns="rashomon_mult", values=metric)
                    fmt  = ".4f" if metric == "accuracy" else "g"
                    plt.figure(figsize=(8, 5))
                    sns.heatmap(hmap, annot=True, fmt=fmt, cmap=cmap,
                                cbar_kws={"label": label}, linewidths=0.5)
                    plt.title(f"LicketyRESPLIT {label} — depth={depth} [{dataset_name}]",
                              fontsize=13, fontweight="bold")
                    plt.xlabel("Rashomon Multiplier", fontsize=11)
                    plt.ylabel("Lambda (Regularization)", fontsize=11)
                    plt.tight_layout()
                    fname = f"lr_heatmap_depth{depth}_{suffix}.png"
                    plt.savefig(out_dir / fname, dpi=300)
                    plt.close()
                    print(f"✓ Saved: {out_dir / fname}")
                except Exception:
                    plt.close()

    # ============================
    # THRESHOLD GUESSING PLOTS
    # ============================

    if not df_gsd.empty:
        sweep_gsd = df_gsd.dropna(subset=["depth_budget", "regularization"])

        # Accuracy vs Tree Size
        if not sweep_gsd.empty and sweep_gsd["n_leaves"].notna().any():
            plt.figure(figsize=(9, 5))
            sc = plt.scatter(sweep_gsd["n_leaves"], sweep_gsd["accuracy"],
                             c=sweep_gsd["depth_budget"], cmap="viridis",
                             s=100, alpha=0.8, edgecolors="black", linewidth=0.5)
            plt.colorbar(sc, label="Depth Budget")
            for _, row in sweep_gsd.iterrows():
                plt.annotate(f"reg={row['regularization']}",
                             (row["n_leaves"], row["accuracy"]),
                             fontsize=7, ha="center",
                             xytext=(0, -13), textcoords="offset points")
            plt.xlabel("Number of Leaves", fontsize=12)
            plt.ylabel("Test Accuracy", fontsize=12)
            plt.title(f"Threshold Guessing: Accuracy vs Complexity [{dataset_name}]",
                      fontsize=13, fontweight="bold")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / "gosdt_accuracy_vs_complexity.png", dpi=300)
            plt.close()
            print(f"Saved: {out_dir / 'gosdt_accuracy_vs_complexity.png'}")

        # Accuracy heatmap (regularization vs depth)
        if not sweep_gsd.empty:
            try:
                hmap = sweep_gsd.pivot(index="regularization", columns="depth_budget", values="accuracy")
                plt.figure(figsize=(8, 5))
                sns.heatmap(hmap, annot=True, fmt=".4f", cmap="viridis",
                            cbar_kws={"label": "Accuracy"}, linewidths=0.5)
                plt.title(f"Threshold Guessing: Accuracy Heatmap [{dataset_name}]",
                          fontsize=13, fontweight="bold")
                plt.xlabel("Depth Budget", fontsize=11)
                plt.ylabel("Regularization", fontsize=11)
                plt.tight_layout()
                plt.savefig(out_dir / "gosdt_heatmap_accuracy.png", dpi=300)
                plt.close()
                print(f"✓ Saved: {out_dir / 'gosdt_heatmap_accuracy.png'}")
            except Exception:
                plt.close()

    # ============================
    # XGBOOST PLOTS
    # ============================

    if not df_xgb.empty:
        sweep_xgb = df_xgb.dropna(subset=["max_depth", "n_estimators"])

        # Accuracy vs Total Leaves
        if not sweep_xgb.empty and sweep_xgb["total_leaves"].notna().any():
            plt.figure(figsize=(9, 5))
            sc = plt.scatter(sweep_xgb["total_leaves"], sweep_xgb["accuracy"],
                             c=sweep_xgb["max_depth"], cmap="viridis",
                             s=100, alpha=0.8, edgecolors="black", linewidth=0.5)
            plt.colorbar(sc, label="max_depth")
            for _, row in sweep_xgb.iterrows():
                plt.annotate(f"n={int(row['n_estimators'])}",
                             (row["total_leaves"], row["accuracy"]),
                             fontsize=7, ha="center",
                             xytext=(0, -13), textcoords="offset points")
            plt.xlabel("Total Leaves (all trees)", fontsize=12)
            plt.ylabel("Test Accuracy", fontsize=12)
            plt.title(f"XGBoost: Accuracy vs Complexity [{dataset_name}]",
                      fontsize=13, fontweight="bold")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / "xgboost_accuracy_vs_complexity.png", dpi=300)
            plt.close()
            print(f"✓ Saved: {out_dir / 'xgboost_accuracy_vs_complexity.png'}")

        # Accuracy heatmap (max_depth vs n_estimators)
        if not sweep_xgb.empty:
            try:
                hmap = sweep_xgb.pivot(index="max_depth", columns="n_estimators", values="accuracy")
                plt.figure(figsize=(8, 5))
                sns.heatmap(hmap, annot=True, fmt=".4f", cmap="viridis",
                            cbar_kws={"label": "Accuracy"}, linewidths=0.5)
                plt.title(f"XGBoost: Accuracy Heatmap [{dataset_name}]",
                          fontsize=13, fontweight="bold")
                plt.xlabel("n_estimators", fontsize=11)
                plt.ylabel("max_depth", fontsize=11)
                plt.tight_layout()
                plt.savefig(out_dir / "xgboost_heatmap_accuracy.png", dpi=300)
                plt.close()
                print(f"✓ Saved: {out_dir / 'xgboost_heatmap_accuracy.png'}")
            except Exception:
                plt.close()

    # ============================
    # CROSS-MODEL COMPARISON
    # ============================

    best_rows = []
    for df, label, param_fn, leaves_col in [
        (df_lr,  "LicketyRESPLIT",    lambda r: f"d={r['depth_budget']}, λ={r['lambda_reg']}, ρ={r['rashomon_mult']}", "n_leaves"),
        (df_gsd, "Threshold Guessing", lambda r: f"d={r['depth_budget']}, reg={r['regularization']}",                   "n_leaves"),
        (df_xgb, "XGBoost",           lambda r: f"depth={r['max_depth']}, n={r['n_estimators']}",                       "total_leaves"),
    ]:
        if df.empty or df["accuracy"].isna().all():
            continue
        idx  = df["accuracy"].idxmax()
        best = df.loc[idx]
        try:
            params = param_fn(best)
        except Exception:
            params = "default"
        macro_f1 = best.get("macro_f1") if "macro_f1" in best.index else None
        n_leaves = best.get(leaves_col) if leaves_col in best.index else None
        best_rows.append({
            "model":         label,
            "best_accuracy": best["accuracy"],
            "params":        params,
        })
        cross_dataset_rows.append({
            "dataset":  dataset_name,
            "model":    label,
            "accuracy": best["accuracy"],
            "macro_f1": macro_f1,
            "n_leaves": n_leaves,
            "params":   params,
        })

    if best_rows:
        df_best = pd.DataFrame(best_rows)
        bar_colors = [MODEL_COLORS.get(m, "#999999") for m in df_best["model"]]

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(df_best["model"], df_best["best_accuracy"],
                      color=bar_colors, alpha=0.85,
                      edgecolor="black", linewidth=0.8)
        for bar, row in zip(bars, best_rows):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{row['best_accuracy']:.4f}\n({row['params']})",
                    ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("Best Test Accuracy", fontsize=12)
        ax.set_title(f"Best Accuracy per Model [{dataset_name}]",
                     fontsize=13, fontweight="bold")
        ax.set_ylim(0, min(1.08, df_best["best_accuracy"].max() + 0.08))
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "cross_model_best_accuracy.png", dpi=300)
        plt.close()
        print(f"Saved: {out_dir / 'cross_model_best_accuracy.png'}")

        df_best.to_csv(out_dir / "best_per_model.csv", index=False)
        print(f"Saved: {out_dir / 'best_per_model.csv'}")

# ============================
# CROSS-DATASET OVERVIEW
# ============================

if cross_dataset_rows:
    cross_out = ANALYSIS_DIR / "cross_dataset"
    os.makedirs(cross_out, exist_ok=True)

    df_cross = pd.DataFrame(cross_dataset_rows)
    df_cross.to_csv(cross_out / "all_best_results.csv", index=False)
    print(f"\n✓ Saved: {cross_out / 'all_best_results.csv'}")

    MODEL_ORDER = ["XGBoost", "Threshold Guessing", "LicketyRESPLIT"]

    for metric, cmap, label, fname in [
        ("accuracy", "viridis", "Best Test Accuracy",  "cross_dataset_accuracy_heatmap.png"),
        ("macro_f1", "magma",   "Best Macro F1 Score", "cross_dataset_f1_heatmap.png"),
    ]:
        pivot = df_cross.pivot(index="model", columns="dataset", values=metric)
        # Sort rows by model order, then by mean descending
        ordered = [m for m in MODEL_ORDER if m in pivot.index]
        remaining = [m for m in pivot.index if m not in ordered]
        pivot = pivot.loc[ordered + remaining]

        if pivot.isna().all().all():
            continue

        plt.figure(figsize=(max(6, len(pivot.columns) * 1.8), max(4, len(pivot) * 1.2)))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap,
                    cbar_kws={"label": label}, linewidths=0.5,
                    mask=pivot.isna())
        plt.title(f"{label} — All Models × All Datasets",
                  fontsize=13, fontweight="bold")
        plt.xlabel("Dataset", fontsize=11)
        plt.ylabel("Model", fontsize=11)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(cross_out / fname, dpi=300)
        plt.close()
        print(f"Saved: {cross_out / fname}")

    # Accuracy vs tree size (log scale) — all models and datasets on one plot
    plot_data = df_cross.dropna(subset=["n_leaves", "accuracy"])
    if not plot_data.empty:
        plt.figure(figsize=(10, 6))
        for model, sub in plot_data.groupby("model"):
            plt.scatter(sub["n_leaves"], sub["accuracy"],
                        color=MODEL_COLORS.get(model, "#999999"),
                        s=120, alpha=0.85, edgecolors="black",
                        linewidth=0.5, label=model, zorder=3)
            for _, row in sub.iterrows():
                plt.annotate(row["dataset"],
                             (row["n_leaves"], row["accuracy"]),
                             fontsize=7, alpha=0.8,
                             xytext=(4, 0), textcoords="offset points")
        plt.xscale("log")
        plt.xlabel("Number of Leaves (log scale)", fontsize=12)
        plt.ylabel("Best Test Accuracy", fontsize=12)
        plt.title("Accuracy vs Model Complexity — All Models & Datasets",
                  fontsize=13, fontweight="bold")
        plt.legend(title="Model")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(cross_out / "cross_dataset_accuracy_vs_complexity.png", dpi=300)
        plt.close()
        print(f"Saved: {cross_out / 'cross_dataset_accuracy_vs_complexity.png'}")

print("\n" + "=" * 60)
print("Analysis complete.")
print(f"All figures saved under: {ANALYSIS_DIR}")
print("=" * 60)
