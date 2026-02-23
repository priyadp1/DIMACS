#!/usr/bin/env python3
"""
Analyze LicketyRESPLIT parameter sweep results.
Reads all {depth}_{lambda}_{rashomon}/ folders and generates comparison plots.
"""

import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================
# CONFIGURATION
# ============================

BASEDIR = Path(__file__).resolve().parent
DATASET_NAME = "breast_cancer_data"  # Change this to analyze different datasets
results_base = BASEDIR / "model_results" / DATASET_NAME

OUTPUT_DIR = BASEDIR / "analysis_figures" / DATASET_NAME
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================
# PARSE ALL PARAMETER FOLDERS
# ============================

results = []

# Pattern: {depth}_{lambda}_{rashomon}/
for folder in results_base.iterdir():
    if not folder.is_dir():
        continue

    # Try to parse folder name as parameter combination
    match = re.match(r'^(\d+)_([\d.]+)_([\d.]+)$', folder.name)
    if not match:
        continue  # Skip folders that don't match the pattern

    depth, lambda_reg, rashomon = match.groups()
    depth = int(depth)
    lambda_reg = float(lambda_reg)
    rashomon = float(rashomon)

    # Read results.txt
    results_file = folder / "licketyresplit_results.txt"
    if not results_file.exists():
        continue

    with open(results_file) as f:
        content = f.read()

    # Extract accuracy
    acc_match = re.search(r"Accuracy:\s*([0-9.]+)", content)
    accuracy = float(acc_match.group(1)) if acc_match else None

    # Extract ensemble accuracy
    ens_match = re.search(r"Ensemble Accuracy:\s*([0-9.]+)", content)
    ensemble_acc = float(ens_match.group(1)) if ens_match else None

    # Extract training time
    time_match = re.search(r"completed in\s+([\d.]+)\s+seconds with\s+(\d+)\s+trees", content)
    if time_match:
        duration = float(time_match.group(1))
        n_trees = int(time_match.group(2))
    else:
        duration = None
        n_trees = None

    # Read tree_size.json
    size_file = folder / "licketyresplit_tree_size.json"
    n_leaves = None
    n_nodes = None
    if size_file.exists():
        with open(size_file) as f:
            sz = json.load(f)
        if "error" not in sz:
            n_leaves = sz.get("n_leaves")
            n_nodes = sz.get("n_nodes")
            if n_trees is None:
                n_trees = sz.get("n_trees_in_set")

    results.append({
        "depth_budget": depth,
        "lambda_reg": lambda_reg,
        "rashomon_mult": rashomon,
        "accuracy": accuracy,
        "ensemble_accuracy": ensemble_acc,
        "n_leaves": n_leaves,
        "n_nodes": n_nodes,
        "n_trees_in_set": n_trees,
        "duration": duration,
    })

df = pd.DataFrame(results)

if df.empty:
    raise ValueError(f"No LicketyRESPLIT results found in {results_base}")

print(f"Found {len(df)} parameter combinations\n")
print(df.to_string(index=False))
print("\n")

# ============================
# SUMMARY STATISTICS
# ============================

print("=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)
print(f"Best single-tree accuracy: {df['accuracy'].max():.4f}")
best_single = df.loc[df['accuracy'].idxmax()]
print(f"  → depth={best_single['depth_budget']}, lambda={best_single['lambda_reg']}, rashomon={best_single['rashomon_mult']}")

print(f"\nBest ensemble accuracy: {df['ensemble_accuracy'].max():.4f}")
best_ensemble = df.loc[df['ensemble_accuracy'].idxmax()]
print(f"  → depth={best_ensemble['depth_budget']}, lambda={best_ensemble['lambda_reg']}, rashomon={best_ensemble['rashomon_mult']}")

print(f"\nSmallest tree (leaves): {df['n_leaves'].min()}")
smallest = df.loc[df['n_leaves'].idxmin()]
print(f"  → depth={smallest['depth_budget']}, lambda={smallest['lambda_reg']}, rashomon={smallest['rashomon_mult']}, accuracy={smallest['accuracy']:.4f}")

print(f"\nLargest Rashomon set: {df['n_trees_in_set'].max()}")
largest_rset = df.loc[df['n_trees_in_set'].idxmax()]
print(f"  → depth={largest_rset['depth_budget']}, lambda={largest_rset['lambda_reg']}, rashomon={largest_rset['rashomon_mult']}")

print("\n")

# ============================
# PLOT 1: Accuracy vs Tree Size (Pareto Frontier)
# ============================

plt.figure(figsize=(10, 6))

# Color by depth_budget
unique_depths = sorted(df['depth_budget'].unique())
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_depths)))
depth_colors = {d: colors[i] for i, d in enumerate(unique_depths)}

for depth in unique_depths:
    subset = df[df['depth_budget'] == depth]
    plt.scatter(subset['n_leaves'], subset['accuracy'],
                c=[depth_colors[depth]], s=100, alpha=0.7,
                label=f'depth={depth}', edgecolors='black', linewidth=0.5)

# Annotate with parameter labels
for _, row in df.iterrows():
    label = f"λ={row['lambda_reg']}\nρ={row['rashomon_mult']}"
    plt.annotate(label, (row['n_leaves'], row['accuracy']),
                 fontsize=7, ha='center', alpha=0.7,
                 xytext=(0, -15), textcoords='offset points')

plt.xlabel('Number of Leaves (Tree Complexity)', fontsize=12)
plt.ylabel('Single-Tree Test Accuracy', fontsize=12)
plt.title('Accuracy vs Model Complexity (LicketyRESPLIT)', fontsize=14, fontweight='bold')
plt.legend(title='Depth Budget')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "accuracy_vs_complexity.png", dpi=300)
plt.close()

print(f"✓ Saved: {OUTPUT_DIR / 'accuracy_vs_complexity.png'}")

# ============================
# PLOT 2: Ensemble Accuracy vs Tree Size
# ============================

plt.figure(figsize=(10, 6))

for depth in unique_depths:
    subset = df[df['depth_budget'] == depth]
    plt.scatter(subset['n_leaves'], subset['ensemble_accuracy'],
                c=[depth_colors[depth]], s=100, alpha=0.7,
                label=f'depth={depth}', edgecolors='black', linewidth=0.5)

plt.xlabel('Number of Leaves (Tree Complexity)', fontsize=12)
plt.ylabel('Ensemble Test Accuracy', fontsize=12)
plt.title('Ensemble Accuracy vs Model Complexity', fontsize=14, fontweight='bold')
plt.legend(title='Depth Budget')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ensemble_accuracy_vs_complexity.png", dpi=300)
plt.close()

print(f"✓ Saved: {OUTPUT_DIR / 'ensemble_accuracy_vs_complexity.png'}")

# ============================
# PLOT 3: Accuracy Heatmap (Lambda vs Rashomon) for each Depth
# ============================

for depth in unique_depths:
    subset = df[df['depth_budget'] == depth]

    # Pivot for heatmap
    heatmap_data = subset.pivot(index='lambda_reg', columns='rashomon_mult', values='accuracy')

    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='viridis',
                cbar_kws={'label': 'Accuracy'}, linewidths=0.5)
    plt.title(f'Accuracy Heatmap (Depth Budget = {depth})', fontsize=14, fontweight='bold')
    plt.xlabel('Rashomon Multiplier', fontsize=12)
    plt.ylabel('Lambda (Regularization)', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"heatmap_depth{depth}_accuracy.png", dpi=300)
    plt.close()

    print(f"✓ Saved: {OUTPUT_DIR / f'heatmap_depth{depth}_accuracy.png'}")

# ============================
# PLOT 4: Rashomon Set Size Heatmap
# ============================

for depth in unique_depths:
    subset = df[df['depth_budget'] == depth]

    heatmap_data = subset.pivot(index='lambda_reg', columns='rashomon_mult', values='n_trees_in_set')

    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='g', cmap='plasma',
                cbar_kws={'label': 'Number of Trees'}, linewidths=0.5)
    plt.title(f'Rashomon Set Size (Depth Budget = {depth})', fontsize=14, fontweight='bold')
    plt.xlabel('Rashomon Multiplier', fontsize=12)
    plt.ylabel('Lambda (Regularization)', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"heatmap_depth{depth}_rashomon_size.png", dpi=300)
    plt.close()

    print(f"✓ Saved: {OUTPUT_DIR / f'heatmap_depth{depth}_rashomon_size.png'}")

# ============================
# PLOT 5: Bar Chart Comparison
# ============================

# Sort by accuracy
df_sorted = df.sort_values('accuracy', ascending=False)

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Top plot: Single-tree accuracy
ax1 = axes[0]
x_labels = [f"d={row['depth_budget']}, λ={row['lambda_reg']}, ρ={row['rashomon_mult']}"
            for _, row in df_sorted.iterrows()]
colors_bar = [depth_colors[row['depth_budget']] for _, row in df_sorted.iterrows()]
ax1.bar(range(len(df_sorted)), df_sorted['accuracy'], color=colors_bar, alpha=0.7, edgecolor='black')
ax1.set_xticks(range(len(df_sorted)))
ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
ax1.set_ylabel('Single-Tree Accuracy', fontsize=11)
ax1.set_title('Parameter Comparison (sorted by accuracy)', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Bottom plot: Tree complexity (leaves)
ax2 = axes[1]
ax2.bar(range(len(df_sorted)), df_sorted['n_leaves'], color=colors_bar, alpha=0.7, edgecolor='black')
ax2.set_xticks(range(len(df_sorted)))
ax2.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
ax2.set_ylabel('Number of Leaves', fontsize=11)
ax2.set_title('Model Complexity', fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "parameter_comparison_bars.png", dpi=300)
plt.close()

print(f"✓ Saved: {OUTPUT_DIR / 'parameter_comparison_bars.png'}")

# ============================
# SAVE SUMMARY TABLE
# ============================

df_sorted.to_csv(OUTPUT_DIR / "parameter_sweep_results.csv", index=False)
print(f"✓ Saved: {OUTPUT_DIR / 'parameter_sweep_results.csv'}")

print("\n" + "=" * 60)
print(f"All analysis figures saved to: {OUTPUT_DIR}")
print("=" * 60)
