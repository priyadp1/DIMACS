#!/usr/bin/env python3
"""
Run all models on a single dataset.
Edit only the DATASET CONFIGURATION section below to switch datasets.
Results are saved to: model_results/<dataset_name>/

Note: each model runs in its own subprocess to avoid C++ library conflicts
between gosdt, split, and resplit (pybind11 type registration clashes).
"""

import os
import sys
import json
import subprocess
from pathlib import Path

BASEDIR = Path(__file__).resolve().parent

# =============================================================================
# DATASET CONFIGURATION — only edit this section to change the dataset
# =============================================================================

DATASET_PATH  = BASEDIR / "datasets" / "Mine" / "breast_cancer_data.csv"
TARGET_COLUMN = 'diagnosis'
DROP_COLUMNS  = ['diagnosis']   # all columns to remove (must include target)
LABEL_MAP     = {"M": 1, "B": 0}  # set to None if labels are already numeric

# =============================================================================

dataset_name = DATASET_PATH.stem
results_dir  = BASEDIR / "model_results" / dataset_name
os.makedirs(results_dir, exist_ok=True)

config = {
    "dataset_path": str(DATASET_PATH),
    "target_column": TARGET_COLUMN,
    "drop_columns":  DROP_COLUMNS,
    "label_map":     LABEL_MAP,
    "results_dir":   str(results_dir),
}

config_file = BASEDIR / "_run_config.json"
with open(config_file, "w") as f:
    json.dump(config, f)

SCRIPTS = [
    ("XGBoost",        BASEDIR / "Code_files/run_xgboost.py"),
    ("SPLIT",          BASEDIR / "SPLIT-ICML/split/src/split/run_split.py"),
    ("GOSDT",          BASEDIR / "gosdt-guesses/examples/run_gosdt.py"),
    ("RESPLIT",        BASEDIR / "SPLIT-ICML/resplit/example/run_resplit.py"),
    ("LicketyRESPLIT", BASEDIR / "LicketyRESPLIT/examples/run_licketyRESPLIT.py"),
]

try:
    for name, script in SCRIPTS:
        print(f"\n{'='*60}\nRunning {name}...")
        result = subprocess.run([sys.executable, str(script)])
        if result.returncode != 0:
            print(f"[WARNING] {name} exited with code {result.returncode}")
        else:
            print(f"{name} done.")
finally:
    config_file.unlink(missing_ok=True)

print(f"\n{'='*60}")
print(f"All results saved to: {results_dir}")

# ── Tree Size Summary ──────────────────────────────────────────
size_files = [
    ("XGBoost",         results_dir / "xgboost_tree_size.json"),
    ("SPLIT",           results_dir / "split_tree_size.json"),
    ("GOSDT",           results_dir / "gosdt_tree_size.json"),
    ("RESPLIT",         results_dir / "resplit_tree_size.json"),
    ("TREEFARMS",       results_dir / "treefarms_tree_size.json"),
    ("LicketyRESPLIT",  results_dir / "licketyresplit_tree_size.json"),
]

print(f"\n{'='*60}")
print("Tree Size Summary (tree index 0 for Rashomon-set models)")
print(f"{'='*60}")
print(f"{'Model':<16} {'Leaves':>7} {'Nodes':>7}  Notes")
print("-" * 60)
for name, path in size_files:
    if not path.exists():
        print(f"{name:<16} {'N/A':>7} {'N/A':>7}")
        continue
    with open(path) as _f:
        _sz = json.load(_f)
    if "error" in _sz:
        print(f"{name:<16}  ERROR: {_sz['error']}")
    elif "n_trees" in _sz:          # XGBoost ensemble
        notes = f"({_sz['n_trees']} trees, {_sz['avg_leaves_per_tree']:.1f} leaves/tree avg)"
        print(f"{name:<16} {_sz['total_leaves']:>7} {_sz['total_nodes']:>7}  {notes}")
    elif "n_trees_in_set" in _sz:   # Rashomon-set model
        notes = f"(Rashomon set: {_sz['n_trees_in_set']} trees)"
        print(f"{name:<16} {_sz['n_leaves']:>7} {_sz['n_nodes']:>7}  {notes}")
    else:                           # single-tree model
        print(f"{name:<16} {_sz['n_leaves']:>7} {_sz['n_nodes']:>7}")
