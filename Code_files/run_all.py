#!/usr/bin/env python3
"""
Run all models on every dataset in datasets/Mine/.
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
# DATASETS — add / remove entries here to control which datasets are run
# =============================================================================

DATASETS = [
    {
        "path":          BASEDIR / "datasets/Mine/bike.csv",
        "target_column": "cnt_binary",
        "drop_columns":  ["cnt_binary"],
        "label_map":     None,
    },
    {
        "path":          BASEDIR / "datasets/Mine/breast_cancer_data.csv",
        "target_column": "diagnosis",
        "drop_columns":  ["id", "diagnosis"],
        "label_map":     {"M": 1, "B": 0},
    },
    {
        "path":          BASEDIR / "datasets/Mine/heloc_original.csv",
        "target_column": "RiskPerformance",
        "drop_columns":  ["RiskPerformance"],
        "label_map":     None,
    },
    {
        "path":          BASEDIR / "datasets/Mine/spambase.csv",
        "target_column": "class",
        "drop_columns":  ["class"],
        "label_map":     None,
    },
]

# =============================================================================

SCRIPTS = [
    ("XGBoost",            BASEDIR / "Code_files/run_xgboost.py"),
    ("Threshold Guessing", BASEDIR / "gosdt-guesses/examples/run_gosdt.py"),
    ("LicketyRESPLIT",     BASEDIR / "LicketyRESPLIT/examples/run_licketyRESPLIT.py"),
    # ("TREEFARMS",          BASEDIR / "Code_files/run_treefarms.py"),
]

config_file = BASEDIR / "_run_config.json"

for dataset in DATASETS:
    dataset_name = dataset["path"].stem
    results_dir  = BASEDIR / "model_results" / dataset_name
    os.makedirs(results_dir, exist_ok=True)

    config = {
        "dataset_path":  str(dataset["path"]),
        "target_column": dataset["target_column"],
        "drop_columns":  dataset["drop_columns"],
        "label_map":     dataset["label_map"],
        "results_dir":   str(results_dir),
    }

    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_name}")
    print(f"{'='*60}")

    try:
        with open(config_file, "w") as f:
            json.dump(config, f)

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
        ("XGBoost",            results_dir / "xgboost_tree_size.json"),
        ("Threshold Guessing", results_dir / "gosdt_tree_size.json"),
        ("LicketyRESPLIT",     results_dir / "licketyresplit_tree_size.json"),
        # ("TREEFARMS",          results_dir / "treefarms_tree_size.json"),
    ]

    print(f"\n{'='*60}")
    print(f"Tree Size Summary — {dataset_name}")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Leaves':>7} {'Nodes':>7}  Notes")
    print("-" * 60)
    for name, path in size_files:
        if not path.exists():
            print(f"{name:<20} {'N/A':>7} {'N/A':>7}")
            continue
        with open(path) as _f:
            _sz = json.load(_f)
        if "error" in _sz:
            print(f"{name:<20}  ERROR: {_sz['error']}")
        elif "n_trees" in _sz:           # XGBoost ensemble
            notes = f"({_sz['n_trees']} trees, {_sz['avg_leaves_per_tree']:.1f} leaves/tree avg)"
            print(f"{name:<20} {_sz['total_leaves']:>7} {_sz['total_nodes']:>7}  {notes}")
        elif "n_trees_in_set" in _sz:    # Rashomon-set model
            notes = f"(Rashomon set: {_sz['n_trees_in_set']} trees)"
            print(f"{name:<20} {_sz['n_leaves']:>7} {_sz['n_nodes']:>7}  {notes}")
        else:                            # single-tree model
            print(f"{name:<20} {_sz['n_leaves']:>7} {_sz['n_nodes']:>7}")
