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

DATASET_PATH  = BASEDIR / "datasets" / "Given" / "bike_binarized.csv"
TARGET_COLUMN = 'label'
DROP_COLUMNS  = ['label']   # all columns to remove (must include target)
#LABEL_MAP     = {"3711": 1, "3196": 0}      # set to None if labels are already numeric

# =============================================================================

dataset_name = DATASET_PATH.stem
results_dir  = BASEDIR / "model_results" / dataset_name
os.makedirs(results_dir, exist_ok=True)

config = {
    "dataset_path": str(DATASET_PATH),
    "target_column": TARGET_COLUMN,
    "drop_columns":  DROP_COLUMNS,
    #"label_map":     LABEL_MAP,
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
