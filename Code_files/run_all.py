import os
import sys
import json
import subprocess
from pathlib import Path

BASEDIR = Path(__file__).resolve().parent.parent  # project root (DIMACS/)
CODEDIR = Path(__file__).resolve().parent         # Code_files/

# =============================================================================
# MODEL SCRIPTS
# =============================================================================

SCRIPTS = [
    ("XGBoost",            CODEDIR / "run_xgboost.py"),
    ("Threshold Guessing", CODEDIR / "run_gosdt.py"),
    ("LicketyRESPLIT",     CODEDIR / "run_licketyRESPLIT.py"),
]

CONFIG_FILE = CODEDIR / "_run_config.json"


# =============================================================================
# CV MODE (if config path is passed as argument)
# =============================================================================

if len(sys.argv) > 1:
    config_path = Path(sys.argv[1])
    with open(config_path) as f:
        config = json.load(f)

    results_dir = Path(config["results_dir"])
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("CROSS-VALIDATION MODE")
    print(f"{'='*60}")
    print(f"Train: {config['train_path']}")
    print(f"Test:  {config['test_path']}")
    print(f"Results: {results_dir}")

    try:
        # Write temp config file that model scripts expect
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f)

        for name, script in SCRIPTS:
            print(f"\n{'='*60}\nRunning {name}...")
            result = subprocess.run([sys.executable, str(script)])
            if result.returncode != 0:
                print(f"[WARNING] {name} exited with code {result.returncode}")
            else:
                print(f"{name} done.")
    finally:
        CONFIG_FILE.unlink(missing_ok=True)

    print(f"\nAll fold results saved to: {results_dir}")
    sys.exit(0)


# =============================================================================
# NORMAL DATASET MODE
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
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f)

        for name, script in SCRIPTS:
            print(f"\n{'='*60}\nRunning {name}...")
            result = subprocess.run([sys.executable, str(script)])
            if result.returncode != 0:
                print(f"[WARNING] {name} exited with code {result.returncode}")
            else:
                print(f"{name} done.")
    finally:
        CONFIG_FILE.unlink(missing_ok=True)

    print(f"\nAll results saved to: {results_dir}")