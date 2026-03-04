from sklearn.model_selection import StratifiedKFold
import pandas as pd
from pathlib import Path
import subprocess
import json
import os
import sys
BASEDIR = Path(__file__).resolve().parent.parent
CODEDIR = Path(__file__).resolve().parent

dataset_name = "leukemia_data"
dataset_path = BASEDIR / "datasets" / "Mine" / "leukemia_data.csv"

print(f"\nLoading dataset from: {dataset_path}")
df = pd.read_csv(dataset_path)

if "label" not in df.columns:
    raise ValueError("Dataset must contain a 'label' column.")

X = df.drop(columns=["label"])
y = df["label"]

print("\nClass distribution:")
print(y.value_counts())

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_index, test_index) in enumerate(skf.split(X, y), start=1):

    print(f"\n{'='*60}")
    print(f"Running Fold {fold}")
    print(f"{'='*60}")

    # Split data
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Verify stratification (optional but recommended)
    print("Train class counts:\n", y_train.value_counts())
    print("Test class counts:\n", y_test.value_counts())


    fold_dir = BASEDIR / "model_results" / dataset_name / f"fold_{fold}"
    os.makedirs(fold_dir, exist_ok=True)

    # Save train/test CSVs
    train_df = pd.concat(
        [X_train.reset_index(drop=True), y_train.reset_index(drop=True)],
        axis=1
    )

    test_df = pd.concat(
        [X_test.reset_index(drop=True), y_test.reset_index(drop=True)],
        axis=1
    )

    train_path = fold_dir / "train.csv"
    test_path  = fold_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    config = {
        "train_path": str(train_path),
        "test_path":  str(test_path),
        "results_dir": str(fold_dir),
    }

    config_path = fold_dir / "config.json"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    subprocess.run(
        [sys.executable, str(CODEDIR / "run_all.py"), str(config_path)],
        check=True
    )

print(f"\n{'='*60}")
print("Cross-validation complete.")
print(f"Results saved in: {BASEDIR / 'model_results' / dataset_name}")
print(f"{'='*60}")