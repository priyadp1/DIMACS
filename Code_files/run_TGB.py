from gosdt import ThresholdGuessBinarizer
import pandas as pd
import os
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent

BASEDIR = current
results_dir = BASEDIR / "TGB_Variables"
os.makedirs(results_dir, exist_ok=True)

DATASETS = {
     "spambase": {
        "path": BASEDIR / "datasets" / "Mine" / "spambase.csv",
         "target_col": "class",
         "drop_cols": ["class"],
        "label_map": None,
     },
    "bike": {
         "path": BASEDIR / "datasets" / "Mine" / "bike.csv",
        "target_col": "cnt_binary",
       "drop_cols": ["instant", "cnt_binary"],
        "label_map": None,
    },
    "compas": {
        "path": BASEDIR / "datasets" / "Mine" / "compas.csv",
        "target_col": "two_year_recid",
        "drop_cols": ["two_year_recid"],
        "label_map": None,
    },
    "heloc": {
        "path": BASEDIR / "datasets" / "Mine" / "heloc_original.csv",
        "target_col": "RiskPerformance",
        "drop_cols": ["RiskPerformance"],
        "label_map": None,
    },
    "breast_cancer": {
        "path": BASEDIR / "datasets" / "Mine" / "breast_cancer_data.csv",
        "target_col": "diagnosis",
        "drop_cols": ["id", "diagnosis"],
        "label_map": {"M": 1, "B": 0},
    },
    "leukemia": {
        "path": BASEDIR / "datasets" / "Mine" / "leukemia_data.csv",
        "target_col": "label",
        "drop_cols": ["label"],
        "label_map": {"ALL": 0, "AML": 1},
    },
}

# Parameters
GBDT_N_EST = 40
GBDT_MAX_DEPTH = 1
REGULARIZATION = 0.001
SIMILAR_SUPPORT = False
DEPTH_BUDGET = 6
TIME_LIMIT = 60
VERBOSE = True

for dataset_name, cfg in DATASETS.items():
    print(f"\n{'='*50}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*50}")

    df = pd.read_csv(cfg["path"])
    df = df.dropna(axis=1, how="all")

    if cfg["label_map"]:
        df[cfg["target_col"]] = df[cfg["target_col"]].map(cfg["label_map"])

    X = df.drop(columns=cfg["drop_cols"])
    Y = df[cfg["target_col"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )
    print("X train shape:{}, X test shape:{}".format(X_train.shape, X_test.shape))

    # Step 1: Guess Thresholds
    enc = ThresholdGuessBinarizer(n_estimators=GBDT_N_EST, max_depth=GBDT_MAX_DEPTH, random_state=42)
    enc.set_output(transform="pandas")
    X_train_guessed = enc.fit_transform(X_train, y_train)
    X_test_guessed = enc.transform(X_test)
    print(f"After guessing, X train shape:{X_train_guessed.shape}, X test shape:{X_test_guessed.shape}")
    print(f"train set column names == test set column names: {list(X_train_guessed.columns)==list(X_test_guessed.columns)}")

    # Step 2: Guess Lower Bounds
    gbdt = GradientBoostingClassifier(n_estimators=GBDT_N_EST, max_depth=GBDT_MAX_DEPTH, random_state=42)
    gbdt.fit(X_train_guessed, y_train)
    warm_labels = gbdt.predict(X_train_guessed)

    # Save TGB outputs (per-dataset subdirectory)
    dataset_results_dir = results_dir / dataset_name
    os.makedirs(dataset_results_dir, exist_ok=True)
    X_train_guessed.to_csv(dataset_results_dir / "X_train_guessed.csv", index=False)
    X_test_guessed.to_csv(dataset_results_dir / "X_test_guessed.csv", index=False)
    pd.Series(warm_labels, name="warm_labels").to_csv(dataset_results_dir / "warm_labels.csv", index=False)
    y_train.reset_index(drop=True).to_csv(dataset_results_dir / "y_train.csv", index=False)
    y_test.reset_index(drop=True).to_csv(dataset_results_dir / "y_test.csv", index=False)
    print(f"Saved binarized data and warm labels to {dataset_results_dir}")
