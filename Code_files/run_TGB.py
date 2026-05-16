from gosdt import ThresholdGuessBinarizer
import pandas as pd
import os
import itertools
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent

BASEDIR = current
results_dir = BASEDIR / "TGB_Variables_Feature_Importance"
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
    "diabetes": {
        "path": BASEDIR / "datasets" / "Mine" / "diabetic_data.csv",
        "target_col": 'readmitted',
        "drop_cols": ['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty', 'max_glu_serum', 'A1Cresult', 'readmitted'],
        "label_map": {">30": 1, "<30": 1, "NO": 0},
    },
    "diabetes_smote": {
        "path": BASEDIR / "datasets" / "Mine" / "diabetes_smote.csv",
        "target_col": 'readmitted',
        "drop_cols": ['readmitted'],
        "label_map": None,
    },
    "creditcard_fraud_smote": {
        "path": BASEDIR / "datasets" / "Mine" / "creditcard_fraud_detection_smote.csv",
        "target_col": 'Class',
        "drop_cols": ['Class'],
        "label_map": None,
    },
}

# TGB Parameters to sweep
GBDT_N_EST = [40, 100, 200]       # number of trees for guessing thresholds
GBDT_MAX_DEPTH = [1, 2, 3]        # max depth for GBDT trees when guessing thresholds
REGULARIZATION = [0.001, 0.01, 0.1]  # regularization for downstream GOSDT (saved in metadata)

SIMILAR_SUPPORT = False
DEPTH_BUDGET = 6
TIME_LIMIT = 60
VERBOSE = True

all_results = []

for dataset_name, cfg in DATASETS.items():
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

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

    # Encode categorical columns (ThresholdGuessBinarizer requires numeric input)
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=False)
        X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=False)
        X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    for n_est, max_depth in itertools.product(GBDT_N_EST, GBDT_MAX_DEPTH):
        param_tag = f"nest{n_est}_depth{max_depth}"

        # Skip if this param combination already has results for this dataset
        out_dir = results_dir / dataset_name / param_tag
        if out_dir.exists():
            print(f"\n  --- Skipping {param_tag} (already exists) ---")
            continue

        print(f"\n  --- Params: n_estimators={n_est}, max_depth={max_depth} ---")

        # Step 1: Guess Thresholds
        enc = ThresholdGuessBinarizer(n_estimators=n_est, max_depth=max_depth, random_state=42)
        enc.set_output(transform="pandas")
        X_train_guessed = enc.fit_transform(X_train, y_train)
        X_test_guessed = enc.transform(X_test)
        print(f"  After guessing, X train shape:{X_train_guessed.shape}, X test shape:{X_test_guessed.shape}")
        print(f"  train/test columns match: {list(X_train_guessed.columns)==list(X_test_guessed.columns)}")

        # Step 2: Guess Lower Bounds
        gbdt = GradientBoostingClassifier(n_estimators=n_est, max_depth=max_depth, random_state=42)
        gbdt.fit(X_train_guessed, y_train)
        warm_labels = gbdt.predict(X_train_guessed)
        gbdt_acc = accuracy_score(y_test, gbdt.predict(X_test_guessed))
        print(f"  GBDT warm-label ensemble accuracy on test set: {gbdt_acc:.4f}")

        # Count how often each binary variable appears as a split node
        SUBSAMPLE_N = 10
        feature_names = list(X_train_guessed.columns)
        from collections import Counter
        split_counts = Counter()
        sampled_trees = gbdt.estimators_[:SUBSAMPLE_N]
        for tree_arr in sampled_trees:
            for estimator in tree_arr:
                tree = estimator.tree_
                for feat_idx in tree.feature:
                    if feat_idx >= 0:
                        split_counts[feature_names[feat_idx]] += 1

        # Save per-combination outputs
        os.makedirs(out_dir, exist_ok=True)

        with open(out_dir / "gbdt_warm_label_results.txt", "w") as f:
            f.write(f"n_estimators={n_est}, max_depth={max_depth}\n")
            f.write(f"GBDT warm-label ensemble accuracy on test set: {gbdt_acc:.4f}\n")

        binary_var_df = pd.DataFrame(
            split_counts.most_common(),
            columns=["binary_variable", "tree_count"]
        )
        binary_var_df["total_binary_variables"] = len(feature_names)
        importance_map = dict(zip(feature_names, gbdt.feature_importances_))
        binary_var_df["importance"] = binary_var_df["binary_variable"].map(importance_map)
        binary_var_df.to_csv(out_dir / "binary_variable_counts.csv", index=False)

        X_train_guessed.to_csv(out_dir / "X_train_guessed.csv", index=False)
        X_test_guessed.to_csv(out_dir / "X_test_guessed.csv", index=False)
        pd.Series(warm_labels, name="warm_labels").to_csv(out_dir / "warm_labels.csv", index=False)
        y_train.reset_index(drop=True).to_csv(out_dir / "y_train.csv", index=False)
        y_test.reset_index(drop=True).to_csv(out_dir / "y_test.csv", index=False)

        all_results.append({
            "dataset": dataset_name,
            "n_estimators": n_est,
            "max_depth": max_depth,
            "num_binary_features": X_train_guessed.shape[1],
            "gbdt_test_accuracy": gbdt_acc,
        })
        print(f"  Saved outputs to {out_dir}")

# Save summary across all datasets and parameter combinations
summary_df = pd.DataFrame(all_results)
summary_path = results_dir / "summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"\nSummary saved to {summary_path}")
print(summary_df.to_string(index=False))
