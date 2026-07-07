import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_curve, auc

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent

BASEDIR = current
tgb_results_dir = BASEDIR / "TGB_Variables_Feature_Importance"
gbdt_results_dir = BASEDIR / "benchmarks_no_TGB_results_all"
tmp_splits_dir   = BASEDIR / "tmp_splits_no_tgb"

parameters = ["nest5_depth1", "nest5_depth2", "nest5_depth3", "nest10_depth1", "nest10_depth2", "nest10_depth3", "nest15_depth1", "nest15_depth2", "nest15_depth3", "nest20_depth1", "nest20_depth2", "nest20_depth3" , "nest25_depth1", "nest25_depth2", "nest25_depth3" , "nest30_depth1", "nest30_depth2", "nest30_depth3", "nest35_depth1", "nest35_depth2", "nest35_depth3", "nest40_depth1", "nest40_depth2", "nest40_depth3", "nest100_depth1", "nest100_depth2", "nest100_depth3", "nest200_depth1", "nest200_depth2", "nest200_depth3"]
datasets = ["bike" , "breast_cancer" , "creditcard_fraud" , "diabetes" , "heloc_original" , "creditcard_fraud_smote" , "diabetes_smote"  , "spambase"]

GBDT_FILES = {
    "XGBoost":  "xgboost_binarized_results.txt",
    #"LightGBM": "lightgbm_results.txt",
    #"CatBoost": "catboost_results.txt",
}


def parse_all_features(results_path):
    """Extract all feature names from a model results txt file."""
    text = results_path.read_text()
    idx = text.find("All Features:")
    if idx == -1:
        return []
    lines = text[idx:].splitlines()
    features = []
    for line in lines[2:5]:
        line = line.strip()
        if not line:
            break
        parts = line.rsplit(None, 1)
        if parts:
            features.append(parts[0].strip())
    return features


def _sanitize_xgb(col):
    return col.replace("[", "{").replace("]", "}").replace("<", "lt")


def _sanitize_lgb(col):
    return re.sub(r'[^A-Za-z0-9_]', '_', col)


def build_reverse_map(original_cols, model):
    """Map sanitized feature name → original column name."""
    if model == "CatBoost":
        return {col: col for col in original_cols}
    if model == "XGBoost":
        return {_sanitize_xgb(col): col for col in original_cols}
    seen, rev = {}, {}
    for col in original_cols:
        s = _sanitize_lgb(col)
        if s in seen:
            seen[s] += 1
            s = f"{s}_{seen[s]}"
        else:
            seen[s] = 0
        rev[s] = col
    return rev


def parse_param(param):
    """Extract (n_estimators, max_depth) from param string like 'nest5_depth1'."""
    m = re.match(r'nest(\d+)_depth(\d+)', param)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def parse_tgb_feature(binary_var):
    """Split 'feature <= threshold' into (feature, threshold). Returns (binary_var, '') if no match."""
    m = re.match(r'^(.+?)\s*<=\s*(.+)$', binary_var)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return binary_var, ""


def compute_auc_and_threshold(y, scores):
    """Return (auc, optimal_youden_threshold). Flips scores if AUC < 0.5."""
    fpr, tpr, thresholds = roc_curve(y, scores)
    roc_auc = auc(fpr, tpr)
    if roc_auc < 0.5:
        fpr, tpr, thresholds = roc_curve(y, 1 - scores)
        roc_auc = auc(fpr, tpr)
    optimal_thresh = thresholds[np.argmax(tpr - fpr)]
    return roc_auc, optimal_thresh


# Pre-load GBDT top-3 feature data per dataset (independent of TGB params)
print("Loading GBDT top-3 feature data...")
gbdt_data = {}   # dataset -> {split -> {model -> [(feat_original, values, y)]}}

for dataset in datasets:
    gbdt_dir = gbdt_results_dir / dataset
    tmp_dir  = tmp_splits_dir   / dataset
    if not gbdt_dir.exists() or not tmp_dir.exists():
        gbdt_data[dataset] = None
        continue

    by_split = {}
    for split in ("train", "test"):
        x_path = tmp_dir / f"X_{split}.csv"
        y_path = tmp_dir / f"y_{split}.csv"
        if not x_path.exists() or not y_path.exists():
            continue
        X = pd.read_csv(x_path)
        y = pd.read_csv(y_path).iloc[:, 0].values
        if len(np.unique(y)) != 2:
            continue

        by_model = {}
        for model, fname in GBDT_FILES.items():
            rpath = gbdt_dir / fname
            if not rpath.exists():
                continue
            entries = [(col, X[col].values, y) for col in X.columns]
            by_model[model] = entries
        by_split[split] = by_model

    gbdt_data[dataset] = by_split
    print(f"  {dataset}: loaded")


out_dir = BASEDIR / "roc_curve_csvs"
out_dir.mkdir(exist_ok=True)

splits_out_dir = BASEDIR / "split_counts_csvs"
splits_out_dir.mkdir(exist_ok=True)

COLS = ["n_estimators", "max_depth", "Model", "Feature", "Threshold (≤)",
        "Train AUC", "Test AUC", "Train-Test gap"]

SPLIT_COLS = ["parameter", "n_estimators", "max_depth", "num_splits_used"]

for dataset in datasets:
    rows = []
    split_rows = []

    # TGB rows — one per (param, feature)
    for param in parameters:
        n_est, max_d = parse_param(param)
        tgb_base = tgb_results_dir / dataset / param
        fi_path  = tgb_base / "binary_variable_counts.csv"
        if not fi_path.exists():
            continue

        fi_df = pd.read_csv(fi_path).sort_values("importance", ascending=False)
        all_feats = fi_df["binary_variable"].tolist()

        split_rows.append({
            "parameter": param,
            "n_estimators": n_est,
            "max_depth": max_d,
            "num_splits_used": len(fi_df),
        })

        tgb_splits = {}
        for split in ("train", "test"):
            x_path = tgb_base / f"X_{split}_guessed.csv"
            y_path = tgb_base / f"y_{split}.csv"
            if not x_path.exists() or not y_path.exists():
                continue
            X = pd.read_csv(x_path)
            y = pd.read_csv(y_path).iloc[:, 0].values
            if len(np.unique(y)) != 2:
                continue
            tgb_splits[split] = (X, y)

        if "train" not in tgb_splits:
            continue

        X_tr, y_tr = tgb_splits["train"]
        for feat in all_feats:
            if feat not in X_tr.columns:
                continue

            train_auc, _ = compute_auc_and_threshold(y_tr, X_tr[feat].values)

            test_auc = None
            if "test" in tgb_splits:
                X_te, y_te = tgb_splits["test"]
                if feat in X_te.columns:
                    test_auc, _ = compute_auc_and_threshold(y_te, X_te[feat].values)

            orig_feat, threshold = parse_tgb_feature(feat)
            rows.append({
                "n_estimators": n_est,
                "max_depth": max_d,
                "Model": "TGB",
                "Feature": orig_feat,
                "Threshold (≤)": threshold,
                "Train AUC": round(train_auc, 4),
                "Test AUC": round(test_auc, 4) if test_auc is not None else "",
                "Train-Test gap": round(train_auc - test_auc, 4) if test_auc is not None else "",
            })

    # GBDT/XGBoost rows — once per dataset, not repeated per param
    gbdt_split_data = gbdt_data.get(dataset) or {}
    train_by_model = gbdt_split_data.get("train", {})
    test_by_model  = gbdt_split_data.get("test",  {})

    for model in train_by_model:
        train_entries = {feat: (vals, y) for feat, vals, y in train_by_model[model]}
        test_entries  = {feat: (vals, y) for feat, vals, y in test_by_model.get(model, [])}

        for feat, (vals_tr, y_tr) in train_entries.items():
            train_auc, opt_thresh = compute_auc_and_threshold(y_tr, vals_tr)

            test_auc = None
            if feat in test_entries:
                vals_te, y_te = test_entries[feat]
                test_auc, _ = compute_auc_and_threshold(y_te, vals_te)

            rows.append({
                "n_estimators": "",
                "max_depth": "",
                "Model": model,
                "Feature": feat,
                "Threshold (≤)": round(opt_thresh, 4),
                "Train AUC": round(train_auc, 4),
                "Test AUC": round(test_auc, 4) if test_auc is not None else "",
                "Train-Test gap": round(train_auc - test_auc, 4) if test_auc is not None else "",
            })

    if split_rows:
        splits_df = pd.DataFrame(split_rows, columns=SPLIT_COLS)
        splits_path = splits_out_dir / f"{dataset}_split_counts.csv"
        splits_df.to_csv(splits_path, index=False)
        print(f"Saved: {splits_path.relative_to(BASEDIR)}")

    if not rows:
        continue

    df = (pd.DataFrame(rows, columns=COLS)
            .sort_values("Test AUC", ascending=False, na_position="last")
            .reset_index(drop=True))
    out_path = out_dir / f"{dataset}_roc_auc.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path.relative_to(BASEDIR)}")
