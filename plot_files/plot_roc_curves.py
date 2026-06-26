import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent

BASEDIR = current
tgb_results_dir = BASEDIR / "TGB_Variables_Feature_Importance"
gbdt_results_dir = BASEDIR / "benchmarks_no_TGB_results_all"
tmp_splits_dir   = BASEDIR / "tmp_splits_no_tgb"

parameters = [
    "nest40_depth1", "nest40_depth2", "nest40_depth3",
    "nest100_depth1", "nest100_depth2", "nest100_depth3",
    "nest200_depth1", "nest200_depth2", "nest200_depth3",
]
datasets = ["bike", "compas", "breast_cancer", "spambase",
            "creditcard_fraud", "creditcard_fraud_smote",
            "diabetes", "diabetes_smote"]

TGB_COLORS  = ["tab:blue", "steelblue", "cornflowerblue"]
GBDT_COLORS = {
    "XGBoost":  ["tab:orange",  "darkorange",  "moccasin"],
    #"LightGBM": ["tab:green",   "seagreen",    "palegreen"],
    #"CatBoost": ["tab:red",     "firebrick",   "lightsalmon"],
}
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
    # lines[0] = "Top 3 Features:", lines[1] = header, lines[2:5] = data rows
    features = []
    for line in lines[2:5]:
        line = line.strip()
        if not line:
            break
        # Feature name is all but the last whitespace-delimited token (Importance)
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

    # LightGBM — sanitize then deduplicate (must mirror run_no_TGB.py exactly)
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


# Pre-load GBDT top-3 feature data per dataset (independent of TGB params)
print("Loading GBDT top-3 feature data...")
gbdt_data = {}   # dataset -> {split -> {model -> [(feat_original, values)]}}

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
            sanitized_feats = parse_all_features(rpath)
            rev_map = build_reverse_map(X.columns, model)
            entries = []
            for sf in sanitized_feats:
                orig = rev_map.get(sf)
                if orig and orig in X.columns:
                    entries.append((orig, X[orig].values, y))
            by_model[model] = entries
        by_split[split] = by_model

    gbdt_data[dataset] = by_split
    print(f"  {dataset}: loaded")


for dataset in datasets:
    for param in parameters:
        tgb_base = tgb_results_dir / dataset / param
        fi_path  = tgb_base / "binary_variable_counts.csv"
        if not fi_path.exists():
            continue

        fi_df = pd.read_csv(fi_path).sort_values("importance", ascending=False)
        top3  = fi_df["binary_variable"].head(3).tolist()

        splits = {}
        for split in ("train", "test"):
            x_path = tgb_base / f"X_{split}_guessed.csv"
            y_path = tgb_base / f"y_{split}.csv"
            if not x_path.exists() or not y_path.exists():
                continue
            X = pd.read_csv(x_path)
            y = pd.read_csv(y_path).iloc[:, 0].values
            if len(np.unique(y)) != 2:
                continue
            splits[split] = (X, y)

        if not splits:
            continue

        fig, axes = plt.subplots(1, len(splits), figsize=(7 * len(splits), 6), squeeze=False)

        for ax, (split, (X_tgb, y_tgb)) in zip(axes[0], splits.items()):
            all_curves = []  # (roc_auc, label, fpr, tpr, color, linestyle)

            # TGB binary variable curves (dashed, blues)
            tgb_raw = []
            for feat in top3:
                if feat not in X_tgb.columns:
                    continue
                fpr, tpr, _ = roc_curve(y_tgb, X_tgb[feat].values)
                roc_auc = auc(fpr, tpr)
                if roc_auc < 0.5:
                    fpr, tpr, _ = roc_curve(y_tgb, 1 - X_tgb[feat].values)
                    roc_auc = auc(fpr, tpr)
                tgb_raw.append((roc_auc, feat, fpr, tpr))
            tgb_raw.sort(key=lambda x: x[0], reverse=True)
            for (roc_auc, feat, fpr, tpr), color in zip(tgb_raw, TGB_COLORS):
                all_curves.append((roc_auc, f"TGB: {feat}", fpr, tpr, color, "--"))

            # GBDT model feature curves (solid, per-model colors)
            split_data = (gbdt_data.get(dataset) or {}).get(split, {})
            for model, entries in split_data.items():
                colors = GBDT_COLORS[model]
                for (feat, values, y_gbdt), color in zip(entries, colors):
                    fpr, tpr, _ = roc_curve(y_gbdt, values)
                    roc_auc = auc(fpr, tpr)
                    if roc_auc < 0.5:
                        fpr, tpr, _ = roc_curve(y_gbdt, 1 - values)
                        roc_auc = auc(fpr, tpr)
                    all_curves.append((roc_auc, f"{model}: {feat}", fpr, tpr, color, "-"))

            # Sort all curves by AUC descending
            all_curves.sort(key=lambda x: x[0], reverse=True)

            for roc_auc, label, fpr, tpr, color, ls in all_curves:
                ax.plot(fpr, tpr, color=color, lw=2, linestyle=ls,
                        label=f"{label} (AUC={roc_auc:.3f})")

            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.02])
            ax.set_xlabel("False Positive Rate", fontsize=12)
            ax.set_ylabel("True Positive Rate", fontsize=12)
            ax.set_title(f"{dataset} | {param} | {split}", fontsize=13)
            ax.legend(loc="lower right", fontsize=9)

        plt.tight_layout()
        out_path = tgb_base / f"{dataset}_{param}_roc_curves.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path.relative_to(BASEDIR)}")
