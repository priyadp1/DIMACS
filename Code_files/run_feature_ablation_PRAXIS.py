import sys
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from praxis import PRAXIS

TOP_K_ABLATE = 1          # how many top raw features to drop from the root
N_BOOTSTRAP = 1000
CI = 0.95
SUBSAMPLE_N = 10           # trees sampled when counting TGB split features (matches run_TGB.py)
SEED = 42

# Matches run_PRAXIS.py so the "first tree" (tree_index=0) is comparable across scripts.
PRAXIS_TREE_IDX   = 0
LAMBDA_REG        = 0.01
DEPTH_BUDGET      = 5
RASHOMON_MULT     = 0.03
LOOKAHEAD_K       = 1

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent
BASEDIR = current

TGB_DIR = BASEDIR / "TGB_Variables_Feature_Importance"
OUT_DIR = BASEDIR / "feature_ablation_results_praxis"
OUT_DIR.mkdir(exist_ok=True)


def raw_feature_of(col_name):
    return col_name.split(" <= ")[0]


def bootstrap_accuracy_ci(y_true, y_pred, n_boot=N_BOOTSTRAP, ci=CI, seed=SEED):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    accs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        accs[i] = accuracy_score(y_true[idx], y_pred[idx])
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(accs, [alpha, 1 - alpha])
    return float(accs.mean()), float(lo), float(hi)


def jaccard(a, b):
    union = a | b
    if not union:
        return float("nan")
    return len(a & b) / len(union)


def fit_tgb_gbdt(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=2, seed=SEED):
    print(f"    [TGB] Fitting GBDT on {X_train.shape[1]} binarized columns ({len(X_train)} rows)...")
    gbdt = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed)
    gbdt.fit(X_train, y_train)
    test_pred = gbdt.predict(X_test)
    acc = accuracy_score(y_test, test_pred)
    print(f"    [TGB] Test accuracy: {acc:.4f}")

    print(f"    [TGB] Counting split features across {min(SUBSAMPLE_N, len(gbdt.estimators_))} sampled trees...")
    feature_names = list(X_train.columns)
    split_counts = Counter()
    for tree_arr in gbdt.estimators_[:SUBSAMPLE_N]:
        tree = tree_arr[0].tree_
        for feat_idx in tree.feature:
            if feat_idx >= 0:
                split_counts[feature_names[feat_idx]] += 1
    split_features = {raw_feature_of(c) for c in split_counts}
    print(f"    [TGB] Raw split features: {sorted(split_features)}")

    return gbdt, acc, test_pred, split_features


def _split_features_from_paths(paths_str, feature_names):
    """Parse PRAXIS's signed-index path strings (e.g. "[+0, -1]") into the
    set of raw features split on, matching PRAXIS.get_tree_paths_str's
    0-indexed, already-shifted feature indices."""
    split_idx = set()
    for path in paths_str:
        body = path.strip("[]")
        if not body:
            continue
        for tok in body.split(","):
            tok = tok.strip()
            if tok:
                split_idx.add(abs(int(tok)))
    return {raw_feature_of(feature_names[i]) for i in split_idx}


def fit_praxis_first_tree(X_train, y_train, X_test, y_test, label):
    """Fit PRAXIS's Rashomon set and evaluate its first tree (tree_index=0),
    matching run_PRAXIS.py's configuration. Returns None on failure."""
    print(f"    [PRAXIS] Fitting on {X_train.shape[1]} columns ({len(X_train)} rows)...")
    try:
        model = PRAXIS()
        model.fit(
            X_train,
            y_train,
            lambda_reg=LAMBDA_REG,
            depth_budget=DEPTH_BUDGET,
            rashomon_mult=RASHOMON_MULT,
            lookahead_k=LOOKAHEAD_K,
        )
        test_pred = np.asarray(model.get_predictions(PRAXIS_TREE_IDX, X_test))
        acc = accuracy_score(y_test, test_pred)
        paths_str, _ = model.get_tree_paths_str(PRAXIS_TREE_IDX)
        split_features = _split_features_from_paths(paths_str, list(X_train.columns))
    except Exception as e:
        print(f"    [PRAXIS] {label} FAILED: {e}")
        return None

    print(f"    [PRAXIS] Test accuracy: {acc:.4f}")
    print(f"    [PRAXIS] Raw split features: {sorted(split_features)}")

    return acc, test_pred, split_features


def run_one(dataset_name, param_tag, tgb_dir):
    out_dir = OUT_DIR / dataset_name / param_tag
    summary_path = out_dir / "ablation_summary_PRAXIS.json"
    if summary_path.exists():
        print(f"  Skipping {dataset_name}/{param_tag} (already done)")
        return json.loads(summary_path.read_text())

    counts_path = tgb_dir / "binary_variable_counts.csv"
    if not counts_path.exists():
        print(f"  Skipping {dataset_name}/{param_tag} (no binary_variable_counts.csv found)")
        return None

    print(f"\n=== {dataset_name}/{param_tag} ===")
    print(f"  Loading TGB outputs from {tgb_dir}...")
    X_train = pd.read_csv(tgb_dir / "X_train_guessed.csv")
    X_test = pd.read_csv(tgb_dir / "X_test_guessed.csv")
    y_train = pd.read_csv(tgb_dir / "y_train.csv").squeeze()
    y_test = pd.read_csv(tgb_dir / "y_test.csv").squeeze()
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    print("  Ranking raw features by summed TGB split importance...")
    counts_df = pd.read_csv(counts_path)
    counts_df["raw_feature"] = counts_df["binary_variable"].map(raw_feature_of)
    raw_importance = counts_df.groupby("raw_feature")["importance"].sum().sort_values(ascending=False)
    root_features = list(raw_importance.head(TOP_K_ABLATE).index)
    print(f"  Root feature(s) to ablate: {root_features}")

    def build_stage(label, X_tr, y_tr, X_te, y_te):
        _, tgb_acc, tgb_pred, tgb_feats = fit_tgb_gbdt(X_tr, y_tr, X_te, y_te)
        praxis_result = fit_praxis_first_tree(X_tr, y_tr, X_te, y_te, label)
        print(f"  Bootstrapping 95% CI for {label} TGB accuracy...")
        _, tgb_lo, tgb_hi = bootstrap_accuracy_ci(y_te, tgb_pred)

        stage = {
            "tgb_accuracy": tgb_acc,
            "tgb_ci95": [tgb_lo, tgb_hi],
            "tgb_split_features": sorted(tgb_feats),
        }
        if praxis_result is None:
            stage.update({
                "praxis_accuracy": None,
                "praxis_ci95": None,
                "praxis_split_features": None,
                "praxis_tgb_overlap_jaccard": None,
                "praxis_error": "PRAXIS fit/predict failed (see log above)",
            })
        else:
            praxis_acc, praxis_pred, praxis_feats = praxis_result
            print(f"  Bootstrapping 95% CI for {label} PRAXIS accuracy...")
            _, praxis_lo, praxis_hi = bootstrap_accuracy_ci(y_te, praxis_pred)
            overlap = jaccard(tgb_feats, praxis_feats)
            print(f"  {label.capitalize()} PRAXIS/TGB feature overlap (Jaccard): {overlap:.3f}")
            stage.update({
                "praxis_accuracy": praxis_acc,
                "praxis_ci95": [praxis_lo, praxis_hi],
                "praxis_split_features": sorted(praxis_feats),
                "praxis_tgb_overlap_jaccard": overlap,
            })
        return stage

    # --- Baseline ---
    print("  --- Baseline (root feature present) ---")
    baseline = build_stage("baseline", X_train, y_train, X_test, y_test)

    # --- Ablated: drop columns derived from the root feature(s) ---
    print(f"  --- Ablated (dropping columns for {root_features}) ---")
    drop_cols = [c for c in X_train.columns if raw_feature_of(c) in root_features]
    print(f"  Dropping {len(drop_cols)} columns: {drop_cols}")
    X_train_abl = X_train.drop(columns=drop_cols)
    X_test_abl = X_test.drop(columns=drop_cols)

    result = {
        "dataset": dataset_name,
        "param_tag": param_tag,
        "root_features_dropped": root_features,
        "n_dropped_columns": len(drop_cols),
        "baseline": baseline,
    }

    if X_train_abl.shape[1] == 0:
        print("  All columns removed by ablation; skipping refit for this combo.")
        result["ablated"] = None
        result["error"] = "ablating root feature(s) removed all columns; skipped"
    else:
        ablated = build_stage("ablated", X_train_abl, y_train, X_test_abl, y_test)
        result["ablated"] = ablated

        def _delta(key):
            b, a = baseline.get(key), ablated.get(key)
            return (a - b) if (b is not None and a is not None) else None

        result["delta"] = {
            "tgb_accuracy": _delta("tgb_accuracy"),
            "praxis_accuracy": _delta("praxis_accuracy"),
            "overlap_jaccard": _delta("praxis_tgb_overlap_jaccard"),
        }
        print(f"  TGB accuracy:    baseline={baseline['tgb_accuracy']:.4f}  ablated={ablated['tgb_accuracy']:.4f}")
        if baseline["praxis_accuracy"] is not None and ablated["praxis_accuracy"] is not None:
            print(f"  PRAXIS accuracy: baseline={baseline['praxis_accuracy']:.4f}  ablated={ablated['praxis_accuracy']:.4f}")
            print(f"  PRAXIS/TGB feature overlap (Jaccard): baseline={baseline['praxis_tgb_overlap_jaccard']:.3f}  "
                  f"ablated={ablated['praxis_tgb_overlap_jaccard']:.3f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2))
    print(f"  Saved summary to {summary_path}")
    return result


def main():
    only_dataset = sys.argv[1] if len(sys.argv) > 1 else None
    only_param_tag = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Scanning {TGB_DIR} for dataset/param_tag combos"
          + (f" (dataset={only_dataset})" if only_dataset else "")
          + (f" (param_tag={only_param_tag})" if only_param_tag else "") + "...")

    all_results = []
    for dataset_dir in sorted(TGB_DIR.iterdir()):
        if not dataset_dir.is_dir():
            continue
        if only_dataset and dataset_dir.name != only_dataset:
            continue
        for param_dir in sorted(dataset_dir.iterdir()):
            if not param_dir.is_dir():
                continue
            if only_param_tag and param_dir.name != only_param_tag:
                continue
            result = run_one(dataset_dir.name, param_dir.name, param_dir)
            if result is not None:
                all_results.append(result)

    print(f"\nProcessed {len(all_results)} dataset/param_tag combos. Building aggregate summary...")
    rows = []
    for r in all_results:
        if "baseline" not in r or r.get("ablated") is None:
            continue
        rows.append({
            "dataset": r["dataset"],
            "param_tag": r["param_tag"],
            "root_features_dropped": ";".join(r["root_features_dropped"]),
            "tgb_acc_baseline": r["baseline"]["tgb_accuracy"],
            "tgb_acc_ablated": r["ablated"]["tgb_accuracy"],
            "praxis_acc_baseline": r["baseline"]["praxis_accuracy"],
            "praxis_acc_ablated": r["ablated"]["praxis_accuracy"],
            "overlap_baseline": r["baseline"]["praxis_tgb_overlap_jaccard"],
            "overlap_ablated": r["ablated"]["praxis_tgb_overlap_jaccard"],
        })
    if rows:
        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(OUT_DIR / "summary_PRAXIS.csv", index=False)
        print(f"\nSummary saved to {OUT_DIR / 'summary_PRAXIS.csv'}")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
