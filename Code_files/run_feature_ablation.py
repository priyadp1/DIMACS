import sys
import json
import subprocess
import tempfile
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

TOP_K_ABLATE = 1          # how many top raw features to drop from the root
N_BOOTSTRAP = 1000
CI = 0.95
SUBSAMPLE_N = 10           # trees sampled when counting TGB split features (matches run_TGB.py)
SEED = 42

# Kept low deliberately: GOSDT's branch-and-bound queue grows with rows and
# depth_budget, and can exhaust memory well before time_limit triggers, so
# lower defaults reduce OOM risk during the sweep (see fit_gosdt_subprocess).
GOSDT_REG        = 0.001
GOSDT_DEPTH      = 4
GOSDT_TIME_LIMIT = 60
GOSDT_MAX_ROWS   = 5000

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent
BASEDIR = current

TGB_DIR = BASEDIR / "TGB_Variables_Feature_Importance"
OUT_DIR = BASEDIR / "feature_ablation_results"
OUT_DIR.mkdir(exist_ok=True)

TMP_DIR = BASEDIR / "tmp_ablation_gosdt"
TMP_DIR.mkdir(exist_ok=True)
GOSDT_WORKER = Path(__file__).parent / "_gosdt_ablation_worker.py"


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
    warm_labels = gbdt.predict(X_train)
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

    return gbdt, warm_labels, acc, test_pred, split_features


def fit_gosdt_subprocess(X_train, y_train, X_test, y_test, warm_labels, label):
    """Fit GOSDT in an isolated subprocess so an OOM kill only takes down
    this worker, not the whole ablation sweep. Returns None on failure."""
    print(f"    [GOSDT] Fitting on {X_train.shape[1]} columns ({len(X_train)} rows) in a subprocess "
          f"(depth_budget={GOSDT_DEPTH}, max_rows={GOSDT_MAX_ROWS})...")
    with tempfile.TemporaryDirectory(dir=TMP_DIR) as tmp:
        tmp = Path(tmp)
        X_train.to_csv(tmp / "X_train.csv", index=False)
        X_test.to_csv(tmp / "X_test.csv", index=False)
        y_train.to_csv(tmp / "y_train.csv", index=False)
        y_test.to_csv(tmp / "y_test.csv", index=False)
        pd.Series(warm_labels, name="warm_labels").to_csv(tmp / "warm_labels.csv", index=False)
        out_path = tmp / "result.json"

        proc = subprocess.run(
            [sys.executable, str(GOSDT_WORKER), str(tmp), str(out_path),
             str(GOSDT_DEPTH), str(GOSDT_REG), str(GOSDT_TIME_LIMIT), str(GOSDT_MAX_ROWS)],
        )
        if proc.returncode != 0 or not out_path.exists():
            reason = f"killed by signal {-proc.returncode} (likely out-of-memory)" \
                if proc.returncode < 0 else f"exited with code {proc.returncode}"
            print(f"    [GOSDT] {label} FAILED: subprocess {reason}")
            return None
        result = json.loads(out_path.read_text())

    acc = result["accuracy"]
    test_pred = np.array(result["test_pred"])
    split_features = set(result["split_features"])
    print(f"    [GOSDT] Test accuracy: {acc:.4f}")
    print(f"    [GOSDT] Raw split features: {sorted(split_features)}")

    return acc, test_pred, split_features


def run_one(dataset_name, param_tag, tgb_dir):
    out_dir = OUT_DIR / dataset_name / param_tag
    summary_path = out_dir / "ablation_summary.json"
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
        _, warm, tgb_acc, tgb_pred, tgb_feats = fit_tgb_gbdt(X_tr, y_tr, X_te, y_te)
        gosdt_result = fit_gosdt_subprocess(X_tr, y_tr, X_te, y_te, warm, label)
        print(f"  Bootstrapping 95% CI for {label} TGB accuracy...")
        _, tgb_lo, tgb_hi = bootstrap_accuracy_ci(y_te, tgb_pred)

        stage = {
            "tgb_accuracy": tgb_acc,
            "tgb_ci95": [tgb_lo, tgb_hi],
            "tgb_split_features": sorted(tgb_feats),
        }
        if gosdt_result is None:
            stage.update({
                "gosdt_accuracy": None,
                "gosdt_ci95": None,
                "gosdt_split_features": None,
                "gosdt_tgb_overlap_jaccard": None,
                "gosdt_error": "subprocess failed (see log above, likely OOM)",
            })
        else:
            gosdt_acc, gosdt_pred, gosdt_feats = gosdt_result
            print(f"  Bootstrapping 95% CI for {label} GOSDT accuracy...")
            _, gosdt_lo, gosdt_hi = bootstrap_accuracy_ci(y_te, gosdt_pred)
            overlap = jaccard(tgb_feats, gosdt_feats)
            print(f"  {label.capitalize()} GOSDT/TGB feature overlap (Jaccard): {overlap:.3f}")
            stage.update({
                "gosdt_accuracy": gosdt_acc,
                "gosdt_ci95": [gosdt_lo, gosdt_hi],
                "gosdt_split_features": sorted(gosdt_feats),
                "gosdt_tgb_overlap_jaccard": overlap,
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
            "gosdt_accuracy": _delta("gosdt_accuracy"),
            "overlap_jaccard": _delta("gosdt_tgb_overlap_jaccard"),
        }
        print(f"  TGB accuracy:   baseline={baseline['tgb_accuracy']:.4f}  ablated={ablated['tgb_accuracy']:.4f}")
        if baseline["gosdt_accuracy"] is not None and ablated["gosdt_accuracy"] is not None:
            print(f"  GOSDT accuracy: baseline={baseline['gosdt_accuracy']:.4f}  ablated={ablated['gosdt_accuracy']:.4f}")
            print(f"  GOSDT/TGB feature overlap (Jaccard): baseline={baseline['gosdt_tgb_overlap_jaccard']:.3f}  "
                  f"ablated={ablated['gosdt_tgb_overlap_jaccard']:.3f}")

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
            "gosdt_acc_baseline": r["baseline"]["gosdt_accuracy"],
            "gosdt_acc_ablated": r["ablated"]["gosdt_accuracy"],
            "overlap_baseline": r["baseline"]["gosdt_tgb_overlap_jaccard"],
            "overlap_ablated": r["ablated"]["gosdt_tgb_overlap_jaccard"],
        })
    if rows:
        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(OUT_DIR / "summary.csv", index=False)
        print(f"\nSummary saved to {OUT_DIR / 'summary.csv'}")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
