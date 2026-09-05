import json
import time
from pathlib import Path
import pandas as pd
from praxis import PRAXIS

LAMBDA_REG    = 0.01
DEPTH_BUDGET  = 5
RASHOMON_MULT = 0.03
LOOKAHEAD_K   = 1
RID_N_BOOT    = 10
RID_SEED      = 0

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent
BASEDIR = current

TGB_DIR     = BASEDIR / "TGB_Variables_Feature_Importance"
RESULTS_DIR = BASEDIR / "benchmarks_TGB_results_all"

targets = sorted(
    out_dir for out_dir in RESULTS_DIR.glob("*/*")
    if (out_dir / "praxis_results.txt").exists() and not (out_dir / "praxis_rid.json").exists()
)
print(f"Found {len(targets)} dataset/param dirs missing praxis_rid.json")

for out_dir in targets:
    dataset_name = out_dir.parent.name
    param_tag    = out_dir.name
    tgb_dir      = TGB_DIR / dataset_name / param_tag

    if not tgb_dir.exists():
        print(f"  [SKIP] {dataset_name}/{param_tag}: no TGB input dir at {tgb_dir}")
        continue

    print(f"\n[{dataset_name}/{param_tag}] Retraining PRAXIS + computing RID...")
    X_train = pd.read_csv(tgb_dir / "X_train_guessed.csv")
    y_train = pd.read_csv(tgb_dir / "y_train.csv").squeeze()
    feature_names = list(X_train.columns)

    model = PRAXIS()
    model.fit(
        X_train,
        y_train,
        lambda_reg=LAMBDA_REG,
        depth_budget=DEPTH_BUDGET,
        rashomon_mult=RASHOMON_MULT,
        lookahead_k=LOOKAHEAD_K,
    )

    try:
        rid_start = time.perf_counter()
        rid_out = model.compute_rid(
            X_train,
            y_train,
            n_boot=RID_N_BOOT,
            lambda_reg=LAMBDA_REG,
            depth_budget=DEPTH_BUDGET,
            rashomon_mult=RASHOMON_MULT,
            lookahead_k=LOOKAHEAD_K,
            seed=RID_SEED,
            memory_efficient=False,
            binning_map=None,
        )
        rid_duration = time.perf_counter() - rid_start

        with open(out_dir / "praxis_rid.json", "w") as f:
            json.dump(
                {
                    "feature_names": feature_names,
                    "mean_sub_mr": [float(v) for v in rid_out["mean_sub_mr"]],
                },
                f,
            )
        print(f"  [OK] RID computed in {rid_duration:.2f}s -> {out_dir / 'praxis_rid.json'}")
    except Exception as e:
        print(f"  [ERROR] RID failed for {dataset_name}/{param_tag}: {e}")

print("\nDone.")
