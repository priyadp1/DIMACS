"""Run PRAXIS (via run_PRAXIS.py) on every dataset/param_tag combo that doesn't
already have cached results, so the calculate_{fairness,robustness,stability}_
rashomon.py sweeps can cover them too. Mirrors run_from_TGB.py's per-param_tag
subprocess-call pattern; run_PRAXIS.py itself skips combos that already have
all expected output files, so this is safe to re-run/resume.
"""
import subprocess
import sys
from pathlib import Path

from praxis_rashomon_common import BASEDIR, TGB_DIR

# Ordered smallest/fastest first: diabetes_* are ~58-70K rows (~minutes/param_tag,
# per timing on diabetes_tomek/nest40_depth3: ~6.3 min for 79 features).
# creditcard_fraud_oversampled is ~364K rows (6x larger) and goes last so its
# potentially much longer runtime doesn't block the more tractable results.
DATASETS = [
    "diabetes_tomek",
    "diabetes_undersampled",
    "diabetes_oversampled",
    "creditcard_fraud_oversampled",
]

RUN_PRAXIS = Path(__file__).parent / "run_PRAXIS.py"

for dataset_name in DATASETS:
    dataset_tgb_dir = TGB_DIR / dataset_name
    if not dataset_tgb_dir.exists():
        print(f"[SKIP] No TGB variables found for {dataset_name}")
        continue

    param_tags = sorted(p.name for p in dataset_tgb_dir.iterdir() if p.is_dir())
    for param_tag in param_tags:
        print(f"\n=== {dataset_name}/{param_tag} ===", flush=True)
        result = subprocess.run(
            [sys.executable, str(RUN_PRAXIS), dataset_name, param_tag],
            cwd=str(BASEDIR),
        )
        if result.returncode != 0:
            print(f"  [ERROR] run_PRAXIS.py exited with code {result.returncode} for {dataset_name}/{param_tag}")
