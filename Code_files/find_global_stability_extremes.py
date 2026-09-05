import json

from calculate_stability_rashomon import SMALL_DATASETS
from praxis_rashomon_common import RESULTS_DIR, discover_param_tags

DATASETS = SMALL_DATASETS
STABILITY_KEY = "test_stability_acc_mean"  # higher = more stable


def find_global_extremes_for_dataset(dataset_name):
    param_tags = discover_param_tags(dataset_name)
    most_stable = None
    least_stable = None

    for param_tag in param_tags:
        path = RESULTS_DIR / dataset_name / param_tag / "praxis_stability_summary.json"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for row in data["trees"]:
            if most_stable is None or row[STABILITY_KEY] > most_stable["value"][STABILITY_KEY]:
                most_stable = {"param_tag": param_tag, "value": row}
            if least_stable is None or row[STABILITY_KEY] < least_stable["value"][STABILITY_KEY]:
                least_stable = {"param_tag": param_tag, "value": row}

    if most_stable is None:
        return None
    return {"most_stable": most_stable, "least_stable": least_stable}


def main():
    for dataset_name in DATASETS:
        result = find_global_extremes_for_dataset(dataset_name)
        if result is None:
            print(f"  [SKIP] {dataset_name}: no cached praxis_stability_summary.json found")
            continue

        out_path = RESULTS_DIR / dataset_name / "praxis_stability_global_extremes.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        ms, ls = result["most_stable"], result["least_stable"]
        print(f"\n{'='*70}\n  Dataset: {dataset_name}\n{'='*70}")
        print(
            f"  MOST stable:  param_tag={ms['param_tag']:<16} tree_index={ms['value']['tree_index']:<6} "
            f"n_leaves={ms['value']['n_leaves']:<3} accuracy={ms['value']['accuracy']:.4f} "
            f"{STABILITY_KEY}={ms['value'][STABILITY_KEY]:.6f}"
        )
        print(
            f"  LEAST stable: param_tag={ls['param_tag']:<16} tree_index={ls['value']['tree_index']:<6} "
            f"n_leaves={ls['value']['n_leaves']:<3} accuracy={ls['value']['accuracy']:.4f} "
            f"{STABILITY_KEY}={ls['value'][STABILITY_KEY]:.6f}"
        )
        print(f"  -> wrote {out_path}")


if __name__ == "__main__":
    main()
