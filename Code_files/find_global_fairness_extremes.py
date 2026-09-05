import json

from praxis_rashomon_common import RESULTS_DIR, discover_param_tags

DATASETS = ["compas", "diabetes_smote", "german_credit"]

METRICS = [
    ("statistical_parity_difference", "most_fair_by_statistical_parity", "least_fair_by_statistical_parity"),
    ("equal_opportunity_difference", "most_fair_by_equal_opportunity", "least_fair_by_equal_opportunity"),
    ("equalized_odds_difference", "most_fair_by_equalized_odds", "least_fair_by_equalized_odds"),
]


def find_global_extremes_for_dataset(dataset_name):
    param_tags = discover_param_tags(dataset_name)
    best = {}  # sens_attr -> metric -> {"most_fair": (...), "least_fair": (...)}

    for param_tag in param_tags:
        path = RESULTS_DIR / dataset_name / param_tag / "praxis_fairness_extreme_trees.json"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for sens_attr, by_metric in data["by_attribute"].items():
            best.setdefault(sens_attr, {})
            for metric_key, most_fair_key, least_fair_key in METRICS:
                best[sens_attr].setdefault(metric_key, {"most_fair": None, "least_fair": None})

                candidate_most = by_metric[most_fair_key]
                candidate_least = by_metric[least_fair_key]

                cur_most = best[sens_attr][metric_key]["most_fair"]
                if cur_most is None or candidate_most[metric_key] < cur_most["value"][metric_key]:
                    best[sens_attr][metric_key]["most_fair"] = {
                        "param_tag": param_tag,
                        "value": candidate_most,
                    }

                cur_least = best[sens_attr][metric_key]["least_fair"]
                if cur_least is None or candidate_least[metric_key] > cur_least["value"][metric_key]:
                    best[sens_attr][metric_key]["least_fair"] = {
                        "param_tag": param_tag,
                        "value": candidate_least,
                    }

    return best


def main():
    summary = {}
    for dataset_name in DATASETS:
        result = find_global_extremes_for_dataset(dataset_name)
        summary[dataset_name] = result

        out_path = RESULTS_DIR / dataset_name / "praxis_fairness_global_extremes.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print(f"\n{'='*70}\n  Dataset: {dataset_name}\n{'='*70}")
        for sens_attr, by_metric in result.items():
            print(f"  Sensitive attribute: {sens_attr}")
            for metric_key, _, _ in METRICS:
                mf = by_metric[metric_key]["most_fair"]
                lf = by_metric[metric_key]["least_fair"]
                print(f"    {metric_key}:")
                print(
                    f"      MOST fair:  param_tag={mf['param_tag']:<16} tree_index={mf['value']['tree_index']:<6} "
                    f"n_leaves={mf['value']['n_leaves']:<3} accuracy={mf['value']['accuracy']:.4f} "
                    f"value={mf['value'][metric_key]:.6f}"
                )
                print(
                    f"      LEAST fair: param_tag={lf['param_tag']:<16} tree_index={lf['value']['tree_index']:<6} "
                    f"n_leaves={lf['value']['n_leaves']:<3} accuracy={lf['value']['accuracy']:.4f} "
                    f"value={lf['value'][metric_key]:.6f}"
                )
        print(f"  -> wrote {out_path}")


if __name__ == "__main__":
    main()
