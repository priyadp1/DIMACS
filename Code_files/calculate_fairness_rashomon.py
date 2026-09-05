import json
import numpy as np

from praxis_rashomon_common import (
    RESULTS_DIR,
    discover_param_tags,
    load_sampled_trees,
    load_test_data,
)
from module.metric.fairness_metric import (  # noqa: E402  (path set up by praxis_rashomon_common)
    statistical_parity_difference,
    equal_opportunity_difference,
    equalized_odds_difference,
)
from sklearn.metrics import accuracy_score

# dataset -> guessed threshold column(s) usable as a binary sensitive attribute
SENSITIVE_ATTR_MAP = {
    "compas": ["sex=female <= 0.5"],
    "diabetes_smote": ["race <= 0.5", "gender <= 0.5"],
    "german_credit": ["Sex_female <= 0.5"],
}

MAX_TREE_SAMPLE = 1000  # cap on how many cached sampled trees to evaluate per param_tag


def _feature_root(col_name):
    """Strip a guessed threshold column down to its underlying raw attribute name.

    "gender <= 0.5" -> "gender"; "sex=female <= 0.5" -> "sex"; "race <= 2.5" -> "race";
    "Sex_female <= 0.5" -> "Sex". Columns sharing a root are different binarizations/
    levels of the same original attribute (e.g. every "race <= k.5" threshold, one-hot
    "sex=male"/"sex=female" pairs, or get_dummies-style "Sex_male"/"Sex_female" pairs),
    so they're treated as "related" to each other.
    """
    base = col_name.split(" <= ")[0].strip()
    base = base.split("=")[0]
    return base.split("_")[0]


def related_feature_indices(feature_names, sens_col):
    """Indices of every column derived from the same raw attribute as sens_col."""
    root = _feature_root(sens_col)
    return {i for i, name in enumerate(feature_names) if _feature_root(name) == root}


def filter_trees_excluding_features(trees, excluded_indices):
    """Drop trees that split on any feature index in excluded_indices anywhere in the tree.

    Used to exclude trees that make decisions directly on the sensitive attribute
    (or a column related to it, e.g. the male/female pair for gender) before
    computing fairness metrics on the remainder of the Rashomon set.
    """
    return {
        idx: tree for idx, tree in trees.items()
        if not any(f in excluded_indices for f in tree.feature if f != -2)
    }


def compute_leaf_fairness(tree, X_test, y_test, sens_attr):
    """Per-leaf breakdown of group composition and each leaf's signed contribution
    to the tree's fairness gaps.

    Since every leaf outputs one constant predicted label, the tree-level metrics
    decompose exactly as a sum over leaves that predict the positive class:
      statistical_parity_difference == abs(sum of spd_contribution)
      equal_opportunity_difference  == abs(sum of eod_contribution)
      equalized_odds_difference     == max(abs(sum of eod_contribution), abs(sum of fpr_gap_contribution))
    (each contribution is 0 for leaves predicting the negative class).
    """
    leaf_ids = tree.apply(X_test)
    classes = tree.classes_
    is_pos = y_test == 1
    is_neg = y_test == 0
    is_g0 = sens_attr == 0
    is_g1 = sens_attr == 1

    n_g0, n_g1 = int(is_g0.sum()), int(is_g1.sum())
    n_pos_g0, n_pos_g1 = int((is_g0 & is_pos).sum()), int((is_g1 & is_pos).sum())
    n_neg_g0, n_neg_g1 = int((is_g0 & is_neg).sum()), int((is_g1 & is_neg).sum())

    def rate(count, total):
        return count / total if total else 0.0

    leaves_out = []
    for leaf_id in sorted(set(leaf_ids.tolist())):
        mask = leaf_ids == leaf_id
        counts = tree.value[leaf_id, 0, :]
        predicted_label = int(classes[np.argmax(counts)])

        n_g0_here, n_g1_here = int((mask & is_g0).sum()), int((mask & is_g1).sum())
        n_pos_g0_here, n_pos_g1_here = int((mask & is_g0 & is_pos).sum()), int((mask & is_g1 & is_pos).sum())
        n_neg_g0_here, n_neg_g1_here = int((mask & is_g0 & is_neg).sum()), int((mask & is_g1 & is_neg).sum())

        leaves_out.append({
            "leaf_id": int(leaf_id),
            "predicted_label": predicted_label,
            "n_samples": int(mask.sum()),
            "group_counts": {"sens=0": n_g0_here, "sens=1": n_g1_here},
            "group_counts_where_y=1": {"sens=0": n_pos_g0_here, "sens=1": n_pos_g1_here},
            "group_counts_where_y=0": {"sens=0": n_neg_g0_here, "sens=1": n_neg_g1_here},
            "spd_contribution": predicted_label * (rate(n_g1_here, n_g1) - rate(n_g0_here, n_g0)),
            "eod_contribution": predicted_label * (rate(n_pos_g1_here, n_pos_g1) - rate(n_pos_g0_here, n_pos_g0)),
            "fpr_gap_contribution": predicted_label * (rate(n_neg_g1_here, n_neg_g1) - rate(n_neg_g0_here, n_neg_g0)),
        })

    return leaves_out


def compute_fairness_for_attr(trees, X_test, y_test, sens_attr, feature_names):
    rows = []
    for idx, tree in trees.items():
        preds = tree.predict(X_test)
        rows.append({
            "tree_index": int(idx),
            "n_leaves": tree.leaves(),
            "accuracy": float(accuracy_score(y_test, preds)),
            "statistical_parity_difference": float(statistical_parity_difference(y_test, preds, sens_attr)),
            "equal_opportunity_difference": float(equal_opportunity_difference(y_test, preds, sens_attr)),
            "equalized_odds_difference": float(equalized_odds_difference(y_test, preds, sens_attr)),
            "leaf_fairness": compute_leaf_fairness(tree, X_test, y_test, sens_attr),
        })

    def with_tree(row):
        tree = trees[row["tree_index"]]
        return {
            **row,
            "tree_structure": tree.export_text(feature_names),
        }

    return {
        "n_trees_evaluated": len(rows),
        "trees": rows,
        "most_fair_by_statistical_parity": with_tree(min(rows, key=lambda r: r["statistical_parity_difference"])),
        "least_fair_by_statistical_parity": with_tree(max(rows, key=lambda r: r["statistical_parity_difference"])),
        "most_fair_by_equal_opportunity": with_tree(min(rows, key=lambda r: r["equal_opportunity_difference"])),
        "least_fair_by_equal_opportunity": with_tree(max(rows, key=lambda r: r["equal_opportunity_difference"])),
        "most_fair_by_equalized_odds": with_tree(min(rows, key=lambda r: r["equalized_odds_difference"])),
        "least_fair_by_equalized_odds": with_tree(max(rows, key=lambda r: r["equalized_odds_difference"])),
    }


def compute_fairness_for_param_tag(dataset_name, param_tag, sens_cols):
    X_test, y_test, feature_names = load_test_data(dataset_name, param_tag)
    available = [c for c in sens_cols if c in feature_names]
    if not available:
        print(f"  [SKIP] {dataset_name}/{param_tag}: none of {sens_cols} among guessed features")
        return None

    trees, meta = load_sampled_trees(dataset_name, param_tag, n_features=X_test.shape[1], max_trees=MAX_TREE_SAMPLE)

    by_attribute = {}
    for sens_col in available:
        excluded_indices = related_feature_indices(feature_names, sens_col)
        related_features = sorted(feature_names[i] for i in excluded_indices)
        filtered_trees = filter_trees_excluding_features(trees, excluded_indices)
        if not filtered_trees:
            print(f"    [SKIP] {sens_col}: every sampled tree splits on a related feature {related_features}")
            continue

        sens_attr = X_test[:, feature_names.index(sens_col)]
        by_attribute[sens_col] = compute_fairness_for_attr(filtered_trees, X_test, y_test, sens_attr, feature_names)
        by_attribute[sens_col]["related_features_excluded"] = related_features
        by_attribute[sens_col]["n_trees_excluded_for_related_features"] = len(trees) - len(filtered_trees)

    if not by_attribute:
        return None

    return {
        "sensitive_attributes": list(by_attribute.keys()),
        "n_total_trees": meta["n_total_trees"],
        "n_trees_loaded": len(trees),
        "by_attribute": by_attribute,
    }


EXTREME_KEYS = (
    "most_fair_by_statistical_parity",
    "least_fair_by_statistical_parity",
    "most_fair_by_equal_opportunity",
    "least_fair_by_equal_opportunity",
    "most_fair_by_equalized_odds",
    "least_fair_by_equalized_odds",
)


def extract_extreme_trees(result):
    """Pull just the most/least fair tree per metric out of a full param_tag result.

    Drops the "trees" (n_trees_evaluated rows for every sampled tree) so the
    output is small enough to skim directly.
    """
    return {
        "sensitive_attributes": result["sensitive_attributes"],
        "n_total_trees": result["n_total_trees"],
        "n_trees_loaded": result["n_trees_loaded"],
        "by_attribute": {
            attr: {key: r[key] for key in EXTREME_KEYS}
            for attr, r in result["by_attribute"].items()
        },
    }


def main():
    for dataset_name, sens_cols in SENSITIVE_ATTR_MAP.items():
        print(f"\n{'='*60}\n  Dataset: {dataset_name} (sensitive attributes: {sens_cols})\n{'='*60}")
        param_tags = discover_param_tags(dataset_name)
        if not param_tags:
            print(f"  [SKIP] No cached PRAXIS Rashomon sets found for {dataset_name}")
            continue

        for param_tag in param_tags:
            out_path = RESULTS_DIR / dataset_name / param_tag / "praxis_fairness_summary.json"
            if out_path.exists():
                print(f"  [SKIP] {dataset_name}/{param_tag}: praxis_fairness_summary.json already exists")
                continue

            print(f"  --- {param_tag} ---")
            result = compute_fairness_for_param_tag(dataset_name, param_tag, sens_cols)
            if result is None:
                continue

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            extremes_path = RESULTS_DIR / dataset_name / param_tag / "praxis_fairness_extreme_trees.json"
            with open(extremes_path, "w", encoding="utf-8") as f:
                json.dump(extract_extreme_trees(result), f, indent=2)

            summaries = ", ".join(
                f"{attr}: {r['n_trees_evaluated']} trees (excluded {r['n_trees_excluded_for_related_features']} for "
                f"splitting on {r['related_features_excluded']}), "
                f"spd most-fair={r['most_fair_by_statistical_parity']['statistical_parity_difference']:.4f} "
                f"least-fair={r['least_fair_by_statistical_parity']['statistical_parity_difference']:.4f}"
                for attr, r in result["by_attribute"].items()
            )
            print(f"    {summaries}")


if __name__ == "__main__":
    main()
