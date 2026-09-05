"""Compute robustness metrics directly on the already-generated PRAXIS Rashomon
sets under benchmarks_TGB_results_all/, reusing the cached tree paths in
praxis_sampled_trees.json instead of re-running PRAXIS.

PRAXIS trees split on already-binarized (threshold-guessed) 0/1 features, so the
usual L-infinity Kantchelian attack (epsilon on raw feature values) is degenerate
here: any epsilon <= 0.5 can't flip a single guessed bit, and any epsilon > 0.5
lets every bit flip at once, collapsing accuracy to 0 regardless of tree quality.
Instead we use a Hamming/L0 attack budget: "the attacker may flip up to k of the
guessed binary threshold indicators". For a single decision tree this has a closed
form -- for each leaf, the cost to route a given row to it is just the number of
that leaf's path literals that disagree with the row's current bits (bits not on
the path are free). A row survives budget k iff every opposite-prediction leaf
costs more than k to reach.
"""
import json

import numpy as np

from praxis_rashomon_common import (
    RESULTS_DIR,
    discover_all_datasets,
    discover_param_tags,
    load_sampled_tree_leaves,
    load_test_data,
)

MAX_TREE_SAMPLE = 1000  # cap on how many cached sampled trees to evaluate per param_tag
MAX_BUDGET = 5  # report robust accuracy for k = 0..MAX_BUDGET flipped bits
REPORT_BUDGET = 2  # budget used to rank "most robust tree"


def leaf_cost_matrix(leaves, X_test):
    """Per-row, per-leaf Hamming cost to satisfy that leaf's path literals."""
    n_rows = X_test.shape[0]
    n_leaves = len(leaves)
    costs = np.zeros((n_rows, n_leaves), dtype=np.int32)
    leaf_preds = np.zeros(n_leaves, dtype=np.int32)

    for j, leaf in enumerate(leaves):
        leaf_preds[j] = leaf["pred"]
        literals = leaf["literals"]
        if literals:
            feat_idx = [f for f, _ in literals]
            bits = np.array([b for _, b in literals], dtype=np.int32)
            costs[:, j] = (X_test[:, feat_idx] != bits).sum(axis=1)

    return costs, leaf_preds


def hamming_robustness(leaves, X_test, y_test, max_budget=MAX_BUDGET):
    costs, leaf_preds = leaf_cost_matrix(leaves, X_test)

    clean_leaf = np.argmin(costs, axis=1)  # cost is exactly 0 at the row's own leaf
    clean_pred = leaf_preds[clean_leaf]
    clean_correct = clean_pred == y_test

    pred_matrix = np.broadcast_to(leaf_preds, costs.shape)
    diff_mask = pred_matrix != clean_pred[:, None]
    masked_costs = np.where(diff_mask, costs, np.iinfo(np.int32).max)
    min_flip_cost = masked_costs.min(axis=1)

    result = {"accuracy": float(clean_correct.mean())}
    for k in range(max_budget + 1):
        robust_correct = clean_correct & (min_flip_cost > k)
        result[f"robust_accuracy_k{k}"] = float(robust_correct.mean())
    return result


def main():
    for dataset_name in discover_all_datasets():
        print(f"\n{'='*60}\n  Dataset: {dataset_name}\n{'='*60}")
        param_tags = discover_param_tags(dataset_name)
        if not param_tags:
            print(f"  [SKIP] No cached PRAXIS Rashomon sets found for {dataset_name}")
            continue

        for param_tag in param_tags:
            out_path = RESULTS_DIR / dataset_name / param_tag / "praxis_robustness_summary.json"
            if out_path.exists():
                print(f"  [SKIP] {dataset_name}/{param_tag}: praxis_robustness_summary.json already exists")
                continue

            print(f"  --- {param_tag} ---")
            X_test, y_test, _ = load_test_data(dataset_name, param_tag)
            tree_leaves, meta = load_sampled_tree_leaves(dataset_name, param_tag, max_trees=MAX_TREE_SAMPLE)

            rows = []
            for idx, leaves in tree_leaves.items():
                metrics = hamming_robustness(leaves, X_test, y_test)
                rows.append({"tree_index": int(idx), "n_leaves": len(leaves), **metrics})

            most_robust = max(rows, key=lambda r: r[f"robust_accuracy_k{REPORT_BUDGET}"])
            result = {
                "max_budget": MAX_BUDGET,
                "report_budget": REPORT_BUDGET,
                "n_total_trees": meta["n_total_trees"],
                "n_trees_evaluated": len(rows),
                "trees": rows,
                "most_robust_tree": most_robust,
            }

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"    Evaluated {len(rows)} trees; most robust at k={REPORT_BUDGET}: "
                  f"tree {most_robust['tree_index']} "
                  f"(robust_acc={most_robust[f'robust_accuracy_k{REPORT_BUDGET}']:.4f}, "
                  f"clean_acc={most_robust['accuracy']:.4f})")


if __name__ == "__main__":
    main()
