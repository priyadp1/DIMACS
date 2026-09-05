import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from find_global_stability_extremes import DATASETS, STABILITY_KEY
from plot_fairness_trees import plot_tree_classifier
from praxis_rashomon_common import RESULTS_DIR, load_sampled_trees, load_test_data

OUT_DIR = RESULTS_DIR.parent / "stability_tree_plots_global"


def render_global_extremes_for_dataset(dataset_name):
    extremes_path = RESULTS_DIR / dataset_name / "praxis_stability_global_extremes.json"
    if not extremes_path.exists():
        print(f"  [SKIP] {dataset_name}: run find_global_stability_extremes.py first")
        return 0

    with open(extremes_path, "r", encoding="utf-8") as f:
        global_extremes = json.load(f)

    out_dir = OUT_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for direction in ("most_stable", "least_stable"):
        record = global_extremes[direction]
        param_tag, entry = record["param_tag"], record["value"]

        X_test, _, feature_names = load_test_data(dataset_name, param_tag)
        trees, _ = load_sampled_trees(dataset_name, param_tag, n_features=X_test.shape[1], max_trees=None)
        tree = trees.get(entry["tree_index"])
        if tree is None:
            continue

        label = direction.replace("_", "-")
        title = (
            f"{dataset_name}  (sweep-wide)\n"
            f"{label}: {STABILITY_KEY} = {entry[STABILITY_KEY]:.4f} "
            f"(param_tag={param_tag}, tree {entry['tree_index']}, "
            f"acc={entry['accuracy']:.4f}, leaves={entry['n_leaves']})"
        )
        fig, _ = plot_tree_classifier(tree, feature_names, title=title)
        fname = f"{direction}.png"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        n_written += 1

    return n_written


def main():
    total = 0
    for dataset_name in DATASETS:
        n = render_global_extremes_for_dataset(dataset_name)
        if n:
            print(f"  {dataset_name}: wrote {n} tree images")
            total += n
    print(f"\nWrote {total} tree images under {OUT_DIR}")


if __name__ == "__main__":
    main()
