import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent
BASEDIR = current
RESULTS_DIR = BASEDIR / "benchmarks_TGB_results_all"

PARAM_RE = re.compile(r"nest(\d+)_depth(\d+)")


def discover_param_tag_values(dataset_dir, summary_name):
    """Return {param_tag: loaded_json} for every param_tag under dataset_dir
    matching nest{N}_depth{D} that has the given cached summary file."""
    values = {}
    for tag_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        if not PARAM_RE.match(tag_dir.name):
            continue
        summary_path = tag_dir / summary_name
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                values[tag_dir.name] = json.load(f)
    return values


def plot_heatmap(param_tag_values, value_fn, label_fn, title, colorbar_label, out_path, cmap="viridis"):
    """Build and save one nest x depth heatmap.

    Args:
        param_tag_values: {param_tag: loaded_json_dict}
        value_fn: loaded_json_dict -> float (cell color)
        label_fn: loaded_json_dict -> str (cell text annotation)
    """
    nest_values = sorted({int(PARAM_RE.match(t).group(1)) for t in param_tag_values})
    depth_values = sorted({int(PARAM_RE.match(t).group(2)) for t in param_tag_values})

    grid = np.full((len(nest_values), len(depth_values)), np.nan)
    labels = [["N/A"] * len(depth_values) for _ in nest_values]

    for tag, data in param_tag_values.items():
        nest, depth = (int(g) for g in PARAM_RE.match(tag).groups())
        row = nest_values.index(nest)
        col = depth_values.index(depth)
        grid[row, col] = value_fn(data)
        labels[row][col] = label_fn(data)

    fig, ax = plt.subplots(figsize=(1.6 * len(depth_values) + 2, 0.5 * len(nest_values) + 2))
    im = ax.imshow(grid, cmap=cmap, aspect="auto")
    fig.colorbar(im, ax=ax, label=colorbar_label)

    ax.set_xticks(range(len(depth_values)))
    ax.set_xticklabels([f"depth{d}" for d in depth_values])
    ax.set_yticks(range(len(nest_values)))
    ax.set_yticklabels([f"nest{n}" for n in nest_values])

    for row in range(len(nest_values)):
        for col in range(len(depth_values)):
            val = grid[row, col]
            if np.isnan(val):
                color = "black"  # N/A cells render as blank/white background
            else:
                # Pick text color from the actual rendered cell color's perceived
                # luminance (not a value-vs-mean heuristic, which picks the wrong
                # color whenever values are tightly clustered -- common for the
                # fairness heatmaps, where most cells sit within ~0.001 of each
                # other but still span the full colormap).
                r, g, b, _ = im.cmap(im.norm(val))
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                color = "black" if luminance > 0.5 else "white"
            ax.text(col, row, labels[row][col], ha="center", va="center", fontsize=8, color=color)

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_robustness(dataset_dir, dataset_name):
    data = discover_param_tag_values(dataset_dir, "praxis_robustness_summary.json")
    if not data:
        return
    report_budget = next(iter(data.values())).get("report_budget", 2)
    key = f"robust_accuracy_k{report_budget}"

    plot_heatmap(
        data,
        value_fn=lambda d: d["most_robust_tree"][key],
        label_fn=lambda d: f"{d['most_robust_tree'][key]:.3f}\n(clean={d['most_robust_tree']['accuracy']:.3f})",
        title=f"Robustness (best tree, k={report_budget} bit-flip budget) for {dataset_name}",
        colorbar_label=f"Robust accuracy (k={report_budget})",
        out_path=dataset_dir / "robustness_heatmap.png",
    )


def plot_stability(dataset_dir, dataset_name):
    data = discover_param_tag_values(dataset_dir, "praxis_stability_summary.json")
    if not data:
        return

    plot_heatmap(
        data,
        value_fn=lambda d: d["most_stable_tree"]["test_stability_acc_mean"],
        label_fn=lambda d: (
            f"{d['most_stable_tree']['test_stability_acc_mean']:.3f}\n"
            f"(clean={d['most_stable_tree']['accuracy']:.3f})"
        ),
        title=f"Stability (best tree, mean accuracy under noise) for {dataset_name}",
        colorbar_label="Stability accuracy (mean under noise)",
        out_path=dataset_dir / "stability_heatmap.png",
    )
    plot_stability_sensitivity(dataset_dir, dataset_name, data)


# StabilityMetric evaluates 5 noise-level variants per tree (see
# rashomon-framework/module/metric/stability_metric.py): four fixed shifts of
# the per-feature flip probability q_f (more noise -> less noise, left to
# right) plus one resampled variant that isn't part of that shift ordering.
STABILITY_VARIANTS = [
    ("_minus0.2", "-0.2"),
    ("_minus0.1", "-0.1"),
    ("", "default"),
    ("_plus0.1", "+0.1"),
    ("_resample0.05", "resample\n±0.05"),
]


def plot_stability_sensitivity(dataset_dir, dataset_name, data):
    """Bar chart of the single best (nest, depth, tree) config's stability
    across all 5 noise-level variants StabilityMetric computes, so the
    heatmap's single-variant summary doesn't leave the rest of that data unused.
    """
    best_tag, best_data = max(data.items(), key=lambda kv: kv[1]["most_stable_tree"]["test_stability_acc_mean"])
    tree = best_data["most_stable_tree"]

    means = [tree[f"test_stability_acc_mean{suffix}"] for suffix, _ in STABILITY_VARIANTS]
    stds = [tree[f"test_stability_acc_std{suffix}"] for suffix, _ in STABILITY_VARIANTS]
    labels = [label for _, label in STABILITY_VARIANTS]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, means, yerr=stds, capsize=4, color="#4C72B0")
    ax.axhline(tree["accuracy"], color="black", linestyle="--", linewidth=1, label=f"clean accuracy ({tree['accuracy']:.3f})")

    ax.set_ylabel("Test accuracy under noise (mean ± std across trials)")
    ax.set_xlabel("Noise-level variant (shift applied to per-feature flip probability q_f)")
    ax.set_title(
        f"Stability vs. noise level for {dataset_name}\n"
        f"best config: {best_tag}, tree {tree['tree_index']} ({tree['n_leaves']} leaves)"
    )
    ax.legend()
    fig.tight_layout()
    out_path = dataset_dir / "stability_noise_sensitivity.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


def _slugify(attr_name):
    return re.sub(r"[^a-z0-9]+", "_", attr_name.lower()).strip("_")


def plot_fairness(dataset_dir, dataset_name):
    data = discover_param_tag_values(dataset_dir, "praxis_fairness_summary.json")
    if not data:
        return

    attrs = set()
    for d in data.values():
        attrs.update(d["by_attribute"].keys())

    for attr in sorted(attrs):
        attr_data = {tag: d for tag, d in data.items() if attr in d["by_attribute"]}
        if not attr_data:
            continue

        plot_heatmap(
            attr_data,
            value_fn=lambda d, attr=attr: d["by_attribute"][attr]["most_fair_by_statistical_parity"][
                "statistical_parity_difference"
            ],
            label_fn=lambda d, attr=attr: (
                f"{d['by_attribute'][attr]['most_fair_by_statistical_parity']['statistical_parity_difference']:.3f}\n"
                f"(acc={d['by_attribute'][attr]['most_fair_by_statistical_parity']['accuracy']:.3f})"
            ),
            title=f"Fairness (best tree, statistical parity difference, lower=fairer)\n{dataset_name} — {attr}",
            colorbar_label="Statistical parity difference (lower = fairer)",
            out_path=dataset_dir / f"fairness_heatmap_{_slugify(attr)}.png",
            cmap="viridis_r",
        )


def main():
    for dataset_dir in sorted(p for p in RESULTS_DIR.iterdir() if p.is_dir()):
        dataset_name = dataset_dir.name
        print(f"\n{'='*60}\n  Dataset: {dataset_name}\n{'='*60}")
        plot_robustness(dataset_dir, dataset_name)
        plot_stability(dataset_dir, dataset_name)
        plot_fairness(dataset_dir, dataset_name)


if __name__ == "__main__":
    main()
