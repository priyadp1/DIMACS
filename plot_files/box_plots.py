import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent
BASEDIR = current

RESULTS_DIR = BASEDIR / "benchmarks_TGB_results_all"
PLOTS_DIR = BASEDIR / "threshold_plots"
PLOTS_DIR.mkdir(exist_ok=True)

PARAM_RE = re.compile(r"nest(\d+)_depth(\d+)")

DATASET_COLORS = plt.get_cmap("tab10")


def n_splits(n_leaves):
    return n_leaves - 1


def get_leaf_values(row, leaf_metric_key):
    """Per-leaf signed values for a leaf-level fairness metric.

    "equalized_odds_active" dynamically picks whichever of the TPR-gap
    (eod_contribution) or FPR-gap (fpr_gap_contribution) components sums to the
    larger magnitude for THIS tree -- matching how equalized_odds_difference =
    max(|TPR gap|, |FPR gap|) is defined tree-level, so the leaf breakdown always
    corresponds to whichever side actually determines that tree's equalized-odds
    value. Verified against the cached data: the TPR side is dominant in ~71% of
    sampled trees, so a fixed choice of "always use the FPR side" would have been
    wrong most of the time.
    """
    leaves = row["leaf_fairness"]
    if leaf_metric_key == "equalized_odds_active":
        eod_vals = [l["eod_contribution"] for l in leaves]
        fpr_vals = [l["fpr_gap_contribution"] for l in leaves]
        return eod_vals if abs(sum(eod_vals)) >= abs(sum(fpr_vals)) else fpr_vals
    return [l[leaf_metric_key] for l in leaves]


def discover_param_dirs(dataset_dir):
    return sorted(p for p in dataset_dir.iterdir() if p.is_dir() and PARAM_RE.match(p.name))


def load_stability_trees(dataset_dir):
    """param_tag -> list of tree dicts (n_leaves, accuracy, test_stability_acc_mean)."""
    out = {}
    for tag_dir in discover_param_dirs(dataset_dir):
        f = tag_dir / "praxis_stability_summary.json"
        if f.exists():
            out[tag_dir.name] = json.loads(f.read_text()).get("trees", [])
    return out


def load_fairness_trees(dataset_dir):
    """param_tag -> {attr: [tree dicts (n_leaves, statistical_parity_difference)]}."""
    out = {}
    for tag_dir in discover_param_dirs(dataset_dir):
        f = tag_dir / "praxis_fairness_summary.json"
        if f.exists():
            out[tag_dir.name] = json.loads(f.read_text()).get("by_attribute", {})
    return out


def list_datasets():
    return sorted(p.name for p in RESULTS_DIR.iterdir() if p.is_dir())


# ── 1. Hyperparameter sweep vs thresholds ─────────────────────────────────────

def plot_hyperparam_sweep_vs_thresholds(dataset_dir, dataset_name):
    stability = load_stability_trees(dataset_dir)
    if not stability:
        return

    # depth -> nest -> (mean, min, max) splits across the Rashomon-set trees sampled
    by_depth = {}
    for tag, trees in stability.items():
        if not trees:
            continue
        nest, depth = (int(g) for g in PARAM_RE.match(tag).groups())
        splits = np.array([n_splits(t["n_leaves"]) for t in trees])
        by_depth.setdefault(depth, {})[nest] = (splits.mean(), splits.min(), splits.max())

    if not by_depth:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, depth in enumerate(sorted(by_depth)):
        nests = sorted(by_depth[depth])
        means = [by_depth[depth][n][0] for n in nests]
        mins  = [by_depth[depth][n][1] for n in nests]
        maxs  = [by_depth[depth][n][2] for n in nests]
        color = DATASET_COLORS(i)
        ax.plot(nests, means, marker="o", color=color, label=f"depth={depth}")
        ax.fill_between(nests, mins, maxs, color=color, alpha=0.15)

    ax.set_xlabel("n_estimators (nest)")
    ax.set_ylabel("Number of decision splits (thresholds) per tree")
    ax.set_title(f"Hyperparameter Sweep vs. Thresholds — {dataset_name}\n(mean over Rashomon set, shaded = min/max)")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    out_path = PLOTS_DIR / f"hyperparam_sweep_vs_thresholds_{dataset_name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── 2. Thresholds vs fairness ─────────────────────────────────────────────────

FAIRNESS_METRICS = [
    ("statistical_parity_difference", "Statistical parity difference", "thresholds_vs_fairness_statistical_parity.png"),
    ("equal_opportunity_difference",  "Equal opportunity difference",  "thresholds_vs_fairness_equal_opportunity.png"),
    ("equalized_odds_difference",     "Equalized odds difference",     "thresholds_vs_fairness_equalized_odds.png"),
]


def plot_thresholds_vs_fairness(metric_key, metric_label, out_name):
    per_dataset = {}
    for dataset_name in list_datasets():
        by_attr_per_tag = load_fairness_trees(RESULTS_DIR / dataset_name)
        points = []
        for by_attr in by_attr_per_tag.values():
            for attr_data in by_attr.values():
                for t in attr_data.get("trees", []):
                    if metric_key in t:
                        points.append((n_splits(t["n_leaves"]), t[metric_key]))
        if points:
            per_dataset[dataset_name] = points

    if not per_dataset:
        print(f"  No fairness data with '{metric_key}', skipping thresholds-vs-fairness ({metric_label}) plot.")
        return

    fig, axes = plt.subplots(1, len(per_dataset), figsize=(5 * len(per_dataset), 4.5), sharey=True)
    if len(per_dataset) == 1:
        axes = [axes]

    for ax, (dataset_name, points) in zip(axes, per_dataset.items()):
        xs, ys = zip(*points)
        ax.scatter(xs, ys, s=12, alpha=0.4, color="#4C72B0")
        ax.set_title(dataset_name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Number of decision splits (thresholds)")
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[0].set_ylabel(f"{metric_label}\n(lower = fairer)")
    fig.suptitle(f"Thresholds vs. Fairness — {metric_label} (per tree, all sampled Rashomon-set trees)", fontweight="bold")
    fig.tight_layout()
    out_path = PLOTS_DIR / out_name
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── 3. Thresholds vs stability ────────────────────────────────────────────────

def plot_thresholds_vs_stability():
    per_dataset = {}
    for dataset_name in list_datasets():
        stability = load_stability_trees(RESULTS_DIR / dataset_name)
        points = []
        for trees in stability.values():
            for t in trees:
                points.append((n_splits(t["n_leaves"]), t["test_stability_acc_mean"]))
        if points:
            per_dataset[dataset_name] = points

    if not per_dataset:
        print("  No stability data found, skipping thresholds-vs-stability plot.")
        return

    n = len(per_dataset)
    ncols = min(n, 4)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), sharey=False)
    axes = np.array(axes).reshape(-1)

    for ax, (dataset_name, points) in zip(axes, per_dataset.items()):
        xs, ys = zip(*points)
        ax.scatter(xs, ys, s=10, alpha=0.35, color="#DD8452")
        ax.set_title(dataset_name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Number of splits")
        ax.set_ylabel("Stability accuracy\n(mean under noise)")
        ax.grid(True, linestyle="--", alpha=0.5)

    for ax in axes[len(per_dataset):]:
        ax.axis("off")

    fig.suptitle("Thresholds vs. Stability (per tree, all sampled Rashomon-set trees)", fontweight="bold")
    fig.tight_layout()
    out_path = PLOTS_DIR / "thresholds_vs_stability.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── 4. Box plot of thresholds per dataset (with points, min/max whiskers) ────

def _box_with_points(ax, labels, data, point_color="#2ca02c"):
    """Draw a box plot with whiskers pinned to true min/max, individual jittered
    points overlaid, and explicit min/max markers."""
    positions = np.arange(1, len(labels) + 1)

    # whis=(0, 100) forces the whiskers to sit at the true min/max
    # rather than the default 1.5*IQR rule.
    ax.boxplot(
        data, positions=positions, widths=0.5, whis=(0, 100),
        showfliers=False,
        boxprops=dict(color="#4C72B0"),
        medianprops=dict(color="#DD8452", linewidth=2),
        whiskerprops=dict(color="#4C72B0"),
        capprops=dict(color="#4C72B0"),
    )

    rng = np.random.default_rng(0)
    for pos, vals in zip(positions, data):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(pos + jitter, vals, s=10, alpha=0.4, color=point_color, zorder=3)

    for pos, vals in zip(positions, data):
        ax.scatter(pos, min(vals), marker="v", s=40, color="black", zorder=4)
        ax.scatter(pos, max(vals), marker="^", s=40, color="black", zorder=4)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)


def plot_thresholds_boxplot_by_dataset():
    per_dataset = {}
    for dataset_name in list_datasets():
        stability = load_stability_trees(RESULTS_DIR / dataset_name)
        splits = [n_splits(t["n_leaves"]) for trees in stability.values() for t in trees]
        if splits:
            per_dataset[dataset_name] = splits

    if not per_dataset:
        print("  No data found, skipping thresholds box plot.")
        return

    labels = sorted(per_dataset, key=lambda d: np.mean(per_dataset[d]))
    data = [per_dataset[d] for d in labels]

    fig, ax = plt.subplots(figsize=(1.4 * len(labels) + 3, 6))
    _box_with_points(ax, labels, data)
    ax.set_ylabel("Number of decision splits (thresholds) per tree")
    ax.set_title("Distribution of Tree Thresholds by Dataset\n(points = individual trees, ▲▼ = max/min, box = IQR/median)",
                 fontweight="bold")
    fig.tight_layout()
    out_path = PLOTS_DIR / "thresholds_boxplot_by_dataset.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── 5. Leaf-level fairness contributions, pooled across every sampled tree ──

LEAF_CONTRIBUTION_METRICS = [
    ("spd_contribution", "Statistical parity contribution", "leaf_fairness_boxplot_statistical_parity.png", "#4C72B0"),
    ("eod_contribution", "Equal opportunity contribution", "leaf_fairness_boxplot_equal_opportunity.png", "#DD8452"),
    ("equalized_odds_active", "Equalized-odds contribution (active TPR/FPR side)", "leaf_fairness_boxplot_equalized_odds.png", "#2ca02c"),
]


def plot_leaf_fairness_boxplot(metric_key, metric_label, out_name, point_color):
    per_group = {}  # "dataset\n(attr)" -> list of leaf contribution values
    for dataset_name in list_datasets():
        by_tag = load_fairness_trees(RESULTS_DIR / dataset_name)
        by_attr_pooled = {}
        for by_attr in by_tag.values():
            for attr, attr_data in by_attr.items():
                for row in attr_data.get("trees", []):
                    if not row.get("leaf_fairness"):
                        continue
                    # Only leaves predicting the positive class carry a nonzero
                    # contribution by construction (see compute_leaf_fairness) --
                    # excluding the rest keeps the plot focused on the leaves that
                    # actually move the metric.
                    vals = get_leaf_values(row, metric_key)
                    by_attr_pooled.setdefault(attr, []).extend(
                        v for leaf, v in zip(row["leaf_fairness"], vals) if leaf["predicted_label"] == 1
                    )

        for attr, vals in by_attr_pooled.items():
            if vals:
                label = f"{dataset_name}\n({attr.split(' ')[0]})"
                per_group[label] = vals

    if not per_group:
        print(f"  No leaf fairness data with '{metric_key}', skipping leaf fairness box plot ({metric_label}).")
        return

    labels = sorted(per_group, key=lambda k: np.mean(np.abs(per_group[k])))
    data = [per_group[label] for label in labels]

    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 3, 6))
    _box_with_points(ax, labels, data, point_color=point_color)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_ylabel(f"Per-leaf {metric_label.lower()}\n(signed, positive-predicting leaves only)")
    ax.set_title("points = individual leaves, pooled across all trees/settings; ▲▼ = max/min", fontsize=8.5, pad=10)
    fig.suptitle(f"Leaf-Level {metric_label} by Dataset", fontweight="bold", fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = PLOTS_DIR / out_name
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── 6/7/8. Leaf-contribution localization: ranked bars, concentration curves,
#           and thresholds vs. concentration ─────────────────────────────────

# (tree-level metric key, leaf-level contribution key, display label, filename slug)
# Equalized odds uses "equalized_odds_active" (see get_leaf_values) since
# equalized_odds_difference = max(|TPR gap|, |FPR gap|) tree-level, and which side
# is dominant varies per tree (TPR dominant ~71% of the time in the sampled trees).
FAIRNESS_METRIC_PAIRS = [
    ("statistical_parity_difference", "spd_contribution", "Statistical Parity", "statistical_parity"),
    ("equal_opportunity_difference",  "eod_contribution", "Equal Opportunity",  "equal_opportunity"),
    ("equalized_odds_difference",     "equalized_odds_active", "Equalized Odds", "equalized_odds"),
]

TOP_FRAC = 0.20  # "top 20% of leaves" threshold used for the concentration summary stat


def load_fairness_groups():
    """(dataset, sensitive_attribute) -> {param_tag: attr_data}, where attr_data["trees"]
    holds every sampled tree's metrics + full leaf_fairness breakdown."""
    groups = {}
    for dataset_name in list_datasets():
        by_tag = load_fairness_trees(RESULTS_DIR / dataset_name)
        attrs = set(attr for by_attr in by_tag.values() for attr in by_attr)
        for attr in sorted(attrs):
            tag_data = {tag: by_attr[attr] for tag, by_attr in by_tag.items() if attr in by_attr}
            if tag_data:
                groups[(dataset_name, attr)] = tag_data
    return groups


def _sorted_abs_contributions(row, leaf_metric_key):
    """Return (signed values, leaf ids) sorted by descending |contribution|."""
    vals = get_leaf_values(row, leaf_metric_key)
    ids = [leaf["leaf_id"] for leaf in row["leaf_fairness"]]
    order = sorted(range(len(vals)), key=lambda i: -abs(vals[i]))
    return [vals[i] for i in order], [ids[i] for i in order]


def _concentration_at_frac(abs_sorted_desc, frac):
    """Fraction of total |contribution| captured by the top `frac` of leaves."""
    total = abs_sorted_desc.sum()
    if total <= 0:
        return None
    k = max(1, round(frac * len(abs_sorted_desc)))
    return abs_sorted_desc[:k].sum() / total


def _concentration_top1(abs_sorted_desc):
    """Fraction of total |contribution| captured by the single largest-magnitude
    leaf. Most sampled trees only have 2-4 leaves, where "top 20%" mechanically
    rounds up to "top 1 of n" anyway -- this makes that comparison explicit and
    comparable across trees of any size, instead of silently depending on rounding."""
    total = abs_sorted_desc.sum()
    if total <= 0:
        return None
    return abs_sorted_desc[0] / total


# (short name, function(abs_sorted_desc) -> fraction in [0,1], display label)
CONCENTRATION_DEFS = [
    ("top20", lambda a: _concentration_at_frac(a, TOP_FRAC), f"Top {int(TOP_FRAC*100)}% of leaves"),
    ("top1", _concentration_top1, "Single largest leaf"),
]


def plot_leaf_contribution_ranked(groups):
    """Plot 1: for the single least-fair tree per (dataset, attribute), bar chart
    of every leaf's signed contribution, sorted by descending |contribution|.
    Tests the localization hypothesis directly: a few large bars + a long
    near-zero tail vs. contributions spread evenly across leaves."""
    for tree_metric_key, leaf_metric_key, metric_label, slug in FAIRNESS_METRIC_PAIRS:
        per_group = {}
        for (dataset_name, attr), tag_data in groups.items():
            best_row = None
            best_val = -1.0
            for attr_data in tag_data.values():
                for row in attr_data.get("trees", []):
                    if not row.get("leaf_fairness"):
                        continue
                    val = abs(row.get(tree_metric_key, 0.0))
                    if val > best_val:
                        best_val, best_row = val, row
            if best_row is not None:
                per_group[(dataset_name, attr)] = best_row

        if not per_group:
            print(f"  No data for {metric_label}, skipping leaf contribution ranked plot.")
            continue

        n = len(per_group)
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), sharey=False)
        if n == 1:
            axes = [axes]

        for ax, ((dataset_name, attr), row) in zip(axes, per_group.items()):
            sorted_vals, sorted_ids = _sorted_abs_contributions(row, leaf_metric_key)
            colors = ["#4C72B0" if v >= 0 else "#C44E52" for v in sorted_vals]
            ax.bar(range(len(sorted_vals)), sorted_vals, color=colors)
            ax.set_xticks(range(len(sorted_vals)))
            ax.set_xticklabels([f"L{lid}" for lid in sorted_ids], fontsize=8)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(
                f"{dataset_name} ({attr.split(' ')[0]})\n"
                f"least-fair tree {row['tree_index']}: {metric_label}={row[tree_metric_key]:.3f}",
                fontsize=9,
            )
            ax.set_xlabel("Leaf, ranked by |contribution|")
            ax.grid(axis="y", linestyle="--", alpha=0.5)

        axes[0].set_ylabel(f"{metric_label} contribution")
        fig.suptitle(f"Leaf-Level {metric_label} Contribution, Ranked (least-fair tree per group)",
                     fontweight="bold", fontsize=13, y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        out_path = PLOTS_DIR / f"leaf_contribution_ranked_{slug}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {out_path}")


def plot_fairness_concentration_curve(groups):
    """Plot 2: cumulative % of total |contribution| captured vs. % of leaves
    included (ranked by |contribution|), averaged over every sampled tree.
    A curve that bows sharply toward the top-left (well above the y=x diagonal)
    means fairness disparity concentrates in a small subset of leaves."""
    grid = np.linspace(0, 100, 21)

    for _, leaf_metric_key, metric_label, slug in FAIRNESS_METRIC_PAIRS:
        per_group_curves, per_group_top = {}, {}
        for (dataset_name, attr), tag_data in groups.items():
            curves, top_vals = [], {name: [] for name, _, _ in CONCENTRATION_DEFS}
            for attr_data in tag_data.values():
                for row in attr_data.get("trees", []):
                    leaves = row.get("leaf_fairness", [])
                    if len(leaves) < 2:
                        continue
                    abs_sorted = np.sort(np.abs(get_leaf_values(row, leaf_metric_key)))[::-1]
                    total = abs_sorted.sum()
                    if total <= 0:
                        continue
                    n = len(abs_sorted)
                    x_full = np.concatenate([[0], 100 * np.arange(1, n + 1) / n])
                    y_full = np.concatenate([[0], 100 * np.cumsum(abs_sorted) / total])
                    curves.append(np.interp(grid, x_full, y_full))
                    for name, fn, _ in CONCENTRATION_DEFS:
                        top_vals[name].append(100 * fn(abs_sorted))
            if curves:
                per_group_curves[(dataset_name, attr)] = np.array(curves)
                per_group_top[(dataset_name, attr)] = (
                    {name: np.mean(vals) for name, vals in top_vals.items()}, len(curves)
                )

        if not per_group_curves:
            print(f"  No data for {metric_label}, skipping concentration curve plot.")
            continue

        n = len(per_group_curves)
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), sharey=True)
        if n == 1:
            axes = [axes]

        for ax, ((dataset_name, attr), curves) in zip(axes, per_group_curves.items()):
            mean_curve = curves.mean(axis=0)
            p25, p75 = np.percentile(curves, 25, axis=0), np.percentile(curves, 75, axis=0)
            top_means, n_trees = per_group_top[(dataset_name, attr)]
            mean_label = "mean (" + ", ".join(
                f"{name}={top_means[name]:.0f}%" for name, _, _ in CONCENTRATION_DEFS
            ) + ")"

            ax.fill_between(grid, p25, p75, color="#4C72B0", alpha=0.2, label="IQR across trees")
            ax.plot(grid, mean_curve, color="#4C72B0", linewidth=2, label=mean_label)
            ax.plot([0, 100], [0, 100], color="gray", linestyle="--", linewidth=1, label="even distribution")
            ax.set_title(f"{dataset_name} ({attr.split(' ')[0]}), n={n_trees} trees", fontsize=9)
            ax.set_xlabel("% of leaves (ranked by |contribution|)")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(fontsize=7, loc="lower right")

        axes[0].set_ylabel(f"Cumulative % of total |{metric_label}|\ncontribution captured")
        fig.suptitle(f"Fairness Concentration Curve — {metric_label}", fontweight="bold", fontsize=13, y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        out_path = PLOTS_DIR / f"fairness_concentration_curve_{slug}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {out_path}")


# Which concentration definition to use per dataset: compas/diabetes_smote trees
# only have 2-4 leaves, where "top 20%" mechanically rounds up to "top 1 of n"
# anyway (see get_leaf_values / CONCENTRATION_DEFS), so top-1 is the meaningful,
# well-defined choice there; german_credit's trees have enough leaves (4-13) for
# top-20% to actually differ from top-1 and be the more informative view.
GROUP_CONCENTRATION_CHOICE = {
    "compas": "top1",
    "diabetes_smote": "top1",
    "german_credit": "top20",
}
CONCENTRATION_DEFS_BY_NAME = {name: (fn, label) for name, fn, label in CONCENTRATION_DEFS}


def plot_thresholds_vs_concentration(groups):
    """Plot 3: per tree, number of decision splits (thresholds) vs. that tree's
    leaf concentration -- does more tree complexity spread fairness disparity
    across more leaves, or leave it just as concentrated? Each dataset panel uses
    whichever concentration definition suits its trees' typical leaf count (see
    GROUP_CONCENTRATION_CHOICE)."""
    for _, leaf_metric_key, metric_label, slug in FAIRNESS_METRIC_PAIRS:
        per_group_points = {}
        per_group_conc_name = {}
        for (dataset_name, attr), tag_data in groups.items():
            conc_name = GROUP_CONCENTRATION_CHOICE.get(dataset_name, "top20")
            conc_fn, _ = CONCENTRATION_DEFS_BY_NAME[conc_name]
            points = []
            for attr_data in tag_data.values():
                for row in attr_data.get("trees", []):
                    leaves = row.get("leaf_fairness", [])
                    if len(leaves) < 2:
                        continue
                    abs_sorted = np.sort(np.abs(get_leaf_values(row, leaf_metric_key)))[::-1]
                    conc = conc_fn(abs_sorted)
                    if conc is not None:
                        points.append((n_splits(row["n_leaves"]), 100 * conc))
            if points:
                per_group_points[(dataset_name, attr)] = points
                per_group_conc_name[(dataset_name, attr)] = conc_name

        if not per_group_points:
            print(f"  No data for {metric_label}, skipping thresholds-vs-concentration plot.")
            continue

        # Panels with more distinct threshold counts (e.g. german_credit, up to 9)
        # need proportionally more width than ones with only 3-4 (compas,
        # diabetes_smote) -- a uniform per-panel width crowded the wide ones.
        box_counts = [len(set(x for x, _ in points)) for points in per_group_points.values()]
        width_ratios = [max(3, 0.55 * c + 1.2) for c in box_counts]
        fig, axes = plt.subplots(
            1, len(per_group_points), figsize=(sum(width_ratios), 4.5),
            gridspec_kw={"width_ratios": width_ratios}, sharey=False,
        )
        if len(per_group_points) == 1:
            axes = [axes]

        for ax, ((dataset_name, attr), points), n_boxes in zip(axes, per_group_points.items(), box_counts):
            conc_name = per_group_conc_name[(dataset_name, attr)]
            _, conc_label = CONCENTRATION_DEFS_BY_NAME[conc_name]

            # One box per distinct threshold count instead of a raw scatter --
            # with up to ~1000 trees sampled per hyperparameter setting, the
            # per-tree cloud was too dense to read (e.g. thousands of trees
            # piled on 3-4 distinct x values).
            by_x = {}
            for x, y in points:
                by_x.setdefault(x, []).append(y)
            xs_sorted = sorted(by_x)
            fig_data = [by_x[x] for x in xs_sorted]
            counts = [len(by_x[x]) for x in xs_sorted]

            ax.boxplot(
                fig_data, positions=xs_sorted, widths=0.6,
                showfliers=False,
                boxprops=dict(color="#4C72B0"),
                medianprops=dict(color="#DD8452", linewidth=2),
                whiskerprops=dict(color="#4C72B0"),
                capprops=dict(color="#4C72B0"),
            )
            ax.set_xticks(xs_sorted)
            # Wide panels put the tree count below the tick on its own rotated
            # line instead of inline, since "n=1234" no longer fits between
            # closely-spaced ticks once there are more than ~5 boxes.
            if n_boxes > 5:
                ax.set_xticklabels(xs_sorted, fontsize=8)
                for x, c in zip(xs_sorted, counts):
                    ax.text(x, -0.18, f"n={c}", ha="center", va="top",
                            fontsize=6, rotation=90, transform=ax.get_xaxis_transform())
                label_pad = 38
            else:
                ax.set_xticklabels([f"{x}\n(n={c})" for x, c in zip(xs_sorted, counts)], fontsize=7)
                label_pad = None
            ax.set_title(f"{dataset_name} ({attr.split(' ')[0]})\n[{conc_label.lower()}]", fontsize=10, fontweight="bold")
            ax.set_xlabel("Number of decision splits (thresholds)", labelpad=label_pad)
            ax.set_ylabel(f"{conc_label} concentration (%)\nof total |{metric_label}| contribution")
            ax.grid(axis="y", linestyle="--", alpha=0.5)

        fig.suptitle(f"Thresholds vs. Fairness Concentration — {metric_label} (per tree)", fontweight="bold")
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.4, top=0.85)
        out_path = PLOTS_DIR / f"thresholds_vs_concentration_{slug}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {out_path}")


def main():
    print("Hyperparameter sweep vs thresholds:")
    for dataset_name in list_datasets():
        plot_hyperparam_sweep_vs_thresholds(RESULTS_DIR / dataset_name, dataset_name)

    print("\nThresholds vs fairness:")
    for metric_key, metric_label, out_name in FAIRNESS_METRICS:
        plot_thresholds_vs_fairness(metric_key, metric_label, out_name)

    print("\nThresholds vs stability:")
    plot_thresholds_vs_stability()

    print("\nThresholds box plot by dataset:")
    plot_thresholds_boxplot_by_dataset()

    print("\nLeaf-level fairness contributions:")
    for metric_key, metric_label, out_name, point_color in LEAF_CONTRIBUTION_METRICS:
        plot_leaf_fairness_boxplot(metric_key, metric_label, out_name, point_color)

    groups = load_fairness_groups()

    print("\nLeaf contribution ranked (localization hypothesis):")
    plot_leaf_contribution_ranked(groups)

    print("\nFairness concentration curves:")
    plot_fairness_concentration_curve(groups)

    print("\nThresholds vs fairness concentration:")
    plot_thresholds_vs_concentration(groups)

    print(f"\nAll plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
