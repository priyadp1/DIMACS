import json
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from calculate_fairness_rashomon import SENSITIVE_ATTR_MAP, MAX_TREE_SAMPLE, EXTREME_KEYS
from praxis_rashomon_common import RESULTS_DIR, discover_param_tags, load_sampled_trees, load_test_data

OUT_DIR = RESULTS_DIR.parent / "fairness_tree_plots"

LEAF_COLORS = {0: "#E69F00", 1: "#009E73"}
NODE_FACE = "#DCEAF4"
EDGE_COLOR = "#4D4D4D"

METRIC_TO_CONTRIBUTION_KEY = {
    "statistical_parity": ("SPD", "spd_contribution"),
    "equal_opportunity": ("TPR", "eod_contribution"),
    "equalized_odds": ("FPR", "fpr_gap_contribution"),
}


def _sanitize(text):
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", text)).strip("_")


def _layout(tree):
    """Assign (x, y) grid positions to every node id: x by leaf order, y by depth."""
    is_leaf = tree.children_left == tree.children_right
    leaf_order = []

    def collect(nid):
        if is_leaf[nid]:
            leaf_order.append(nid)
            return
        collect(tree.children_left[nid])
        collect(tree.children_right[nid])

    collect(0)
    leaf_x = {nid: i for i, nid in enumerate(leaf_order)}
    positions = {}

    def dfs(nid, depth):
        if is_leaf[nid]:
            positions[nid] = (leaf_x[nid], -depth)
            return positions[nid]
        xl, _ = dfs(tree.children_left[nid], depth + 1)
        xr, _ = dfs(tree.children_right[nid], depth + 1)
        positions[nid] = (0.5 * (xl + xr), -depth)
        return positions[nid]

    dfs(0, 0)
    depth = max(-y for _, y in positions.values())
    return positions, is_leaf, len(leaf_order), depth


def _shrink_segment(x1, y1, x2, y2, r1, r2):
    dx, dy = x2 - x1, y2 - y1
    dist = (dx * dx + dy * dy) ** 0.5
    if dist == 0:
        return x1, y1, x2, y2
    ux, uy = dx / dist, dy / dist
    return x1 + ux * r1, y1 + uy * r1, x2 - ux * r2, y2 - uy * r2


def plot_tree_classifier(tree, feature_names, title=None, figsize=(8, 6), leaf_fairness=None, contribution_key=None):
    """Draw a TreeClassifier as a node/edge diagram (PRAXIS-style), return (fig, ax).

    leaf_fairness is a list of per-leaf dicts (see compute_leaf_fairness). contribution_key
    is a (label, field) pair, e.g. ("SPD", "spd_contribution") from METRIC_TO_CONTRIBUTION_KEY,
    picking which single fairness contribution to show for each leaf (matching the metric the
    plot is about) and which one drives the colored halo ring around each leaf.
    """
    positions, is_leaf, n_leaves, depth = _layout(tree)
    x_scale, y_scale = 3.2, 2.2
    positions = {nid: (x * x_scale, y * y_scale) for nid, (x, y) in positions.items()}

    width = max(figsize[0], 1.6 * n_leaves)
    height = max(figsize[1], 1.4 * depth)
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_axis_off()

    internal_r, leaf_r = 0.34, 0.40

    def node_radius(nid):
        return leaf_r if is_leaf[nid] else internal_r

    leaf_lookup = {rec["leaf_id"]: rec for rec in leaf_fairness} if leaf_fairness else {}
    contrib_label, contrib_field = contribution_key if contribution_key else (None, None)
    leaf_contribs = {
        nid: leaf_lookup[nid][contrib_field]
        for nid in range(tree.node_count) if is_leaf[nid] and nid in leaf_lookup
    } if leaf_lookup and contrib_field else {}
    max_abs_contrib = max((abs(c) for c in leaf_contribs.values()), default=0.0)

    for nid in range(tree.node_count):
        if is_leaf[nid]:
            continue
        x, y = positions[nid]
        for child in (tree.children_left[nid], tree.children_right[nid]):
            x2, y2 = positions[child]
            sx, sy, ex, ey = _shrink_segment(x, y, x2, y2, node_radius(nid), node_radius(child))
            ax.add_line(Line2D([sx, ex], [sy, ey], color=EDGE_COLOR, linewidth=2.2))

    for nid in range(tree.node_count):
        x, y = positions[nid]
        if is_leaf[nid]:
            pred = int(tree.classes_[tree.value[nid, 0, :].argmax()])
            face = LEAF_COLORS.get(pred, "#7F7F7F")
            label = str(pred)
            radius = leaf_r
        else:
            face, label, radius = NODE_FACE, None, internal_r

        if is_leaf[nid] and nid in leaf_contribs and max_abs_contrib > 0:
            contrib = leaf_contribs[nid]
            halo_color = plt.cm.coolwarm(0.5 + 0.5 * contrib / max_abs_contrib)
            halo_lw = 1.5 + 4.5 * (abs(contrib) / max_abs_contrib)
            ax.add_patch(Circle((x, y), radius + 0.12, facecolor="none",
                                 edgecolor=halo_color, linewidth=halo_lw, zorder=5))

        ax.add_patch(Circle((x, y), radius, facecolor=face, edgecolor=EDGE_COLOR, linewidth=1.6))

        if not is_leaf[nid]:
            ax.text(
                x, y + radius + 0.22, feature_names[tree.feature[nid]],
                ha="center", va="bottom", fontsize=11, color="#222222",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.95),
                zorder=10,
            )
        else:
            ax.text(x, y, label, ha="center", va="center", fontsize=12, fontweight="bold",
                     color="white", zorder=10)
            leaf_rec = leaf_lookup.get(nid)
            if leaf_rec is not None:
                gc = leaf_rec["group_counts"]
                caption = f"n={leaf_rec['n_samples']} ({gc['sens=0']}/{gc['sens=1']})"
                if contrib_field:
                    caption += f"\n{contrib_label}{leaf_rec[contrib_field]:+.2f}"
                ax.text(x, y - radius - 0.16, caption, ha="center", va="top", fontsize=8.5,
                         color="#222222", zorder=10)

    xs, ys = zip(*positions.values())
    pad = 1.6
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal", adjustable="box")
    if title:
        ax.set_title(title, fontsize=11)
    return fig, ax


def render_extreme_trees_for_param_tag(dataset_name, param_tag):
    extremes_path = RESULTS_DIR / dataset_name / param_tag / "praxis_fairness_extreme_trees.json"
    if not extremes_path.exists():
        return 0

    with open(extremes_path, "r", encoding="utf-8") as f:
        extremes = json.load(f)

    X_test, y_test, feature_names = load_test_data(dataset_name, param_tag)
    trees, _ = load_sampled_trees(dataset_name, param_tag, n_features=X_test.shape[1], max_trees=MAX_TREE_SAMPLE)

    out_dir = OUT_DIR / dataset_name / param_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for attr, by_metric in extremes["by_attribute"].items():
        for key in EXTREME_KEYS:
            entry = by_metric[key]
            tree = trees.get(entry["tree_index"])
            if tree is None:
                continue

            metric_name = key.replace("most_fair_by_", "").replace("least_fair_by_", "")
            metric_value = entry[f"{metric_name}_difference"]
            direction = "most" if key.startswith("most_fair") else "least"
            title = (
                f"{dataset_name}/{param_tag}  —  {attr}\n"
                f"{direction}-fair by {metric_name} = {metric_value:.4f} "
                f"(tree {entry['tree_index']}, acc={entry['accuracy']:.4f}, leaves={entry['n_leaves']})"
            )
            fig, _ = plot_tree_classifier(
                tree, feature_names, title=title,
                leaf_fairness=entry.get("leaf_fairness"),
                contribution_key=METRIC_TO_CONTRIBUTION_KEY[metric_name],
            )
            fname = f"{_sanitize(attr)}__{key}.png"
            fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            n_written += 1

    return n_written


def main():
    total = 0
    for dataset_name in SENSITIVE_ATTR_MAP:
        for param_tag in discover_param_tags(dataset_name):
            n = render_extreme_trees_for_param_tag(dataset_name, param_tag)
            if n:
                print(f"  {dataset_name}/{param_tag}: wrote {n} tree images")
                total += n
    print(f"\nWrote {total} tree images under {OUT_DIR}")


if __name__ == "__main__":
    main()
