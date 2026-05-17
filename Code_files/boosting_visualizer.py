import json
import os
import re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class TreeNode:
    def __init__(self, node_id, feature=None, threshold=None, value=None):
        self.node_id = node_id
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.left = None
        self.right = None

def parse_xgboost(file_path):

    with open(file_path, "r") as f:
        lines = f.readlines()

    trees = []
    current_nodes = {}

    pattern = r'(\d+):\[(.*?)<(.*?)\] yes=(\d+),no=(\d+)'
    leaf_pattern = r'(\d+):leaf=(.*)'

    for line in lines:
        line = line.strip()

        if not line:
            if current_nodes:
                trees.append(current_nodes)
                current_nodes = {}
            continue

        split_match = re.match(pattern, line)
        if split_match:
            node_id = int(split_match.group(1))
            feature = split_match.group(2)
            threshold = float(split_match.group(3))
            left = int(split_match.group(4))
            right = int(split_match.group(5))
            node = TreeNode(node_id, feature, threshold)
            node.left = left
            node.right = right
            current_nodes[node_id] = node
        else:
            leaf_match = re.match(leaf_pattern, line)
            if leaf_match:
                node_id = int(leaf_match.group(1))
                value = float(leaf_match.group(2))
                current_nodes[node_id] = TreeNode(node_id, value=value)

    if current_nodes:
        trees.append(current_nodes)

    return trees


def parse_lightgbm(file_path):

    with open(file_path, "r") as f:
        text = f.read()

    header = text.split("Tree=")[0]
    names_match = re.search(r'feature_names=(.*)', header)
    feature_names = names_match.group(1).split() if names_match else []

    trees_raw = text.split("Tree=")[1:]
    parsed_trees = []

    for tree_text in trees_raw:

        feature_match = re.search(r'split_feature=(.*)', tree_text)
        threshold_match = re.search(r'threshold=(.*)', tree_text)
        left_match = re.search(r'left_child=(.*)', tree_text)
        right_match = re.search(r'right_child=(.*)', tree_text)
        leaf_match = re.search(r'leaf_value=(.*)', tree_text)

        if not feature_match or not left_match:
            continue

        features = feature_match.group(1).split()
        thresholds = threshold_match.group(1).split() if threshold_match else []
        left_children = list(map(int, left_match.group(1).split()))
        right_children = list(map(int, right_match.group(1).split()))
        leaf_values = (
            list(map(float, leaf_match.group(1).split())) if leaf_match else []
        )

        num_internal = len(features)
        tree = {}

        for i in range(num_internal):
            th = float(thresholds[i]) if i < len(thresholds) else None
            idx = int(features[i])
            fname = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            tree[i] = TreeNode(i, feature=fname, threshold=th)

        for i, val in enumerate(leaf_values):
            leaf_id = num_internal + i
            tree[leaf_id] = TreeNode(leaf_id, value=val)

        # Negative child index = leaf; positive = internal node.
        # LightGBM leaf refs are -(leaf_index + 1).
        for i in range(num_internal):
            lc = left_children[i]
            rc = right_children[i]
            tree[i].left = (num_internal + (-lc - 1)) if lc < 0 else lc
            tree[i].right = (num_internal + (-rc - 1)) if rc < 0 else rc

        parsed_trees.append(tree)

    return parsed_trees


def parse_catboost(file_path):

    with open(file_path, "r") as f:
        model = json.load(f)

    feat_map = {
        f["flat_feature_index"]: f["feature_id"]
        for f in model.get("features_info", {}).get("float_features", [])
    }

    trees = []

    for tree_data in model["oblivious_trees"]:

        splits = tree_data["splits"]
        leaf_values = tree_data["leaf_values"]
        depth = len(splits)
        num_internal = (1 << depth) - 1  # 2^depth - 1

        tree = {}

        # Internal nodes, built level by level so IDs are BFS-ordered.
        nid = 0
        for level in range(depth):
            split = splits[level]
            fname = feat_map.get(
                split["float_feature_index"],
                f"feature_{split['float_feature_index']}",
            )
            for _ in range(1 << level):
                tree[nid] = TreeNode(
                    nid,
                    feature=fname,
                    threshold=split["border"],
                )
                nid += 1

        # Leaf nodes
        for i, val in enumerate(leaf_values):
            leaf_id = num_internal + i
            tree[leaf_id] = TreeNode(leaf_id, value=val)

        # Wire up parent → children for internal levels
        for level in range(depth - 1):
            for pos in range(1 << level):
                parent_id = (1 << level) - 1 + pos
                tree[parent_id].left = (1 << (level + 1)) - 1 + 2 * pos
                tree[parent_id].right = (1 << (level + 1)) - 1 + 2 * pos + 1

        # Last internal level → leaves
        last = depth - 1
        for pos in range(1 << last):
            parent_id = (1 << last) - 1 + pos
            tree[parent_id].left = num_internal + 2 * pos
            tree[parent_id].right = num_internal + 2 * pos + 1

        trees.append(tree)

    return trees


def _compute_positions(tree, root_id=0):
    """Assign (x, y) coordinates; leaves spread evenly, y = -depth."""
    pos = {}
    counter = [0]

    def dfs(nid, depth):
        node = tree[nid]
        has_left = isinstance(node.left, int) and node.left in tree
        has_right = isinstance(node.right, int) and node.right in tree

        if not has_left and not has_right:
            pos[nid] = (counter[0], -depth)
            counter[0] += 1
            return

        if has_left:
            dfs(node.left, depth + 1)
        if has_right:
            dfs(node.right, depth + 1)

        lx = pos[node.left][0] if has_left else pos[node.right][0]
        rx = pos[node.right][0] if has_right else pos[node.left][0]
        pos[nid] = ((lx + rx) / 2, -depth)

    dfs(root_id, 0)
    return pos


def render_tree(tree, output_name):

    pos = _compute_positions(tree, root_id=0)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_span = max(xs) - min(xs) + 2
    y_span = max(ys) - min(ys) + 2
    fig, ax = plt.subplots(figsize=(max(10, x_span * 1.5), max(6, y_span * 1.5)))
    ax.axis('off')

    # Edges
    for nid, node in tree.items():
        if nid not in pos:
            continue
        x, y = pos[nid]
        for child_id, label in [(node.left, 'Yes'), (node.right, 'No')]:
            if not isinstance(child_id, int) or child_id not in pos:
                continue
            cx, cy = pos[child_id]
            ax.plot([x, cx], [y, cy], 'k-', linewidth=0.8, zorder=1)
            ax.text(
                (x + cx) / 2, (y + cy) / 2, label,
                fontsize=10, ha='center', va='center', color='black',
                fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='none', pad=1),
            )

    # Nodes
    for nid, node in tree.items():
        if nid not in pos:
            continue
        x, y = pos[nid]
        if node.value is not None:
            label = f"Leaf\n{round(node.value, 4)}"
            color = '#AED6F1'
        else:
            label = f"{node.feature}\n< {round(float(node.threshold), 4)}"
            color = '#A9DFBF'
        ax.text(
            x, y, label,
            ha='center', va='center', fontsize=8,
            bbox=dict(
                boxstyle='round,pad=0.4',
                facecolor=color,
                edgecolor='black',
                linewidth=0.8,
            ),
            zorder=2,
        )

    ax.set_xlim(min(xs) - 1, max(xs) + 1)
    ax.set_ylim(min(ys) - 1, max(ys) + 1)
    plt.tight_layout()
    plt.savefig(output_name + '.png', dpi=100, bbox_inches='tight')
    plt.close(fig)


ENSEMBLE_FILES = {
    "xgboost_ensemble.txt": ("xgboost", parse_xgboost),
    "lightgbm_ensemble.txt": ("lightgbm", parse_lightgbm),
    "catboost_ensemble.json": ("catboost", parse_catboost),
}


def process_benchmark_dir(benchmark_root, output_root):
    benchmark_root = Path(benchmark_root)
    output_root = Path(output_root)

    for dirpath, _, filenames in os.walk(benchmark_root):
        dirpath = Path(dirpath)
        rel = dirpath.relative_to(benchmark_root)

        for filename, (model_name, parser_fn) in ENSEMBLE_FILES.items():
            if filename not in filenames:
                continue

            filepath = dirpath / filename
            out_dir = output_root / rel
            out_dir.mkdir(parents=True, exist_ok=True)

            try:
                trees = parser_fn(str(filepath))
            except Exception as e:
                print(f"  [ERROR] parsing {filepath}: {e}")
                continue

            for i, tree in enumerate(trees):
                out_name = str(out_dir / f"{model_name}_tree_{i}")
                try:
                    render_tree(tree, out_name)
                except Exception as e:
                    print(f"  [ERROR] rendering {model_name} tree {i}: {e}")

            print(f"  {model_name}: {len(trees)} trees -> {out_dir}")


if __name__ == "__main__":

    project_root = Path(__file__).parent.parent
    output_root = project_root / "boosting_tree_visualizer"

    benchmarks = {
        "no_TGB": project_root / "benchmarks_no_TGB_results",
        "TGB": project_root / "benchmarks_TGB_results",
    }

    for label, bench_dir in benchmarks.items():
        if not bench_dir.exists():
            print(f"Skipping {bench_dir} (not found)")
            continue
        print(f"\nProcessing {label} ({bench_dir})")
        process_benchmark_dir(bench_dir, output_root / label)

    print("\nDone rendering trees.")
