import re
import json
import time
import random
from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Pairwise TED is O(k^2) tree comparisons; cap how many sampled trees we compare.
MAX_TREES_FOR_EDIT_DISTANCE = 200

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent
BASEDIR = current

RESULTS_DIR = BASEDIR / "benchmarks_TGB_results_all"


class _Node:
    __slots__ = ("label", "children")

    def __init__(self, label, children=None):
        self.label = label
        self.children = children or []


def _parse_path(path_str):
    """"[+4, -56]" -> [4, -56] (signed feature indices, already 0-indexed)."""
    return [int(tok) for tok in re.findall(r"[+-]\d+", path_str)]


def build_tree(paths_str, predictions):
    """Reconstruct the ordered binary tree PRAXIS grew from its leaf paths,
    matching PRAXIS.plot_tree's reconstruction. +f -> left/True, -f -> right/False.
    """

    class _Raw:
        __slots__ = ("feature", "left", "right", "prediction")

        def __init__(self):
            self.feature = None
            self.left = None
            self.right = None
            self.prediction = None

    root = _Raw()
    for path_str, pred in zip(paths_str, predictions):
        cur = root
        for signed_f in _parse_path(path_str):
            f = abs(signed_f)
            go_left = signed_f > 0
            if cur.feature is None:
                cur.feature = f
                cur.left = _Raw()
                cur.right = _Raw()
            cur = cur.left if go_left else cur.right
        cur.prediction = pred

    def to_labeled(node):
        if node.prediction is not None or (node.left is None and node.right is None):
            return _Node(f"leaf:{node.prediction}")
        return _Node(f"f{node.feature}", [to_labeled(node.left), to_labeled(node.right)])

    return to_labeled(root)


def _annotate(root):
    """Postorder node list + leftmost-leaf-descendant index per node (Zhang-Shasha prep)."""
    postorder_nodes = []
    lmld = []

    def visit(node):
        if not node.children:
            idx = len(postorder_nodes)
            postorder_nodes.append(node)
            lmld.append(idx)
            return idx
        child_indices = [visit(c) for c in node.children]
        idx = len(postorder_nodes)
        postorder_nodes.append(node)
        lmld.append(lmld[child_indices[0]])
        return idx

    visit(root)
    return postorder_nodes, lmld


def tree_edit_distance(root1, root2, insert_cost=1.0, delete_cost=1.0, rename_cost=None):
    """Zhang-Shasha ordered tree edit distance between two labeled trees."""
    if rename_cost is None:
        rename_cost = lambda a, b: 0.0 if a.label == b.label else 1.0

    nodes1, lmld1 = _annotate(root1)
    nodes2, lmld2 = _annotate(root2)
    n, m = len(nodes1), len(nodes2)

    def keyroots(lmld):
        last = {}
        for i, l in enumerate(lmld):
            last[l] = i
        return sorted(last.values())

    treedist = [[0.0] * m for _ in range(n)]

    for i in keyroots(lmld1):
        for j in keyroots(lmld2):
            li, lj = lmld1[i], lmld2[j]
            rows, cols = i - li + 2, j - lj + 2
            forestdist = [[0.0] * cols for _ in range(rows)]

            for r in range(1, rows):
                forestdist[r][0] = forestdist[r - 1][0] + delete_cost
            for c in range(1, cols):
                forestdist[0][c] = forestdist[0][c - 1] + insert_cost

            for r in range(1, rows):
                p = li + r - 1
                for c in range(1, cols):
                    q = lj + c - 1
                    if lmld1[p] == li and lmld2[q] == lj:
                        cost_sub = rename_cost(nodes1[p], nodes2[q])
                        best = min(
                            forestdist[r - 1][c] + delete_cost,
                            forestdist[r][c - 1] + insert_cost,
                            forestdist[r - 1][c - 1] + cost_sub,
                        )
                        forestdist[r][c] = best
                        treedist[p][q] = best
                    else:
                        p_row, q_col = lmld1[p] - li, lmld2[q] - lj
                        forestdist[r][c] = min(
                            forestdist[r - 1][c] + delete_cost,
                            forestdist[r][c - 1] + insert_cost,
                            forestdist[p_row][q_col] + treedist[p][q],
                        )

    return treedist[n - 1][m - 1] if n and m else float(max(n, m))


targets = sorted(
    out_dir for out_dir in RESULTS_DIR.glob("*/*")
    if (out_dir / "praxis_sampled_trees.json").exists()
    and not (out_dir / "praxis_edit_distance_summary.json").exists()
)
print(f"Found {len(targets)} dataset/param dirs missing praxis_edit_distance_summary.json")

for out_dir in targets:
    dataset_name = out_dir.parent.name
    param_tag    = out_dir.name

    with open(out_dir / "praxis_sampled_trees.json") as f:
        metadata = json.load(f)

    trees = metadata["trees"]
    if len(trees) > MAX_TREES_FOR_EDIT_DISTANCE:
        trees = sorted(random.sample(trees, MAX_TREES_FOR_EDIT_DISTANCE), key=lambda t: t["tree_index"])

    print(f"\n[{dataset_name}/{param_tag}] Building {len(trees)} tree(s)...")
    tree_indices = [t["tree_index"] for t in trees]
    built_trees = [build_tree(t["paths"], t["predictions"]) for t in trees]

    k = len(built_trees)
    dist_matrix = np.zeros((k, k), dtype=float)

    print(f"  [PRAXIS-EditDistance] Computing {k * (k - 1) // 2} pairwise edit distances...")
    start = time.perf_counter()
    for i in range(k):
        for j in range(i + 1, k):
            d = tree_edit_distance(built_trees[i], built_trees[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    duration = time.perf_counter() - start

    df = pd.DataFrame(dist_matrix, index=tree_indices, columns=tree_indices)
    df.to_csv(out_dir / "praxis_edit_distance_matrix.csv")

    pair_distances = dist_matrix[np.triu_indices(k, k=1)]
    summary = {
        "n_trees_compared": k,
        "n_total_sampled_trees": len(metadata["trees"]),
        "n_pairs": int(pair_distances.size),
        "mean_distance": float(pair_distances.mean()) if pair_distances.size else None,
        "std_distance": float(pair_distances.std()) if pair_distances.size else None,
        "min_distance": float(pair_distances.min()) if pair_distances.size else None,
        "max_distance": float(pair_distances.max()) if pair_distances.size else None,
        "duration_seconds": duration,
    }
    with open(out_dir / "praxis_edit_distance_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  [PRAXIS-EditDistance] Done in {duration:.2f}s | mean={summary['mean_distance']}, max={summary['max_distance']}")

print("\nDone.")
