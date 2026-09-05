import re
import json
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent
BASEDIR      = current
RESULTS_DIR  = BASEDIR / "benchmarks_TGB_results_all"

PARAM_PATTERN = re.compile(r'^nest(\d+)_depth(\d+)$')

TOP_N = 40  # max rows shown per heatmap; keeps very high-cardinality datasets legible


def load_dataset_values(dataset_dir):
    """Return long-form rows: (nest, depth, split_condition, mean_sub_mr)."""
    rows = []
    for param_dir in sorted(dataset_dir.iterdir()):
        m = PARAM_PATTERN.match(param_dir.name)
        rid_path = param_dir / "praxis_rid.json"
        if not m or not rid_path.exists():
            continue
        nest, depth = int(m.group(1)), int(m.group(2))
        data = json.loads(rid_path.read_text())
        for fname, val in zip(data["feature_names"], data["mean_sub_mr"]):
            rows.append((nest, depth, fname, float(val)))
    return rows


def plot_dataset(dataset_name, rows, out_dir, top_n=TOP_N):
    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["nest", "depth", "feature", "value"])
    agg = df.groupby(["feature", "nest", "depth"], as_index=False)["value"].sum()

    totals = agg.groupby("feature")["value"].sum().sort_values(ascending=False)
    top_features = totals.index[:top_n]
    agg = agg[agg["feature"].isin(top_features)]

    agg["param"] = agg.apply(lambda r: f"n{int(r.nest)}_d{int(r.depth)}", axis=1)
    param_order = sorted(agg[["nest", "depth"]].drop_duplicates().itertuples(index=False),
                          key=lambda t: (t.depth, t.nest))
    col_order = [f"n{int(n)}_d{int(d)}" for n, d in param_order]

    pivot = agg.pivot_table(index="feature", columns="param", values="value", fill_value=0.0)
    pivot = pivot.reindex(index=top_features, columns=col_order)

    n_rows, n_cols = pivot.shape
    fig_w = max(14, 0.7 * n_cols + 5)
    fig_h = max(8, 0.45 * n_rows + 3)
    plt.figure(figsize=(fig_w, fig_h))
    sns.heatmap(pivot, cmap="viridis", cbar_kws={"label": "sum(mean_sub_mr)"},
                linewidths=0.3, linecolor="white")
    plt.title(f"PRAXIS RID values across parameter sweep — {dataset_name}"
              + (f" (top {top_n} split conditions)" if len(totals) > top_n else ""),
              fontsize=16)
    plt.xlabel("nest_depth", fontsize=13)
    plt.ylabel("split condition", fontsize=13)
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()

    out_path = out_dir / "praxis_rid_heatmap.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=TOP_N,
                         help="max split conditions (rows) to show per heatmap")
    parser.add_argument("--dataset", type=str, default=None,
                         help="only process this dataset (folder name under benchmarks_TGB_results_all)")
    args = parser.parse_args()

    dataset_dirs = sorted(d for d in RESULTS_DIR.iterdir() if d.is_dir())
    if args.dataset:
        dataset_dirs = [d for d in dataset_dirs if d.name == args.dataset]

    for dataset_dir in dataset_dirs:
        rows = load_dataset_values(dataset_dir)
        if not rows:
            continue
        out_path = plot_dataset(dataset_dir.name, rows, dataset_dir, top_n=args.top_n)
        n_features = len({r[2] for r in rows})
        n_combos = len({(r[0], r[1]) for r in rows})
        print(f"[OK] {dataset_dir.name}: {n_features} features x {n_combos} param combos -> {out_path}")


if __name__ == "__main__":
    main()
