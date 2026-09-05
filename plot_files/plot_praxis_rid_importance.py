import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent

BASEDIR = current
rid_results_dir = BASEDIR / "benchmarks_TGB_results_all"

# Sequential red ramp (light -> dark), one hue for magnitude
SEQUENTIAL_RED = [
    "#fbdccd", "#f8c6b1", "#f5b096", "#f2997a", "#ef825e", "#ec6b43",
    "#e85430", "#d4402a", "#bc3325", "#a3271f", "#8a1c19", "#711214", "#590a0e",
]
IMPORTANCE_CMAP = LinearSegmentedColormap.from_list("importance_red", SEQUENTIAL_RED)
NOT_USED_COLOR = "#e1e0d9"   # gridline-hairline gray: split had no RID value at that setting
GRIDLINE_COLOR = "#e1e0d9"
MUTED_INK = "#898781"
PRIMARY_INK = "#0b0b0b"

parameters = ["nest5_depth1", "nest5_depth2", "nest5_depth3", "nest10_depth1", "nest10_depth2", "nest10_depth3", "nest15_depth1", "nest15_depth2", "nest15_depth3", "nest20_depth1", "nest20_depth2", "nest20_depth3" , "nest25_depth1", "nest25_depth2", "nest25_depth3" , "nest30_depth1", "nest30_depth2", "nest30_depth3", "nest35_depth1", "nest35_depth2", "nest35_depth3", "nest40_depth1", "nest40_depth2", "nest40_depth3", "nest100_depth1", "nest100_depth2", "nest100_depth3", "nest200_depth1", "nest200_depth2", "nest200_depth3"]
datasets = sorted(d.name for d in rid_results_dir.iterdir() if d.is_dir())

# praxis_rid.json reports mean_sub_mr for every binarized feature (hundreds per
# dataset), not just the splits that surfaced as "top" splits elsewhere, so cap
# rows to the most important ones for a readable heatmap.
TOP_N = 40


def parse_param(param):
    """Extract (n_estimators, max_depth) from param string like 'nest5_depth1'."""
    m = re.match(r'nest(\d+)_depth(\d+)', param)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


DETAIL_COLS = ["parameter", "n_estimators", "max_depth", "split", "importance"]

matrix_out_dir = BASEDIR / "praxis_rid_matrix_csvs"
matrix_out_dir.mkdir(exist_ok=True)

heatmap_out_dir = BASEDIR / "praxis_rid_heatmap_plots"
heatmap_out_dir.mkdir(exist_ok=True)


def plot_rid_heatmap(matrix_df, dataset, out_path):
    """Heatmap of PRAXIS RID importance (rows) across the parameter sweep (columns)."""
    param_cols = [c for c in matrix_df.columns if c != "split"]
    plot_df = matrix_df.set_index("split")[param_cols]
    # Most important splits (by peak importance across the sweep) on top
    plot_df = plot_df.loc[plot_df.max(axis=1).sort_values(ascending=False).index]

    values = plot_df.to_numpy(dtype=float)
    n_rows, n_cols = values.shape

    fig_w = max(18, n_cols * 0.8 + 6)
    fig_h = max(6, n_rows * 0.45 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = IMPORTANCE_CMAP.copy()
    cmap.set_bad(NOT_USED_COLOR)
    masked = np.ma.masked_invalid(values)
    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0, vmax=np.nanmax(values))

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [p.replace("nest", "n").replace("_depth", " d") for p in param_cols],
        rotation=90, fontsize=14, color=MUTED_INK,
    )
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(plot_df.index, fontsize=11, color=PRIMARY_INK)

    # Recessive hairline grid between cells
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color=GRIDLINE_COLOR, linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Thicker separators every 3 columns: one nest-count group spans depth 1-3
    for x in range(2, n_cols - 1, 3):
        ax.axvline(x + 0.5, color=GRIDLINE_COLOR, linewidth=1.4)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.015)
    cbar.set_label("RID importance (mean_sub_mr)", color=MUTED_INK, fontsize=15)
    cbar.ax.tick_params(labelsize=13, color=MUTED_INK, labelcolor=MUTED_INK)

    ax.set_title(f"{dataset}: PRAXIS RID importance across parameter sweep (top {n_rows} splits)",
                 fontsize=19, color=PRIMARY_INK, loc="left", pad=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


for dataset in datasets:
    detail_rows = []

    for param in parameters:
        n_est, max_d = parse_param(param)
        rid_path = rid_results_dir / dataset / param / "praxis_rid.json"
        if not rid_path.exists():
            continue

        with open(rid_path) as f:
            rid_data = json.load(f)

        for feat, val in zip(rid_data["feature_names"], rid_data["mean_sub_mr"]):
            detail_rows.append({
                "parameter": param,
                "n_estimators": n_est,
                "max_depth": max_d,
                "split": feat,
                "importance": val,
            })

    if not detail_rows:
        continue

    detail_df = pd.DataFrame(detail_rows, columns=DETAIL_COLS)

    # Wide pivot: rows = unique splits, columns = parameter, cell = RID importance
    matrix_df = detail_df.pivot_table(index="split", columns="parameter",
                                       values="importance", aggfunc="first")
    matrix_df = matrix_df.reindex(columns=[p for p in parameters if p in matrix_df.columns])

    top_splits = matrix_df.max(axis=1).sort_values(ascending=False).head(TOP_N).index
    matrix_df = matrix_df.loc[top_splits]

    matrix_df = matrix_df.reset_index()
    matrix_path = matrix_out_dir / f"{dataset}_praxis_rid_matrix.csv"
    matrix_df.to_csv(matrix_path, index=False)
    print(f"Saved: {matrix_path.relative_to(BASEDIR)}")

    heatmap_path = heatmap_out_dir / f"{dataset}_praxis_rid_heatmap.png"
    plot_rid_heatmap(matrix_df, dataset, heatmap_path)
    print(f"Saved: {heatmap_path.relative_to(BASEDIR)}")
