import re
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent
BASEDIR = current

RESULTS_DIR = BASEDIR / "benchmarks_TGB_results_all"

PARAM_RE = re.compile(r"nest(\d+)_depth(\d+)")


def load_summary(path):
    with open(path) as f:
        return json.load(f)


datasets = sorted(d.name for d in RESULTS_DIR.iterdir() if d.is_dir())

for dataset in datasets:
    dataset_dir = RESULTS_DIR / dataset
    param_dirs = sorted(
        p for p in dataset_dir.iterdir()
        if p.is_dir() and PARAM_RE.match(p.name)
        and (p / "praxis_edit_distance_summary.json").exists()
    )
    if not param_dirs:
        print(f"Skipping {dataset}: no praxis_edit_distance_summary.json found")
        continue

    params = [p.name for p in param_dirs]
    nest_values = sorted({int(PARAM_RE.match(p).group(1)) for p in params})
    depth_values = sorted({int(PARAM_RE.match(p).group(2)) for p in params})

    mean_grid = np.full((len(nest_values), len(depth_values)), np.nan)
    std_grid = np.full((len(nest_values), len(depth_values)), np.nan)

    for p in param_dirs:
        nest, depth = (int(g) for g in PARAM_RE.match(p.name).groups())
        row = nest_values.index(nest)
        col = depth_values.index(depth)
        summary = load_summary(p / "praxis_edit_distance_summary.json")
        mean_grid[row, col] = summary["mean_distance"]
        std_grid[row, col] = summary["std_distance"]

    fig, ax = plt.subplots(figsize=(1.6 * len(depth_values) + 2, 0.5 * len(nest_values) + 2))
    im = ax.imshow(mean_grid, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, label="Mean pairwise tree edit distance")

    ax.set_xticks(range(len(depth_values)))
    ax.set_xticklabels([f"depth{d}" for d in depth_values])
    ax.set_yticks(range(len(nest_values)))
    ax.set_yticklabels([f"nest{n}" for n in nest_values])

    for row in range(len(nest_values)):
        for col in range(len(depth_values)):
            mean = mean_grid[row, col]
            std = std_grid[row, col]
            if np.isnan(mean):
                text = "N/A"
            else:
                text = f"{mean:.2f}\n±{std:.2f}"
            color = "white" if (not np.isnan(mean) and mean < np.nanmean(mean_grid)) else "black"
            ax.text(col, row, text, ha="center", va="center", fontsize=8, color=color)

    ax.set_title(f"PRAXIS pairwise tree edit distance (mean ± std) for {dataset}")
    fig.tight_layout()
    plot_path = dataset_dir / "edit_distance_heatmap.png"
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved {plot_path}")
