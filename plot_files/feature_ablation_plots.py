import re
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent
BASEDIR = current

PARAM_RE = re.compile(r"nest(\d+)_depth(\d+)")

GRIDLINE_COLOR = "#e1e0d9"
MUTED_INK = "#898781"
PRIMARY_INK = "#0b0b0b"
DIVERGING_CMAP = "RdBu"  # red = drop, blue = gain, white = no change

# Each source directory has its own model name baked into the JSON keys
# (e.g. "gosdt_accuracy") and, inconsistently, into the summary filename.
SOURCES = [
    {
        "dir_name": "feature_ablation_results_gosdt",
        "model_key": "gosdt",
        "model_label": "GOSDT",
        "json_names": ["ablation_summary.json"],
    },
    {
        "dir_name": "feature_ablation_results_praxis",
        "model_key": "praxis",
        "model_label": "PRAXIS",
        "json_names": ["ablation_summary_PRAXIS.json", "ablation_summary.json"],
    },
]


def find_summary_path(param_dir, json_names):
    for name in json_names:
        candidate = param_dir / name
        if candidate.exists():
            return candidate
    return None


def load_summary(path):
    with open(path) as f:
        return json.load(f)


def build_grid(param_dirs, json_names, value_fn):
    params = [p.name for p in param_dirs]
    nest_values = sorted({int(PARAM_RE.match(p).group(1)) for p in params})
    depth_values = sorted({int(PARAM_RE.match(p).group(2)) for p in params})

    grid = np.full((len(nest_values), len(depth_values)), np.nan)
    for p in param_dirs:
        nest, depth = (int(g) for g in PARAM_RE.match(p.name).groups())
        row = nest_values.index(nest)
        col = depth_values.index(depth)
        summary = load_summary(find_summary_path(p, json_names))
        if "error" in summary:
            continue
        grid[row, col] = value_fn(summary)

    return nest_values, depth_values, grid


def plot_delta_heatmap(grid, nest_values, depth_values, title, cbar_label, out_path):
    n_rows, n_cols = grid.shape
    fig_w = max(6, 1.6 * n_cols + 2)
    fig_h = max(4, 0.5 * n_rows + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    finite = grid[np.isfinite(grid)]
    limit = max(np.abs(finite).max(), 1e-6) if finite.size else 1e-6

    cmap = plt.get_cmap(DIVERGING_CMAP).copy()
    cmap.set_bad(GRIDLINE_COLOR)
    masked = np.ma.masked_invalid(grid)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    im = ax.imshow(masked, cmap=cmap, aspect="auto", norm=norm)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"depth{d}" for d in depth_values])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f"nest{n}" for n in nest_values])

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color=GRIDLINE_COLOR, linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for row in range(n_rows):
        for col in range(n_cols):
            value = grid[row, col]
            if np.isnan(value):
                text = "N/A"
                color = MUTED_INK
            else:
                text = f"{value:.3f}"
                color = "white" if abs(value) > 0.6 * limit else PRIMARY_INK
            ax.text(col, row, text, ha="center", va="center", fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(cbar_label, color=MUTED_INK)

    ax.set_title(title, fontsize=13, color=PRIMARY_INK, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def metrics_for(model_key, model_label):
    return [
        (
            "tgb_accuracy",
            lambda s: s["delta"]["tgb_accuracy"],
            "TGB accuracy change (ablated − baseline)",
            "Accuracy change",
        ),
        (
            "model_accuracy",
            lambda s: s["delta"][f"{model_key}_accuracy"],
            f"{model_label} accuracy change (ablated − baseline)",
            "Accuracy change",
        ),
        (
            "overlap_jaccard",
            lambda s: s["delta"]["overlap_jaccard"],
            f"TGB/{model_label} split overlap change (Jaccard)",
            "Jaccard change",
        ),
    ]


for source in SOURCES:
    results_dir = BASEDIR / source["dir_name"]
    if not results_dir.exists():
        print(f"Skipping {source['dir_name']}: directory not found")
        continue

    model_key = source["model_key"]
    model_label = source["model_label"]
    json_names = source["json_names"]
    metrics = metrics_for(model_key, model_label)

    datasets = sorted(d.name for d in results_dir.iterdir() if d.is_dir())

    for dataset in datasets:
        dataset_dir = results_dir / dataset
        param_dirs = sorted(
            p for p in dataset_dir.iterdir()
            if p.is_dir() and PARAM_RE.match(p.name)
            and find_summary_path(p, json_names) is not None
        )
        if not param_dirs:
            print(f"Skipping {source['dir_name']}/{dataset}: no ablation summary found")
            continue

        for metric_name, value_fn, title, cbar_label in metrics:
            nest_values, depth_values, grid = build_grid(param_dirs, json_names, value_fn)
            plot_path = dataset_dir / f"ablation_heatmap_{metric_name}.png"
            plot_delta_heatmap(
                grid, nest_values, depth_values,
                title=f"{model_label} – {dataset}: {title}",
                cbar_label=cbar_label,
                out_path=plot_path,
            )
            print(f"Saved {plot_path}")
