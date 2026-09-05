import re
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent
BASEDIR = current
results_dir = BASEDIR / "benchmarks_TGB_results_all"
os.makedirs(results_dir, exist_ok=True)
parameters = ["nest5_depth1", "nest5_depth2", "nest5_depth3", "nest10_depth1", "nest10_depth2", "nest10_depth3", "nest15_depth1", "nest15_depth2", "nest15_depth3", "nest20_depth1", "nest20_depth2", "nest20_depth3" , "nest25_depth1", "nest25_depth2", "nest25_depth3" , "nest30_depth1", "nest30_depth2", "nest30_depth3", "nest35_depth1", "nest35_depth2", "nest35_depth3", "nest40_depth1", "nest40_depth2", "nest40_depth3", "nest100_depth1", "nest100_depth2"]
datasets = ["creditcard_fraud_smote"]

PARAM_RE = re.compile(r"nest(\d+)_depth(\d+)")
nest_values = sorted({int(PARAM_RE.match(p).group(1)) for p in parameters})
depth_values = sorted({int(PARAM_RE.match(p).group(2)) for p in parameters})


def parse_praxis_results(path):
    text = path.read_text()
    acc = re.search(r"^Accuracy:\s*([\d.]+)", text, re.MULTILINE)
    rashomon_size = re.search(r"Rashomon set size:\s*(\d+)", text)
    return (
        float(acc.group(1)) if acc else None,
        int(rashomon_size.group(1)) if rashomon_size else None,
    )


for j in datasets:
    accuracy_grid = np.full((len(nest_values), len(depth_values)), np.nan)
    rashomon_grid = np.full((len(nest_values), len(depth_values)), np.nan)
    found_any = False

    for i in parameters:
        result_file = results_dir / j / i / "praxis_results.txt"
        if not result_file.exists():
            print(f"Skipping {j}/{i}: missing results file")
            continue
        accuracy, rashomon_size = parse_praxis_results(result_file)
        if accuracy is None or rashomon_size is None:
            print(f"Skipping {j}/{i}: could not parse accuracy/Rashomon set size")
            continue
        nest, depth = (int(g) for g in PARAM_RE.match(i).groups())
        row = nest_values.index(nest)
        col = depth_values.index(depth)
        accuracy_grid[row, col] = accuracy
        rashomon_grid[row, col] = rashomon_size
        found_any = True

    if not found_any:
        print(f"Skipping {j}: no data found")
        continue

    fig, ax = plt.subplots(figsize=(1.6 * len(depth_values) + 2, 0.5 * len(nest_values) + 2))
    im = ax.imshow(accuracy_grid, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, label="Accuracy")

    ax.set_xticks(range(len(depth_values)))
    ax.set_xticklabels([f"depth{d}" for d in depth_values])
    ax.set_yticks(range(len(nest_values)))
    ax.set_yticklabels([f"nest{n}" for n in nest_values])

    for row in range(len(nest_values)):
        for col in range(len(depth_values)):
            acc = accuracy_grid[row, col]
            size = rashomon_grid[row, col]
            if np.isnan(acc):
                text = "N/A"
            else:
                text = f"{acc:.3f}\n({int(size)})"
            color = "white" if (not np.isnan(acc) and acc < np.nanmean(accuracy_grid)) else "black"
            ax.text(col, row, text, ha="center", va="center", fontsize=8, color=color)

    ax.set_title(f"Accuracy (color) & Rashomon Set Size (label) for {j}")
    fig.tight_layout()
    plot_path = results_dir / j / "accuracy_rashomon_heatmap.png"
    fig.savefig(plot_path)
    plt.close(fig)
