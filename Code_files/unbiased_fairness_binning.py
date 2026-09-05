import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_BINS = 5
EPSILON = 0.05
MAX_QUADRATIC_N = 1500

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent
BASEDIR = current

sys.path.insert(0, str(BASEDIR / "UnbiasedBinning"))
from UnbiasedBinning import unbiased_binning
from EbiasedBinning import ebias_binning
from EbiasedDnC import ebias_dnc

DATA_DIR    = BASEDIR / "datasets" / "Mine"
RESULTS_DIR = BASEDIR / "benchmarks_unbiased_fairness_binning_raw_results"

DATASETS = {
    "compas": {
        "path": DATA_DIR / "compas.csv",
        "x_col": "age",
        "group_col": "sex=female",
        "group_map": None,
    },
    "german_credit": {
        "path": DATA_DIR / "german_credit_data.csv",
        "x_col": "Age",
        "group_col": "Sex",
        "group_map": {"male": 0, "female": 1},
    },
    "diabetes_smote": {
        "path": DATA_DIR / "diabetes_smote.csv",
        "x_col": "num_medications",
        "group_col": "race",
        "group_map": None,
    },
}


def load_dataset(cfg):
    df = pd.read_csv(cfg["path"])
    df = df[[cfg["x_col"], cfg["group_col"]]].dropna()

    if cfg["group_map"]:
        df[cfg["group_col"]] = df[cfg["group_col"]].map(cfg["group_map"])

    x = df[cfg["x_col"]].to_numpy()
    if not np.issubdtype(x.dtype, np.integer):
        x = (x * 100).astype(int)

    g = df[cfg["group_col"]].to_numpy().astype(int)
    ell = int(len(np.unique(g)))

    D = np.column_stack((x, g))
    return D, ell


def run_unbiased_binning(D, k):
    result = unbiased_binning(D, k=k, x=0, groups=1)
    if len(result) == 3 and result[0] == [-2]:
        return {"feasible": False, "n_boundary_candidates": int(result[2])}
    values, indices, m = result
    return {
        "feasible": True,
        "boundary_values": [float(v) for v in values],
        "boundary_indices": [int(i) for i in indices],
        "n_boundary_candidates": int(m),
    }


def run_ebias_binning(D, k, eps):
    values, indices = ebias_binning(D, k=k, x=0, groups=1, eps=eps)
    if values == [-2]:
        return {"feasible": False}
    return {
        "feasible": True,
        "boundary_values": [float(v) for v in values],
        "boundary_indices": [int(i) for i in indices],
    }


def run_ebias_dnc(D, k, eps):
    boundaries, obj = ebias_dnc(D, k=k, x=0, groups=1, eps=eps)
    if boundaries == [-1]:
        return {"feasible": False}
    return {
        "feasible": True,
        "boundaries": [int(b) for b in boundaries],
        "objective": int(obj),
    }


def canonical_boundaries(res, n, style):
    """Normalize every algorithm's output to a length-(k+1) list of prefix
    cut points [0, ..., n], so bins can be read as sorted_data[b[t]:b[t+1]]."""
    if not res["feasible"]:
        return None
    if style == "cut_indices":  # unbiased_binning / ebias_binning
        idx = res["boundary_indices"]
        return [0] + idx[:-1] + [n]
    return res["boundaries"]  # ebias_dnc already returns the full list


def bin_group_ratios(sorted_groups, boundaries, ell):
    bins = []
    for t in range(len(boundaries) - 1):
        seg = sorted_groups[boundaries[t]:boundaries[t + 1]]
        size = int(len(seg))
        counts = [int(np.sum(seg == gval)) for gval in range(ell)]
        props = [c / size if size else 0.0 for c in counts]
        bins.append({
            "start": int(boundaries[t]),
            "end": int(boundaries[t + 1]),
            "size": size,
            "group_counts": counts,
            "group_proportions": props,
        })
    return bins


def max_disparity(bins, overall_props):
    max_dev = 0.0
    for b in bins:
        for gi, p in enumerate(b["group_proportions"]):
            max_dev = max(max_dev, abs(p - overall_props[gi]))
    return max_dev


def run_dataset(dataset_name, cfg):
    out_dir = RESULTS_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    _expected_outputs = [
        "binning_results.txt",
        "binning_details.json",
    ]
    if all((out_dir / f).exists() for f in _expected_outputs):
        print(f"  [BINNING] Skipping {dataset_name} — results already exist.")
        return

    D, ell = load_dataset(cfg)
    n = len(D)
    sorted_data = D[D[:, 0].argsort()]
    sorted_groups = sorted_data[:, 1]
    overall_props = [float(np.mean(sorted_groups == gval)) for gval in range(ell)]

    print(f"  [BINNING] Running on {dataset_name} (n={n}, groups={ell}, "
          f"x='{cfg['x_col']}', group='{cfg['group_col']}')...")

    algo_results = {}

    print(f"  [BINNING] unbiased_binning (exact, zero-bias)...")
    start = time.perf_counter()
    res = run_unbiased_binning(D, k=N_BINS)
    res["duration_sec"] = time.perf_counter() - start
    algo_results["unbiased_binning"] = res

    print(f"  [BINNING] ebias_binning (exact DP, eps={EPSILON})...")
    D_dp = D
    subsampled = n > MAX_QUADRATIC_N
    if subsampled:
        rng = np.random.default_rng(RANDOM_SEED)
        sample_idx = rng.choice(n, size=MAX_QUADRATIC_N, replace=False)
        D_dp = D[sample_idx]
    start = time.perf_counter()
    res = run_ebias_binning(D_dp, k=N_BINS, eps=EPSILON)
    res["duration_sec"] = time.perf_counter() - start
    res["subsampled"] = subsampled
    res["n_used"] = int(len(D_dp))
    algo_results["ebias_binning"] = res

    print(f"  [BINNING] ebias_dnc (local search, eps={EPSILON})...")
    start = time.perf_counter()
    res = run_ebias_dnc(D, k=N_BINS, eps=EPSILON)
    res["duration_sec"] = time.perf_counter() - start
    algo_results["ebias_dnc"] = res

    styles = {
        "unbiased_binning": "cut_indices",
        "ebias_binning": "cut_indices",
        "ebias_dnc": "boundaries",
    }
    for algo_name, res in algo_results.items():
        n_for_bins = res["n_used"] if algo_name == "ebias_binning" else n
        data_for_bins = D_dp[D_dp[:, 0].argsort()][:, 1] if algo_name == "ebias_binning" else sorted_groups
        bounds = canonical_boundaries(res, n_for_bins, styles[algo_name])
        if bounds is None:
            continue
        res["bin_group_ratios"] = bin_group_ratios(data_for_bins, bounds, ell)
        res["max_group_proportion_deviation"] = max_disparity(res["bin_group_ratios"], overall_props)

    with open(out_dir / "binning_details.json", "w") as f:
        json.dump({
            "dataset": dataset_name,
            "n": n,
            "n_groups": ell,
            "k_bins": N_BINS,
            "epsilon": EPSILON,
            "x_col": cfg["x_col"],
            "group_col": cfg["group_col"],
            "overall_group_proportions": overall_props,
            "algorithms": algo_results,
        }, f, indent=2)

    with open(out_dir / "binning_results.txt", "w") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"n = {n}, k = {N_BINS} bins, eps = {EPSILON}\n")
        f.write(f"Attribute binned: {cfg['x_col']}\n")
        f.write(f"Group attribute: {cfg['group_col']} ({ell} groups)\n")
        f.write(f"Overall group proportions: {[round(p, 4) for p in overall_props]}\n\n")

        for algo_name, res in algo_results.items():
            f.write(f"--- {algo_name} ---\n")
            f.write(f"  Runtime: {res['duration_sec']:.4f}s\n")
            if not res["feasible"]:
                f.write("  Infeasible: no valid binning satisfying the constraint was found.\n\n")
                continue
            if "objective" in res:
                f.write(f"  Objective (bin-size disparity): {res['objective']}\n")
            if algo_name == "ebias_binning" and res.get("subsampled"):
                f.write(f"  Subsampled to n={res['n_used']} (exact DP is quadratic in n)\n")
            f.write(f"  Max group-proportion deviation across bins: "
                    f"{res['max_group_proportion_deviation']:.4f}\n")
            for b in res["bin_group_ratios"]:
                props = [round(p, 3) for p in b["group_proportions"]]
                f.write(f"    bin[{b['start']}:{b['end']}) size={b['size']} proportions={props}\n")
            f.write("\n")

    print(f"  [BINNING] {dataset_name} done.")


if __name__ == "__main__":
    for dataset_name, cfg in DATASETS.items():
        print(f"\n{'='*60}\nDataset: {dataset_name}\n{'='*60}")
        try:
            run_dataset(dataset_name, cfg)
        except Exception as e:
            print(f"  [BINNING] {dataset_name} failed: {e}")
