import json
import sys

import numpy as np
from sklearn.metrics import accuracy_score

from praxis_rashomon_common import (
    BASEDIR,
    RESULTS_DIR as PRAXIS_RESULTS_DIR,
    discover_param_tags as praxis_discover_param_tags,
    load_sampled_trees as praxis_load_sampled_trees,
    load_test_data as praxis_load_test_data,
    load_train_data as praxis_load_train_data,
)
from arborenum_rashomon_common import (
    RESULTS_DIRS as ARBORENUM_RESULTS_DIRS,
    discover_all_datasets as arborenum_discover_all_datasets,
    discover_param_tags as arborenum_discover_param_tags,
    load_sampled_trees as arborenum_load_sampled_trees,
    load_test_data as arborenum_load_test_data,
    load_train_data as arborenum_load_train_data,
)
from praxis_unbiased_binning_rashomon_common import (
    RESULTS_DIR as UNBIASED_BINNING_RESULTS_DIR,
    discover_all_datasets as unbiased_binning_discover_all_datasets,
    discover_param_tags as unbiased_binning_discover_param_tags,
    load_sampled_trees as unbiased_binning_load_sampled_trees,
    load_test_data as unbiased_binning_load_test_data,
    load_train_data as unbiased_binning_load_train_data,
)

sys.path.insert(0, str(BASEDIR / "rashomon-framework"))
from module.utils import generate_stability_distribution  # noqa: E402
from module.metric.stability_metric import StabilityMetric  # noqa: E402

MAX_TREE_SAMPLE = 1000  # cap on how many cached sampled trees to evaluate per param_tag (matches fairness/robustness)
NUM_TRIALS = 10  # passed through as StabilityMetric's num_trials (its own default is 5000)
NOISE_MEAN = 0.9  # matches experiment.py's stability defaults (noise_mean/noise_std)
NOISE_STD = 0.1
SEED = 42
SMALL_DATASETS = [
    "breast_cancer", "spambase", "compas", "heloc_original", "bike",
    "german_credit", "german_credit_dropna", "diabetes_smote",
]

# Each backend supplies its own dataset discovery / tree loading and where to write
# "<prefix>_stability_summary.json". PRAXIS's dataset list stays the fixed SMALL_DATASETS
# (its TGB param_tags are per-dataset feature-importance runs, not auto-enumerable the same
# way); ArborEnum's and the unbiased-binning PRAXIS backend's dataset names/configs differ
# (e.g. german_credit_dropna, no "heloc_original" alias) so their dataset lists are whatever
# each *_rashomon_common module actually finds cached.
BACKENDS = [
    {
        "label": "PRAXIS",
        "prefix": "praxis",
        "datasets": SMALL_DATASETS,
        "discover_param_tags": praxis_discover_param_tags,
        "load_train_data": praxis_load_train_data,
        "load_test_data": praxis_load_test_data,
        "load_sampled_trees": praxis_load_sampled_trees,
        "output_dir": lambda dataset_name, param_tag: PRAXIS_RESULTS_DIR / dataset_name / param_tag,
    },
    {
        "label": "ArborEnum",
        "prefix": "arborenum",
        "datasets": arborenum_discover_all_datasets(),
        "discover_param_tags": arborenum_discover_param_tags,
        "load_train_data": arborenum_load_train_data,
        "load_test_data": arborenum_load_test_data,
        "load_sampled_trees": arborenum_load_sampled_trees,
        "output_dir": lambda dataset_name, param_tag: ARBORENUM_RESULTS_DIRS[param_tag] / dataset_name,
    },
    {
        "label": "PRAXIS-UnbiasedBinning",
        "prefix": "unbiased_binning",
        "datasets": unbiased_binning_discover_all_datasets(),
        "discover_param_tags": unbiased_binning_discover_param_tags,
        "load_train_data": unbiased_binning_load_train_data,
        "load_test_data": unbiased_binning_load_test_data,
        "load_sampled_trees": unbiased_binning_load_sampled_trees,
        "output_dir": lambda dataset_name, param_tag: UNBIASED_BINNING_RESULTS_DIR / dataset_name,
    },
]


class _DummySplit:
    """Minimal stand-in for module.datasets.Split.

    StabilityMetric.setup() only reads X_train/X_test/y_train/y_test, checks
    for a `.binarizer` attribute (absent here, since our data is already the
    TGB-guessed binary representation -- this makes it fall back to
    `.preprocess()`, which is just the identity here).
    """

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.binarizer = None

    def preprocess(self, X):
        return X


class _DummyHparams:
    """Minimal stand-in for module.hparams.Hparams.

    StabilityMetric only reads `.rs` (for seeding the trial RNGs) and
    `.noise_dist_qf` (precomputed here, bypassing experiment.py's late-binding
    -1 sentinel since we're not running a live training loop).
    """

    def __init__(self, rs, noise_dist_qf):
        self.rs = rs
        self.noise_dist_qf = noise_dist_qf


def stability_for_param_tag(trees, X_train, y_train, X_test, y_test, n_features):
    rng = np.random.default_rng(SEED)
    q_f = generate_stability_distribution(rng, mean=NOISE_MEAN, std=NOISE_STD, size=n_features)

    # StabilityMetric._sample_noise_data does an in-place `X[:, f] += perturbation`
    # where perturbation can be negative -- needs a signed dtype, unlike our
    # uint8 guessed features (which is fine for TreeClassifier.predict itself).
    split_data = _DummySplit(X_train.astype(np.int64), y_train, X_test.astype(np.int64), y_test)
    hparams = _DummyHparams(rs=SEED, noise_dist_qf=q_f)

    metric = StabilityMetric()
    metric.setup(None, hparams, split_data, num_trials=NUM_TRIALS)

    rows = []
    for idx, tree in trees.items():
        pred_train = tree.predict(X_train)
        pred_test = tree.predict(X_test)
        predictions = {"pred_fn": tree.predict, "train": pred_train, "test": pred_test, "model": tree}
        result = metric.compute(predictions, split_data, num_trials=NUM_TRIALS)
        rows.append({
            "tree_index": int(idx),
            "n_leaves": tree.leaves(),
            "accuracy": float(accuracy_score(y_test, pred_test)),
            **result,
        })

    metric.cleanup()
    return rows


def run_backend(backend, dataset_names):
    for dataset_name in dataset_names:
        print(f"\n{'='*60}\n  [{backend['label']}] Dataset: {dataset_name}\n{'='*60}")
        param_tags = backend["discover_param_tags"](dataset_name)
        if not param_tags:
            print(f"  [SKIP] No cached {backend['label']} Rashomon sets found for {dataset_name}")
            continue

        for param_tag in param_tags:
            out_dir = backend["output_dir"](dataset_name, param_tag)
            out_path = out_dir / f"{backend['prefix']}_stability_summary.json"
            if out_path.exists():
                print(f"  [SKIP] {dataset_name}/{param_tag}: {out_path.name} already exists")
                continue

            print(f"  --- {param_tag} ---")
            X_train, y_train, _ = backend["load_train_data"](dataset_name, param_tag)
            X_test, y_test, _ = backend["load_test_data"](dataset_name, param_tag)
            trees, meta = backend["load_sampled_trees"](
                dataset_name, param_tag, n_features=X_test.shape[1], max_trees=MAX_TREE_SAMPLE
            )

            rows = stability_for_param_tag(trees, X_train, y_train, X_test, y_test, X_test.shape[1])

            most_stable = max(rows, key=lambda r: r["test_stability_acc_mean"])
            result = {
                "num_trials": NUM_TRIALS,
                "noise_mean": NOISE_MEAN,
                "noise_std": NOISE_STD,
                "n_total_trees": meta["n_total_trees"],
                "n_trees_evaluated": len(rows),
                "trees": rows,
                "most_stable_tree": most_stable,
            }

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"    Evaluated {len(rows)} trees; most stable: tree {most_stable['tree_index']} "
                  f"(test_stability_acc_mean={most_stable['test_stability_acc_mean']:.4f}, "
                  f"clean_acc={most_stable['accuracy']:.4f})")


def main():
    cli_datasets = sys.argv[1:] or None
    for backend in BACKENDS:
        run_backend(backend, cli_datasets if cli_datasets is not None else backend["datasets"])


if __name__ == "__main__":
    main()
