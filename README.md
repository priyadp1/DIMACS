# DIMACS

Comparison of interpretable tree-based classifiers — **SPLIT**, **RESPLIT**, **TREEFARMS**, **LicketyRESPLIT**, **Threshold Guessing (GOSDT)**, and **XGBoost** — across multiple tabular datasets.

---

## Models

| Model | Script | Description |
|---|---|---|
| XGBoost | `run_xgboost.py` | Gradient-boosted ensemble (max_depth=3, n_estimators=25) |
| GOSDT | `run_gosdt.py` | GOSDT with ThresholdGuessBinarizer preprocessing |
| LicketyRESPLIT | `run_licketyRESPLIT.py` | Rashomon-set decision tree on raw numeric features |
| LicketyRESPLIT + Binarizer | `run_licketyRESPLIT_given.py` | LicketyRESPLIT after ThresholdGuessBinarizer |
| LicketyRESPLIT (no binarizer, given splits) | `run_licketyRESPLIT_given_no_binarizer.py` | LicketyRESPLIT on pre-split data without binarization |
| SPLIT | `run_split.py` | Single optimal decision tree with internal binarization (`binarize=True`) |
| RESPLIT | `run_resplit.py` | Rashomon-set decision tree using CART lookahead; fills the set via TREEFARMS |
| TREEFARMS | `run_treefarms.py` | Rashomon-set model from `resplit.model.treefarms` |

**ThresholdGuessBinarizer** (from the `gosdt` package) fits a GradientBoosting model internally to find optimal split thresholds, then replaces each continuous feature with binary columns of the form `feature <= threshold`. Both GOSDT and the binarized LicketyRESPLIT variant receive this binary representation; the plain LicketyRESPLIT receives raw numeric features.

**SPLIT** uses `binarize=True`, so it handles binarization internally — raw numeric features are passed directly and the model computes its own thresholds during training. Parameters: `lookahead_depth_budget=2`, `full_depth_budget=5`, `reg=0.01`.

**Rashomon set**: LicketyRESPLIT, RESPLIT, and TREEFARMS return all decision trees whose training objective is within `rashomon_mult * 100`% of the optimal. Ensemble accuracy is the majority vote over this set. RESPLIT uses CART lookahead (`cart_lookahead_depth=3`) with `fill_tree='treefarms'`. TREEFARMS runs independently with `depth_budget=3`, `reg=0.01`, `rashomon_bound_multiplier=0.01`.

---

## Datasets

All datasets are in `datasets/Mine/`. All feature columns are numeric (no categorical encoding needed).

| Dataset | Target | Notes |
|---|---|---|
| `spambase.csv` | `class` | Binary spam classification |
| `breast_cancer_data.csv` | `diagnosis` | M/B mapped to 1/0 |
| `heloc_original.csv` | `RiskPerformance` | Credit risk |
| `bike.csv` | `cnt_binary` | Bike share demand (binarized count) |
| `compas.csv` | `two_year_recid` | Recidivism prediction |
| `leukemia_data.csv` | `label` | ALL/AML mapped to 0/1; used for cross-validation |

---

## Running Experiments

### Single dataset (all models)

Edit the active dataset in `run_all.py` (uncomment the desired entry in `DATASETS`), then run:

```bash
python Code_files/run_all.py
```

Results are saved to `model_results/<dataset_name>/`.

### Parameter sweep (all datasets)

```bash
python Code_files/run_parameter_sweep.py
```

Sweeps the following grids and saves each config to its own subdirectory:

- **LicketyRESPLIT**: depth ∈ {3, 5}, λ ∈ {0.01, 0.05}, ε ∈ {0.01, 0.05}
- **GOSDT**: depth ∈ {3, 5}, regularization ∈ {0.001, 0.01}
- **XGBoost**: max_depth ∈ {3, 5}, n_estimators ∈ {25, 50}

### ThresholdGuessBinarizer preprocessing (all datasets)

```bash
python Code_files/run_TGB.py
```

Runs ThresholdGuessBinarizer + GBDT warm-label generation on all datasets and saves the outputs to `TGB_Variables/<dataset_name>/`:
- `X_train_guessed.csv`, `X_test_guessed.csv` — binarized features
- `warm_labels.csv` — GBDT predictions on training set (used as warm start for GOSDT)
- `y_train.csv`, `y_test.csv` — train/test labels

### 5-fold cross-validation (leukemia dataset)

```bash
python Code_files/run_cross_validation.py
```

Runs stratified 5-fold CV on `leukemia_data.csv` (ALL=0, AML=1). Each fold's train/test split and config are saved under `model_results/leukemia_data/fold_<n>/`, then `run_all.py` is invoked for each fold.

---

## Analysis & Plots

### Full analysis

```bash
python Code_files/analyze_all.py
```

Scans `model_results/` for all result files and generates figures under `analysis_figures/`:

**Per-dataset** (one subfolder per dataset):
- Accuracy vs. tree size scatter
- Ensemble accuracy vs. complexity
- Parameter heatmaps (accuracy and Rashomon set size by λ and ε)
- Cross-model best accuracy bar chart

**Cross-dataset** (`analysis_figures/cross_dataset/`):
- `cross_dataset_accuracy_heatmap.png` — test accuracy for all models × datasets
- `cross_dataset_f1_heatmap.png` — macro F1 for all models × datasets
- `cross_dataset_accuracy_vs_complexity.png` — accuracy vs. number of leaves (log scale)
- `all_best_results.csv` — summary table

### Plot files

All plot scripts are in `plot_files/`:

| Script | Description |
|---|---|
| `cross_dataset_figures.py` | Per-metric bar charts with datasets on the x-axis; compares No Binarizer, ThresholdBinarizer, GOSDT, and XGBoost across all parameter settings |
| `parameter_sweep_figures.py` | Per-dataset bar charts sweeping parameter settings; No Binarizer vs ThresholdBinarizer bars with GOSDT/XGBoost as reference lines |
| `LicketyRESPLIT_noBin_figures.py` | Per-dataset bar charts for LicketyRESPLIT (no binarizer) only, sweeping parameter settings |
| `LicketyRESPLIT_Bin_figures.py` | Per-dataset bar charts for LicketyRESPLIT (ThresholdBinarizer) only, sweeping parameter settings |

Output files are saved to `LicketyRESPLIT_plots/` with prefixes `cross_`, `sweep_`, `noBin_`, `bin_`.

---

## Utility Scripts

| Script | Description |
|---|---|
| `EDA.py` | Exploratory data analysis across datasets |
| `clean_bike_binarized.py` | Sanitizes column names in `bike_binarized.csv` for XGBoost compatibility (replaces `<=`, `[`, `]`) |
| `download_dataset.py` | Downloads datasets via `kagglehub` and converts R data files via `pyreadr` |
| `run_thresholdguessing.py` | Standalone threshold guessing script (single dataset via `_run_config.json`) |

---

## Output File Structure

```
DIMACS/
│
├── Code_files/                              ← experiment & utility scripts
│   ├── run_all.py
│   ├── run_gosdt.py
│   ├── run_xgboost.py
│   ├── run_licketyRESPLIT.py
│   ├── run_licketyRESPLIT_given.py
│   ├── run_licketyRESPLIT_given_no_binarizer.py
│   ├── run_split.py
│   ├── run_resplit.py
│   ├── run_treefarms.py
│   ├── run_TGB.py
│   ├── run_parameter_sweep.py
│   ├── run_cross_validation.py
│   ├── analyze_all.py
│   ├── EDA.py
│   ├── clean_bike_binarized.py
│   ├── download_dataset.py
│   └── _run_config.json
│
├── datasets/
│   ├── Mine/                                ← main experiment datasets
│   │   ├── bike.csv
│   │   ├── breast_cancer_data.csv
│   │   ├── compas.csv
│   │   ├── heloc_original.csv
│   │   ├── leukemia_data.csv
│   │   └── spambase.csv
│   └── Given/                               ← pre-provided / reference datasets
│       ├── bike_binarized.csv
│       ├── bike_binarized_new.csv
│       ├── broward_general_2y.csv
│       └── ...
│
├── plot_files/                              ← plot scripts
│   ├── cross_dataset_figures.py
│   ├── parameter_sweep_figures.py
│   ├── LicketyRESPLIT_noBin_figures.py
│   └── LicketyRESPLIT_Bin_figures.py
│
├── gosdt-guesses/                           ← GOSDT package source
│
├── SPLIT-ICML/                              ← SPLIT / RESPLIT package source
│   ├── split/
│   └── resplit/
│
├── LicketyRESPLIT/                          ← LicketyRESPLIT package source
│
├── model_results/                           ← per-dataset model outputs
│   ├── <dataset_name>/
│   │   ├── xgboost_results.txt
│   │   ├── xgboost_tree_size.json
│   │   ├── gosdt_results.txt
│   │   ├── gosdt_tree_size.json
│   │   ├── licketyresplit_results.txt
│   │   ├── licketyresplit_tree_size.json
│   │   ├── licketyresplit_binarized_results.txt
│   │   ├── licketyresplit_binarized_tree_size.json
│   │   ├── split_results.txt
│   │   ├── split_tree_size.json
│   │   └── <depth>_<lambda>_<rashomon>/     ← parameter sweep subdirectories
│   │       ├── resplit_results.txt
│   │       ├── resplit_tree_size.json
│   │       ├── treefarms_results.txt
│   │       └── treefarms_tree_size.json
│   └── leukemia_data/
│       └── fold_1/ ... fold_5/
│           ├── train.csv
│           ├── test.csv
│           ├── config.json
│           └── <model>_results.txt
│
├── TGB_Variables/                           ← ThresholdGuessBinarizer outputs
│   └── <dataset_name>/
│       ├── X_train_guessed.csv
│       ├── X_test_guessed.csv
│       ├── warm_labels.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── LicketyRESPLIT_EXP/                      ← LicketyRESPLIT sweep (no binarizer)
│   └── <depth>_<lambda>_<rashomon>/
│       ├── <dataset>_results.txt
│       └── <dataset>_tree_size.json
│
├── LicketyRESPLIT_EXP_ThresholdBinarizer/   ← LicketyRESPLIT sweep (with binarizer)
│   └── <depth>_<lambda>_<rashomon>/
│       ├── <dataset>_results.txt
│       └── <dataset>_tree_size.json
│
├── LicketyRESPLIT_plots/                    ← generated comparison plots
│   ├── compare_accuracy.png
│   ├── compare_ensemble_accuracy.png
│   ├── compare_n_leaves.png
│   ├── compare_n_trees_in_set.png
│   └── compare_duration_sec.png
│
└── analysis_figures/                        ← generated analysis figures
    ├── <dataset_name>/
    │   ├── cross_model_best_accuracy.png
    │   ├── lr_accuracy_vs_complexity.png
    │   └── ...
    └── cross_dataset/
        ├── cross_dataset_accuracy_heatmap.png
        ├── cross_dataset_f1_heatmap.png
        ├── cross_dataset_accuracy_vs_complexity.png
        └── all_best_results.csv
```

---

## Dependencies

- `licketyresplit`
- `split`
- `resplit` (provides `RESPLIT` and `resplit.model.treefarms.TREEFARMS`)
- `gosdt` (provides `ThresholdGuessBinarizer` and `GOSDTClassifier`)
- `xgboost`
- `scikit-learn`
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `kagglehub`, `pyreadr` (for dataset downloading)
