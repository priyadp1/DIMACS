# DIMACS

Comparison of interpretable tree-based classifiers — **SPLIT**, **LicketySPLIT**, **RESPLIT**, **TREEFARMS**, **LicketyRESPLIT**, **GOSDT**, **XGBoost**, **LightGBM**, and **CatBoost** — across multiple tabular datasets, with and without ThresholdGuessBinarizer (TGB) preprocessing.

---

## Models

| Model | Script | Description |
|---|---|---|
| XGBoost | `run_xgboost.py` | Gradient-boosted ensemble (max_depth=3, n_estimators=25) |
| GOSDT | `run_gosdt.py` | GOSDT with ThresholdGuessBinarizer preprocessing |
| LicketyRESPLIT | `run_licketyRESPLIT.py` | Rashomon-set decision tree on raw numeric features |
| LicketyRESPLIT (no binarizer, given splits) | `run_licketyRESPLIT_given_no_binarizer.py` | LicketyRESPLIT on pre-split data without binarization |
| SPLIT | `run_split.py` | Single optimal decision tree with internal binarization (`binarize=True`) |
| RESPLIT | `run_resplit.py` | Rashomon-set decision tree using CART lookahead; fills the set via TREEFARMS |
| TREEFARMS | `run_treefarms.py` | Rashomon-set model from `resplit.model.treefarms` |

**ThresholdGuessBinarizer** (from the `gosdt` package) fits a GradientBoosting model internally to find optimal split thresholds, then replaces each continuous feature with binary columns of the form `feature <= threshold`. Both GOSDT and the binarized LicketyRESPLIT variant receive this binary representation; the plain LicketyRESPLIT receives raw numeric features.

**SPLIT / LicketySPLIT** use `binarize=True`, so they handle binarization internally — raw numeric features are passed directly and the model computes its own thresholds during training. Parameters: `lookahead_depth_budget=2`, `full_depth_budget=5`, `reg=0.01`.

**Rashomon set**: LicketyRESPLIT, RESPLIT, and TREEFARMS return all decision trees whose training objective is within `rashomon_mult * 100`% of the optimal. Ensemble accuracy is the majority vote over this set. RESPLIT uses CART lookahead (`cart_lookahead_depth=3`) with `fill_tree='treefarms'`. TREEFARMS runs independently with `depth_budget=3`, `reg=0.01`, `rashomon_bound_multiplier=0.01`.

---

## Datasets

All datasets are in `datasets/Mine/`.

| Dataset | Target | Notes |
|---|---|---|
| `spambase.csv` | `class` | Binary spam classification |
| `breast_cancer_data.csv` | `diagnosis` | M/B mapped to 1/0 |
| `heloc_original.csv` | `RiskPerformance` | Credit risk |
| `bike.csv` | `cnt_binary` | Bike share demand (binarized count) |
| `compas.csv` | `two_year_recid` | Recidivism prediction |
| `leukemia_data.csv` | `label` | ALL/AML mapped to 0/1; used for cross-validation |
| `diabetic_data.csv` | `readmitted` | Diabetes readmission (>30/<30 → 1, NO → 0) |
| `diabetes_smote.csv` | `readmitted` | SMOTE-resampled diabetes training set |
| `creditcard_fraud_detection.csv` | `Class` | Credit card fraud (highly imbalanced) |
| `creditcard_fraud_detection_smote.csv` | `Class` | SMOTE-resampled fraud training set |
| `creditcard_fraud_detection_test.csv` | `Class` | Scaled test split for fraud evaluation |

---

## Running Experiments

### Single dataset (all models)

Edit the active dataset in `run_all.py` (uncomment the desired entry in `DATASETS`), then run:

```bash
python Code_files/run_all.py
```

Results are saved to `model_results/<dataset_name>/`.

### Benchmark suite — with TGB binarization

Runs all models (GOSDT, SPLIT, LicketySPLIT, LicketyRESPLIT, XGBoost, LightGBM, CatBoost) on pre-computed TGB outputs. Reads from `TGB_Variables/<dataset_name>/<param_tag>/`.

```bash
python Code_files/run_from_TGB.py
```

Results are saved to `benchmarks_TGB_results/<dataset_name>/<param_tag>/`. Saves ensemble dump files for visualization (see `boosting_visualizer.py`).

### Benchmark suite — without TGB binarization

Runs the same model set on raw (non-binarized) features. GOSDT and RESPLIT are invoked as subprocesses to avoid pybind11 conflicts with SPLIT.

```bash
python Code_files/run_no_TGB.py
```

Results are saved to `benchmarks_no_TGB_results/<dataset_name>/`. Intermediate train/test splits are cached in `tmp_splits_no_tgb/<dataset_name>/`.

### Parameter sweep (all datasets)

```bash
python Code_files/run_parameter_sweep.py
```

Sweeps the following grids and saves each config to its own subdirectory:

- **LicketyRESPLIT**: depth ∈ {3, 5}, λ ∈ {0.01, 0.05}, ε ∈ {0.01, 0.05}
- **GOSDT**: depth ∈ {3, 5}, regularization ∈ {0.001, 0.01}
- **XGBoost**: max_depth ∈ {3, 5}, n_estimators ∈ {25, 50}

### LicketyRESPLIT + TGB sweep

Sweeps LicketyRESPLIT with ThresholdGuessBinarizer across a finer parameter grid on spambase, bike, compas, and breast_cancer.

```bash
python Code_files/licketyRESPLIT_TGB.py
```

Grid: depth ∈ {2, 3, 4}, λ ∈ {0.001, 0.003, 0.01}, ε ∈ {0.01, 0.05, 0.1}. Results saved to `LicketyRESPLIT_EXP_ThresholdBinarizer/<dataset_name>/<param_str>/`.

### ThresholdGuessBinarizer preprocessing (all datasets)

```bash
python Code_files/run_TGB.py
```

Runs a parameter sweep of ThresholdGuessBinarizer + GBDT warm-label generation across all datasets. Grid: n_estimators ∈ {40, 100, 200}, max_depth ∈ {1, 2, 3}. Outputs saved to `TGB_Variables_Feature_Importance/<dataset_name>/<nest{N}_depth{D}>/`:
- `X_train_guessed.csv`, `X_test_guessed.csv` — binarized features
- `warm_labels.csv` — GBDT predictions on training set (used as warm start for GOSDT)
- `y_train.csv`, `y_test.csv` — train/test labels
- `binary_variable_counts.csv` — feature importance and split-count analysis
- `best_tgb_tree.png` — visualization of the highest-importance TGB tree
- `gbdt_warm_label_results.txt` — GBDT accuracy summary

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
| `TGB_figures.py` | Per-dataset accuracy plots comparing GOSDT, LicketyRESPLIT+TGB, and XGBoost from `model_results/` |
| `pacmap_gosdt_gbdt_plots.py` | PaCMAP embeddings of binarized feature spaces from `benchmarks_TGB_results/` |
| `pacmap_tgb_plots.py` | PaCMAP embeddings comparing TGB vs. raw feature spaces across datasets and TGB param settings |
| `plot_roc_curves.py` | ROC curves for GBDT warm-label models across TGB parameter combinations |

Most output files are saved to `LicketyRESPLIT_plots/` with prefixes `cross_`, `sweep_`, `noBin_`, `bin_`. PaCMAP plots go to `pacmap_embeddings/`.

---

## Utility Scripts

| Script | Description |
|---|---|
| `EDA.py` | Exploratory data analysis across datasets |
| `SMOTE.py` | SMOTE oversampling for `creditcard_fraud_detection.csv`; saves resampled train set and scaled test set to `datasets/Mine/` |
| `boosting_visualizer.py` | Renders individual trees from XGBoost (`.txt`), LightGBM (`.txt`), and CatBoost (`.json`) ensemble dump files to PNG; outputs go to `boosting_tree_visualizer/` |
| `clean_bike_binarized.py` | Sanitizes column names in `bike_binarized.csv` for XGBoost compatibility (replaces `<=`, `[`, `]`) |
| `download_dataset.py` | Downloads datasets via `kagglehub` and converts R data files via `pyreadr` |
| `run_gosdt_no_tgb.py` | Subprocess helper: trains GOSDT on raw features; called by `run_no_TGB.py` to avoid pybind11 conflicts |
| `run_resplit_no_tgb.py` | Subprocess helper: trains RESPLIT on raw features; called by `run_no_TGB.py` |

---

## Output File Structure

```
DIMACS/
│
├── Code_files/                                  ← experiment & utility scripts
│   ├── run_all.py
│   ├── run_from_TGB.py                          ← benchmark suite (TGB path)
│   ├── run_no_TGB.py                            ← benchmark suite (raw-data path)
│   ├── run_gosdt.py
│   ├── run_gosdt_no_tgb.py                      ← subprocess helper for GOSDT
│   ├── run_xgboost.py
│   ├── run_licketyRESPLIT.py
│   ├── run_licketyRESPLIT_given_no_binarizer.py
│   ├── licketyRESPLIT_TGB.py                    ← LicketyRESPLIT+TGB sweep
│   ├── run_split.py
│   ├── run_resplit.py
│   ├── run_resplit_no_tgb.py                    ← subprocess helper for RESPLIT
│   ├── run_treefarms.py
│   ├── run_TGB.py
│   ├── run_parameter_sweep.py
│   ├── run_cross_validation.py
│   ├── analyze_all.py
│   ├── boosting_visualizer.py
│   ├── SMOTE.py
│   ├── EDA.py
│   ├── clean_bike_binarized.py
│   ├── download_dataset.py
│   └── _run_config.json
│
├── datasets/
│   ├── Mine/                                    ← main experiment datasets
│   │   ├── bike.csv
│   │   ├── breast_cancer_data.csv
│   │   ├── compas.csv
│   │   ├── heloc_original.csv
│   │   ├── leukemia_data.csv
│   │   ├── spambase.csv
│   │   ├── diabetic_data.csv
│   │   ├── diabetes_smote.csv
│   │   ├── creditcard_fraud_detection.csv
│   │   ├── creditcard_fraud_detection_smote.csv
│   │   └── creditcard_fraud_detection_test.csv
│   └── Given/                                   ← pre-provided / reference datasets
│       ├── bike_binarized.csv
│       ├── bike_binarized_new.csv
│       └── broward_general_2y.csv
│
├── plot_files/                                  ← plot scripts
│   ├── cross_dataset_figures.py
│   ├── parameter_sweep_figures.py
│   ├── LicketyRESPLIT_noBin_figures.py
│   ├── LicketyRESPLIT_Bin_figures.py
│   ├── TGB_figures.py
│   ├── pacmap_gosdt_gbdt_plots.py
│   ├── pacmap_tgb_plots.py
│   └── plot_roc_curves.py
│
├── gosdt-guesses/                               ← GOSDT package source
│
├── SPLIT-ICML/                                  ← SPLIT / RESPLIT package source
│   ├── split/
│   └── resplit/
│
├── LicketyRESPLIT/                              ← LicketyRESPLIT package source
│
├── model_results/                               ← per-dataset model outputs (run_all.py)
│   ├── <dataset_name>/
│   │   ├── xgboost_results.txt
│   │   ├── gosdt_results.txt
│   │   ├── licketyresplit_results.txt
│   │   ├── licketyresplit_binarized_results.txt
│   │   ├── split_results.txt
│   │   └── <depth>_<lambda>_<rashomon>/         ← parameter sweep subdirectories
│   └── leukemia_data/
│       └── fold_1/ ... fold_5/
│
├── benchmarks_TGB_results/                      ← benchmark outputs (run_from_TGB.py)
│   └── <dataset_name>/
│       └── <nest{N}_depth{D}>/
│           ├── gosdt_results.txt
│           ├── split_results.txt
│           ├── licketysplit_results.txt
│           ├── licketyresplit_binarized_results.txt
│           ├── xgboost_binarized_results.txt
│           ├── lightgbm_results.txt
│           ├── catboost_results.txt
│           ├── xgboost_ensemble.txt
│           ├── lightgbm_ensemble.txt
│           └── catboost_ensemble.json
│
├── benchmarks_no_TGB_results/                   ← benchmark outputs (run_no_TGB.py)
│   └── <dataset_name>/
│       ├── gosdt_results.txt
│       ├── split_results.txt
│       ├── licketysplit_results.txt
│       ├── licketyresplit_binarized_results.txt
│       ├── xgboost_binarized_results.txt
│       ├── lightgbm_results.txt
│       ├── catboost_results.txt
│       ├── xgboost_ensemble.txt
│       ├── lightgbm_ensemble.txt
│       └── catboost_ensemble.json
│
├── tmp_splits_no_tgb/                           ← cached train/test splits (run_no_TGB.py)
│   └── <dataset_name>/
│       ├── X_train.csv, X_test.csv
│       └── y_train.csv, y_test.csv
│
├── TGB_Variables_Feature_Importance/            ← TGB sweep outputs (run_TGB.py)
│   └── <dataset_name>/
│       └── <nest{N}_depth{D}>/
│           ├── X_train_guessed.csv
│           ├── X_test_guessed.csv
│           ├── warm_labels.csv
│           ├── y_train.csv, y_test.csv
│           ├── binary_variable_counts.csv
│           ├── best_tgb_tree.png
│           └── gbdt_warm_label_results.txt
│
├── LicketyRESPLIT_EXP/                          ← LicketyRESPLIT sweep (no binarizer)
│   └── <depth>_<lambda>_<rashomon>/
│       ├── <dataset>_results.txt
│       └── <dataset>_tree_size.json
│
├── LicketyRESPLIT_EXP_ThresholdBinarizer/       ← LicketyRESPLIT+TGB sweep
│   └── <dataset_name>/
│       └── <param_str>/
│           ├── licketyresplit_binarized_results.txt
│           └── licketyresplit_binarized_tree_size.json
│
├── boosting_tree_visualizer/                    ← rendered boosting tree images
│   ├── TGB/
│   │   └── <dataset_name>/<param_tag>/
│   │       ├── xgboost_tree_<i>.png
│   │       ├── lightgbm_tree_<i>.png
│   │       └── catboost_tree_<i>.png
│   └── no_TGB/
│       └── <dataset_name>/
│
├── pacmap_embeddings/                           ← PaCMAP projection plots
│
├── LicketyRESPLIT_plots/                        ← generated comparison plots
│   └── compare_*.png
│
└── analysis_figures/                            ← generated analysis figures
    ├── <dataset_name>/
    │   ├── cross_model_best_accuracy.png
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
- `split` (provides `SPLIT` and `LicketySPLIT`)
- `resplit` (provides `RESPLIT` and `resplit.model.treefarms.TREEFARMS`)
- `gosdt` (provides `ThresholdGuessBinarizer` and `GOSDTClassifier`)
- `xgboost`
- `lightgbm`
- `catboost`
- `scikit-learn`
- `imbalanced-learn` (for `SMOTE`)
- `pacmap`
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `kagglehub`, `pyreadr` (for dataset downloading)
