# DIMACS

Comparison of interpretable tree-based classifiers — **LicketyRESPLIT**, **Threshold Guessing (GOSDT)**, and **XGBoost** — across multiple tabular datasets.

---

## Models

| Model | Script | Description |
|---|---|---|
| XGBoost | `run_xgboost.py` | Gradient-boosted ensemble (max_depth=3, n_estimators=25) |
| Threshold Guessing | `run_gosdt.py` | GOSDT with ThresholdGuessBinarizer preprocessing |
| LicketyRESPLIT | `run_licketyRESPLIT.py` | Rashomon-set decision tree on raw numeric features |
| LicketyRESPLIT + Binarizer | `run_licketyRESPLIT_given.py` | LicketyRESPLIT after ThresholdGuessBinarizer |

**ThresholdGuessBinarizer** (from the `gosdt` package) fits a GradientBoosting model internally to find optimal split thresholds, then replaces each continuous feature with binary columns of the form `feature <= threshold`. Both GOSDT and the binarized LicketyRESPLIT variant receive this binary representation; the plain LicketyRESPLIT receives raw numeric features.

**Rashomon set**: LicketyRESPLIT returns all decision trees whose training objective is within `rashomon_mult * 100`% of the optimal. Ensemble accuracy is the majority vote over this set.

---

## Datasets

All datasets are in `datasets/Mine/`. All feature columns are numeric (no categorical encoding needed).

| Dataset | Target | Notes |
|---|---|---|
| `spambase.csv` | `class` | Binary spam classification |
| `breast_cancer_data.csv` | `diagnosis` | M/B mapped to 1/0 |
| `heloc_original.csv` | `RiskPerformance` | Credit risk |
| `bike.csv` | `cnt_binary` | Bike share demand (binarized count) |
| `leukemia_data.csv` | `label` | ALL/AML mapped to 1/0; used for cross-validation only |

---

## Running Experiments

### Single dataset (all four models)

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

### 5-fold cross-validation (leukemia dataset)

```bash
python Code_files/run_cross_validation.py
```

Runs stratified 5-fold CV on `leukemia_data.csv` (ALL=1, AML=0). Each fold's train/test split and config are saved under `model_results/leukemia_data/fold_<n>/`, then `run_all.py` is invoked for each fold.

---

## Analysis & Plots

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

The cross-dataset plots use top-level result files only (not parameter sweep subdirectories) and include: `spambase`, `breast_cancer_data`, `bike`, `heloc_original`.

---

## Output File Structure

```
model_results/
  <dataset_name>/
    xgboost_results.txt
    xgboost_tree_size.json
    gosdt_results.txt
    gosdt_tree_size.json
    licketyresplit_results.txt
    licketyresplit_tree_size.json
    licketyresplit_binarized_results.txt
    licketyresplit_binarized_tree_size.json
    <depth>_<lambda>_<rashomon>/      # LicketyRESPLIT sweep subfolders
    gosdt_<depth>_<reg>/
    xgboost_<depth>_<n_est>/
  leukemia_data/
    fold_1/ ... fold_5/
      train.csv, test.csv, config.json, <model>_results.txt, ...

analysis_figures/
  <dataset_name>/
    cross_model_best_accuracy.png
    lr_accuracy_vs_complexity.png
    ...
  cross_dataset/
    cross_dataset_accuracy_heatmap.png
    cross_dataset_f1_heatmap.png
    cross_dataset_accuracy_vs_complexity.png
    all_best_results.csv
```

---

## Dependencies

- `licketyresplit`
- `gosdt` (provides `ThresholdGuessBinarizer` and `GOSDTClassifier`)
- `xgboost`
- `scikit-learn`
- `pandas`, `numpy`, `matplotlib`, `seaborn`
