# Training

This step runs a research-grade XGBoost training workflow — baselines, Optuna
hyperparameter tuning, train/val/test split, SHAP, confusion matrix, and final
stratified 10-fold cross-validation.

---

## Supported Modalities

Each modality's processed CSV trains to its own isolated output directory.
Results, models, figures, and metrics for each modality are never mixed.

| Modality | Input CSV | Output Directory |
|----------|-----------|-----------------|
| T1 | `outputs/features_processed_t1.csv` | `outputs/t1/` |
| T2 | `outputs/features_processed_t2.csv` | `outputs/t2/` |
| T1GD | `outputs/features_processed_t1gd.csv` | `outputs/t1gd/` |
| FLAIR | `outputs/features_processed_flair.csv` | `outputs/flair/` |

> **Prerequisite:** Run `preprocessing.py` for the chosen modality before training.
> See [preprocessing.md](preprocessing.md).

---

## CLI Usage

### T1
```bash
python src/train.py \
  --input  outputs/features_processed_t1.csv \
  --output outputs/t1
```

### T2
```bash
python src/train.py \
  --input  outputs/features_processed_t2.csv \
  --output outputs/t2
```

### T1GD
```bash
python src/train.py \
  --input  outputs/features_processed_t1gd.csv \
  --output outputs/t1gd
```

### FLAIR
```bash
python src/train.py \
  --input  outputs/features_processed_flair.csv \
  --output outputs/flair
```

### All 4 in sequence (PowerShell)
```powershell
foreach ($mod in @("t1","t2","t1gd","flair")) {
    python src/train.py `
        --input  "outputs/features_processed_$mod.csv" `
        --output "outputs/$mod"
}
```

### All 4 in sequence (bash)
```bash
for mod in t1 t2 t1gd flair; do
    python src/train.py \
        --input  "outputs/features_processed_${mod}.csv" \
        --output "outputs/${mod}"
done
```

> **Note:** `train.py` currently reads `--input` and output directory from
> `config.json → training.input_csv` and `training.output_dir`. To support
> per-modality runs without editing config each time, pass `--input` and
> `--output` directly on the CLI. *(If these flags are not yet wired in
> `train.py`, update `parse_args()` to accept them — see the note at the
> bottom of this page.)*

---

## All CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `config.json` | Path to config.json |
| `--input` | `training.input_csv` from config | Processed features CSV (modality-specific) |
| `--output` | `training.output_dir` from config | Root output directory for this run |

---

## Training Workflow (in order)

### 1. Dataset Loading
- Reads the processed CSV; validates all 16 feature columns + `risk_label` are present.
- Drops any rows where `risk_label` is not 0 or 1.
- Imputes remaining feature nulls with column medians.

### 2. Exploratory Plots
Saved to `{output}/figures/`:
- `class_distribution.png` — bar chart of class balance
- `feature_distributions.png` — 4×4 histograms per feature, coloured by class
- `correlation_heatmap.png` — Pearson correlation matrix heatmap

### 3. Train / Val / Test Split
- **70 / 15 / 15** stratified split.
- Split assignments saved to `{output}/tables/split_assignments.csv`.

### 4. Baseline Models
Evaluated on the validation split. Saved to `{output}/metrics/baseline_results.csv`.

| Model | Scaling |
|-------|---------|
| Logistic Regression | RobustScaler + VarianceThreshold |
| SVM (RBF) | RobustScaler + VarianceThreshold |
| KNN | RobustScaler + VarianceThreshold |
| Random Forest | None (tree model) |
| Gradient Boosting | None (tree model) |

### 5. Validation XGBoost
- Fixed hyperparameters; early stopping on validation AUC.
- Metrics → `{output}/metrics/validation_xgboost_metrics.json`.

### 6. Optuna Hyperparameter Tuning
- TPE sampler; stratified CV on train+val set.
- `n_trials` and `optuna_cv_folds` come from `config.json → training`.
- Trial log → `{output}/logs/optuna_trials.csv`.

### 7. Best Model — Fit & Evaluate
- Retrained on full train+val using best Optuna params.
- Evaluated on held-out test set.
- Metrics → `{output}/metrics/test_metrics.json`.
- Classification report → `{output}/metrics/test_classification_report.csv`.
- Confusion matrix → `{output}/figures/confusion_matrix.png`.

### 8. Feature Importance & SHAP
- XGBoost built-in importance → `{output}/figures/feature_importance.png`.
- SHAP summary beeswarm → `{output}/figures/shap_summary.png`.
- SHAP bar chart → `{output}/figures/shap_importance.png`.

### 9. Final Stratified 10-Fold CV
- Runs on the full dataset using best Optuna params.
- Reports mean AUC, 95% CI (`scipy.stats.t.interval`), and one-sided p-value vs AUC = 0.5.
- Fold metrics → `{output}/metrics/final_cv_fold_metrics.csv`.
- Summary → `{output}/metrics/final_cv_summary.json`.

---

## Inputs

| Item | Source |
|------|--------|
| Features CSV | `outputs/features_processed_{modality}.csv` |
| Target column | `risk_label` (0 = low-risk, 1 = high-risk) |
| Feature columns | 16 fixed radiomic features |
| Hyperparameter config | `config.json → training` |

---

## Full Output Tree (per modality)

```
outputs/{modality}/
├── figures/
│   ├── class_distribution.png
│   ├── feature_distributions.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── shap_summary.png
│   └── shap_importance.png
├── metrics/
│   ├── baseline_results.csv
│   ├── dataset_summary.json
│   ├── feature_summary.json
│   ├── validation_xgboost_metrics.json
│   ├── test_metrics.json
│   ├── test_classification_report.csv
│   ├── final_cv_fold_metrics.csv
│   └── final_cv_summary.json
├── models/
│   ├── best_xgb_risk_classifier.pkl
│   └── best_params.json
├── tables/
│   ├── split_assignments.csv
│   ├── feature_importance.csv
│   └── high_correlation_pairs.csv
├── logs/
│   ├── optuna_trials.csv
│   └── gpu_status.json
├── reports/
│   └── training_summary.md
├── final_model.json
├── feature_list.json
├── cv_results.json
└── training_report.txt
```

---

## `config.json` — Relevant Keys

```json
"training": {
  "input_csv":                         "outputs/features_processed.csv",
  "output_dir":                        "outputs",
  "test_size":                         0.15,
  "validation_fraction_of_trainval":   0.176470588,
  "optuna_trials":                     50,
  "optuna_cv_folds":                   5,
  "final_cv_folds":                    10,
  "xgboost_early_stopping_rounds":     25,
  "random_seed":                       42
}
```

> When passing `--input` and `--output` on the CLI, the `input_csv` and
> `output_dir` keys in config are **overridden** by the CLI values.

---

## Adding `--input` / `--output` to `train.py` (if not already present)

If `train.py`'s `parse_args()` does not yet expose `--input` and `--output`,
add these two arguments:

```python
parser.add_argument(
    "--input",
    type=Path,
    default=None,
    help="Processed features CSV. Overrides training.input_csv in config.",
)
parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Root output directory. Overrides training.output_dir in config.",
)
```

Then in `main()`, resolve with:
```python
input_csv  = args.input  or resolve_path(root, training_cfg.get("input_csv", "outputs/features_processed.csv"))
output_root = args.output or resolve_path(root, training_cfg.get("output_dir", "outputs"))
```
