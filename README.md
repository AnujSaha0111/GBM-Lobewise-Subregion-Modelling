# Multimodal Lobewise Subregion Modelling for Glioblastoma Survival Risk Analysis

An interpretable machine-learning framework for characterising spatial tumour occupancy patterns across the four lobes of the brain and their association with molecular markers and survival outcomes in glioblastoma.

---

## Overview

Glioblastoma (GBM) is the most aggressive primary brain tumour, with highly variable survival that current clinical and molecular markers incompletely explain. This repository implements a complete radiogenomic pipeline that extracts anatomically referenced spatial occupancy features from routine MRI, characterises their association with molecular markers (MGMT promoter methylation, IDH mutation), and builds complementary prognostic models for survival risk stratification.

The core approach is:

1. **Spatial feature extraction** -- Register a 4-lobe atlas (SRI24) into each patient's native MRI space and compute 16 features per modality: the fraction of enhancing tumour (EN), non-enhancing core (NC), and peritumoural oedema (ED) within each lobe, plus four global tumour-burden ratios.

2. **Multimodal fusion** -- Concatenate features across four MRI sequences (T1, T2, T1GD, FLAIR) to form a 64-dimensional spatial representation per patient.

3. **Radiogenomic analysis** -- Test which spatial features associate with molecular markers using Mann-Whitney U tests with FDR correction, and build multivariable logistic regression predictors for MGMT and IDH status.

4. **Complementary prognostic modelling** -- Evaluate whether spatial features add independent prognostic value over clinical-molecular covariates using nested Cox proportional hazards models, likelihood ratio tests, and incremental discrimination metrics (NRI, IDI).

5. **Explainability** -- Use SHAP (SHapley Additive exPlanations) and permutation importance to identify which anatomical lobe-subregion combinations carry the strongest survival signal.

6. **Robustness validation** -- Assess proportional hazards assumptions, feature selection stability, prediction reliability across nested cross-validation, and spatial representation dimensionality.

The framework uses the UCSF-PDGM dataset and an SRI24 atlas-based parcellation. It is designed for reproducibility: all pipelines use fixed random seeds, deterministic preprocessing, and bootstrap confidence intervals.

---

## Key Features

- **MRI preprocessing and atlas registration** -- SRI24 atlas registration with ANTs or affine fallback, 4-lobe parcellation with dilated labels, and 16-feature extraction per modality
- **Multimodal spatial occupancy features** -- 64-dimensional feature vectors combining spatial overlap measurements from T1, T2, T1GD, and FLAIR
- **Anatomically referenced measurements** -- Features report tumour subregion fractions within anatomically defined lobes, enabling interpretable spatial characterisation
- **Radiogenomic association analysis** -- Univariate (Mann-Whitney U, Cliff's delta, ROC-AUC) and multivariable (logistic regression) association testing with FDR correction
- **Cox proportional hazards modelling** -- Nested Cox models with likelihood ratio tests for quantifying independent prognostic value
- **Random Survival Forests** -- Non-linear survival modelling for comparison with Cox PH
- **Incremental prognostic evaluation** -- Net Reclassification Index (NRI) and Integrated Discrimination Improvement (IDI) at multiple time horizons
- **Explainability via SHAP** -- TreeExplainer-based and permutation-based feature importance, aggregated by modality, lobe, and subregion
- **Calibration analysis** -- Brier score, calibration curves, and Hosmer-Lemeshow testing across spatial, clinical, and combined feature sets
- **Decision curve analysis** -- Clinical utility assessment with bootstrap confidence intervals across threshold probabilities
- **Baseline classifier comparison** -- Logistic Regression, Random Forest, XGBoost, and SVM (RBF) across spatial, clinical-molecular, and combined feature sets
- **Bootstrap confidence intervals** -- 95% CIs for AUC, accuracy, precision, recall, F1, C-index, and NRI/IDI using 5,000 bootstrap replicates
- **Statistical validation** -- Pairwise DeLong AUC comparisons, Schoenfeld residual tests, stratified Cox, time-varying coefficients, and RMST
- **Reproducible pipelines** -- Fixed random seeds (42), deterministic preprocessing, checkpointed models, and self-contained experiment outputs

---

## Repository Structure

```
GBM-Lobewise-Subregion-Modelling/
├── src/                                  # Core pipeline source code
│   ├── multimodal_lobewise/              # Multimodal dataset construction, training, and analysis
│   │   ├── build_multimodal_dataset.py   # Merge 4 modality CSVs into a single multimodal table
│   │   ├── preprocess_multimodal.py      # Median imputation + StandardScaler (leakage-corrected)
│   │   ├── train_multimodal_model.py     # XGBoost classifier with early stopping
│   │   ├── train_svm_model.py            # Leakage-corrected SVM (RBF) with GridSearchCV
│   │   ├── analyze_feature_importance.py # SHAP + XGBoost gain importance by modality/lobe/subregion
│   │   ├── analyze_svm_results.py        # Permutation importance for SVM
│   │   ├── statistical_analysis.py       # Mann-Whitney U, FDR, Cohen's d, volcano plot
│   │   ├── correlation_analysis.py       # Pearson/Spearman matrices, redundancy quantification
│   │   ├── compute_auc_significance.py   # DeLong AUC comparison testing
│   │   ├── evaluate_model_stability.py   # Model stability evaluation
│   │   ├── load_metadata.py              # Metadata loading utilities
│   │   ├── plot_cv_roc_curves.py         # Cross-validation ROC curve plotting
│   │   ├── shap_stability_evaluation.py  # SHAP rank stability across folds
│   │   ├── svm_shap_analysis.py          # SHAP analysis for SVM models
│   │   └── svm_shap_stability.py         # SHAP stability for SVM models
│   ├── multimodal_lobewise_svm_comparison/ # Feature-set comparison experiments
│   │   ├── calibration_analysis.py       # Brier score, calibration curves across feature sets
│   │   ├── compare_feature_sets.py       # AUC comparison: spatial vs clinical vs combined
│   │   ├── dca_bootstrap_ci.py           # Decision curve analysis with bootstrap CIs
│   │   └── decision_curve_analysis.py    # Clinical utility assessment
│   ├── atlas_registration.py            # SRI24 atlas registration + 16-feature extraction
│   ├── preprocessing.py                 # Per-modality feature preprocessing
│   ├── train.py                         # XGBoost risk classifier training with Optuna tuning
│   ├── shap_analysis.py                 # SHAP analysis utilities
│   ├── survival_analysis.py             # Cox PH + Random Survival Forest survival analysis
│   ├── radiogenomic_analysis.py         # Spatial feature vs molecular marker associations
│   ├── statistical_comparison.py        # Pairwise DeLong AUC comparisons
│   ├── baseline_comparison.py           # LR/RF/XGB/SVM baseline comparisons
│   ├── nested_cox_analysis.py           # Nested Cox + LRT for incremental value
│   ├── nri_idi_analysis.py              # NRI/IDI reclassification analysis
│   ├── nri_sensitivity_analysis.py      # NRI/IDI sensitivity analysis
│   ├── compute_metric_bootstrap_cis.py  # Bootstrap 95% CIs for test metrics
│   ├── generate_anatomical_overlay.py   # 4-panel anatomical overlay figure
│   ├── find_representative_patient.py   # Identify representative patients
│   └── run_idh_wt_fix.py               # IDH-wildtype subgroup processing
│
├── experiments/                          # Research experiments by scientific purpose
│   ├── robustness/                      # Model robustness assessments
│   │   ├── feature_selection/            # Regularized feature selection (L1, ElasticNet, RFECV)
│   │   ├── proportional_hazards/         # PH assumption diagnostics + alternatives
│   │   ├── representation_ablation/      # Spatial feature dimensionality ablation
│   │   └── reproducibility/             # Nested repeated CV for model comparisons
│   ├── statistics/                       # Statistical validation
│   │   ├── bootstrap/                    # Bootstrap 95% CI for MGMT logistic regression AUC
│   │   └── cohort_audit/                # CONSORT-style patient exclusion audit
│   ├── radiogenomics/                    # Radiogenomic association experiments
│   │   ├── association_analysis/         # Feature-molecular association studies
│   │   └── molecular_prediction/        # MGMT/IDH prediction from spatial features
│   ├── prognosis/                        # Prognostic model experiments
│   │   ├── calibration/                  # Model calibration analysis
│   │   └── decision_curves/             # Decision curve analysis
│   └── explainability/                   # Model interpretability
│
├── outputs/                              # Generated data, models, and figures
│   ├── multimodal_lobewise/              # XGBoost model outputs and merged features
│   ├── multimodal_lobewise_svm/          # SVM model outputs and importance analyses
│   ├── multimodal_lobewise_svm_comparison/ # Feature-set comparison results
│   │   ├── calibration/                  # Calibration curves and metrics
│   │   └── dca/                          # Decision curve data and figures
│   ├── baselines/                        # Baseline classifier results (LR, RF, XGB, SVM)
│   ├── survival_analysis/                # Cox PH and RSF survival results
│   ├── survival_incremental_value/       # Nested Cox, NRI/IDI, and LRT results
│   ├── radiogenomics/                    # Radiogenomic association results
│   ├── confidence_intervals/             # Bootstrap confidence intervals
│   ├── statistical_comparisons/          # Pairwise DeLong AUC comparison results
│   ├── statistical_power/                # Minimum detectable effect calculations
│   ├── shap_stability/                   # SHAP rank stability across folds
│   ├── figures/                          # Generated plots and visualisations
│   ├── tables/                           # Generated CSV tables
│   ├── reports/                          # Generated Markdown reports
│   ├── metrics/                          # Model performance metrics (JSON, CSV)
│   ├── models/                           # Serialized trained models
│   ├── logs/                             # Optuna trial logs
│   └── features_raw.csv                  # Raw spatial features per modality
│
├── data/
│   └── SRI24/                            # SRI24 atlas parcellation data
│
├── config.json                           # Pipeline configuration
├── requirements.txt                      # Python dependencies
├── LICENSE                               # Apache License 2.0
└── UCSF-PDGM-metadata_v5.csv            # Patient clinical metadata
```

---

## Pipeline

The complete workflow proceeds through eleven stages:

### 1. Data Acquisition

Patient MRI volumes (T1, T2, T1GD, FLAIR) and tumour segmentation masks are sourced from the UCSF-PDGM dataset. Clinical metadata (age, sex, MGMT, IDH, WHO grade, extent of resection, overall survival) is stored in `UCSF-PDGM-metadata_v5.csv`.

### 2. MRI Preprocessing

Raw NIfTI volumes are loaded and standardised. Tumour segmentation labels are mapped to three subregions: enhancing tumour (EN, label 4), non-enhancing core (NC, label 1), and peritumoural oedema (ED, label 2).

### 3. Atlas Registration

The SRI24 atlas is parcellated into four lobes (frontal, temporal, parietal, occipital) by aggregating LPBA40 region labels. The lobe atlas is registered to each patient's native MRI space using ANTs affine registration (with fallback for Windows). Dilated lobe boundaries (3-voxel dilation) ensure overlap with tumour voxels.

### 4. Spatial Occupancy Feature Extraction

For each patient and modality, 16 features are computed: 12 lobar ratios (EN, NC, ED fraction within each of four lobes) and 4 global tumour-burden metrics (NC/EN ratio, ED/EN ratio, ED/total ratio, tumour burden index).

### 5. Feature Preprocessing

Per-modality raw features are filtered for lobe-assignment reliability and non-missing survival data. Missing values are imputed with column medians. Features are standard-scaled (zero mean, unit variance). The four modality CSVs are merged into a single 64-dimensional multimodal table via inner join on patient ID.

### 6. Radiogenomic Analysis

For each of the 64 spatial features, Mann-Whitney U tests compare distributions between molecular subgroups (MGMT methylated vs unmethylated, IDH mutant vs wildtype). Effect sizes (Cohen's d, Cliff's delta) are computed. Benjamini-Hochberg FDR correction controls for multiple testing. Multivariable logistic regression builds predictors for MGMT and IDH status.

### 7. Prognostic Modelling

Cox proportional hazards models are fit using three feature sets: spatial (64 features), clinical-molecular (6 features), and combined (70 features). Random Survival Forests provide a non-linear comparison. Nested Cox models with likelihood ratio tests assess whether spatial features add independent prognostic value over clinical covariates.

### 8. Explainability

SHAP (TreeExplainer for XGBoost, permutation-based for SVM) decomposes predictions into additive feature contributions. Contributions are aggregated by modality, lobe, and subregion to identify anatomical axes with the strongest survival signal.

### 9. Robustness Analysis

- **Proportional hazards diagnostics** -- Schoenfeld residual tests, stratified Cox, time-varying coefficients, and RMST
- **Feature selection stability** -- L1, ElasticNet, RFECV, and mutual information feature selection
- **Representation ablation** -- Reduced vs full 64-dimensional spatial representation
- **Nested cross-validation** -- Repeated stratified CV to confirm model comparison results

### 10. Statistical Validation

Bootstrap 95% confidence intervals (B = 5,000) for AUC, accuracy, precision, recall, F1, C-index, NRI, and IDI. Pairwise DeLong AUC comparisons between feature sets and classifiers. CONSORT-style cohort exclusion audit.

### 11. Result Generation

All outputs are saved to `outputs/`. Each experiment produces self-contained results (CSV, JSON, PNG, Markdown report) in its own `results/` directory.

---

## Methods

### Spatial Features

16 features per modality: lobar EN/NC/ED fractions within four lobes, plus four global tumour-burden ratios. Across four modalities (T1, T2, T1GD, FLAIR), this yields 64 spatial features per patient.

### Clinical-Molecular Features

6 features: age, sex, IDH mutation status, MGMT promoter methylation, WHO grade, and extent of resection.

### Combined Features

70 features: the union of spatial (64) and clinical-molecular (6) feature sets.

### Radiogenomic Analysis

Univariate testing (Mann-Whitney U, Cliff's delta, ROC-AUC) with BH FDR correction, followed by multivariable logistic regression for MGMT and IDH status prediction. 5-fold stratified cross-validation with bootstrap confidence intervals.

### Survival Analysis

Cox proportional hazards with ridge penalty and Random Survival Forests. Models evaluated by C-index, time-dependent AUC, and integrated Brier score. Nested models assessed via likelihood ratio tests.

### Incremental Value

Net Reclassification Index (NRI) and Integrated Discrimination Improvement (IDI) at 12, 24, and 36 months using continuous and categorical definitions. Bootstrap inference (B = 5,000).

### Calibration

Brier score, calibration curves, and Hosmer-Lemeshow goodness-of-fit testing across spatial, clinical, and combined feature sets.

### Decision Curve Analysis

Net benefit across threshold probabilities from 0% to 60%, with bootstrap 95% confidence intervals.

### Explainability

SHAP TreeExplainer for XGBoost, permutation importance (30 repeats) for SVM. Feature contributions aggregated by modality, lobe, and subregion.

### Incremental Value Analysis

Nested Cox models with likelihood ratio tests, delta C-index bootstrap, and NRI/IDI reclassification at multiple time horizons.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<username>/GBM-Lobewise-Subregion-Modelling.git
cd GBM-Lobewise-Subregion-Modelling

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python >= 3.10, numpy, pandas, scipy, scikit-learn, xgboost, shap, matplotlib, seaborn, joblib, optuna, nibabel, lifelines, sksurv.

**Optional (Linux/macOS only):** ANTs registration (`pip install antspy`). If unavailable, set `atlas.use_ants_registration = false` in `config.json` to use the affine fallback.

---

## Dataset

### Source

This project uses the **UCSF-PDGM** dataset, a public collection of pre-operative MRI and clinical metadata for glioblastoma patients.

### Expected Directory Structure

```
UCSF/
├── DATA-IMAGE-STRUCTURAL/
│   ├── UCSF-PDGM-0001/
│   │   ├── UCSF-PDGM-0001_T1.nii.gz
│   │   ├── UCSF-PDGM-0001_T2.nii.gz
│   │   ├── UCSF-PDGM-0001_T1GD.nii.gz
│   │   └── UCSF-PDGM-0001_FLAIR.nii.gz
│   └── ...
├── DATA-AUTOMATED-SEGMENT/
│   ├── UCSF-PDGM-0001_tumor_segmentation.nii.gz
│   └── ...
```

### Clinical Metadata

Place `UCSF-PDGM-metadata_v5.csv` in the project root. This file contains patient-level clinical and molecular data (age, sex, MGMT, IDH, WHO grade, extent of resection, overall survival).

### Data Preparation

1. Download the UCSF-PDGM dataset and place NIfTI files in the expected directory structure.
2. Place `UCSF-PDGM-metadata_v5.csv` in the project root.
3. Ensure the SRI24 atlas data is in `data/SRI24/` (parcellation labels are included in the repository).
4. Run the atlas registration pipeline to extract spatial features.

**Note:** Raw NIfTI files and the UCSF dataset are not included in this repository due to data use agreements.

---

## Usage

### Feature Extraction

```bash
# Register SRI24 atlas and extract 16 features per modality
python src/atlas_registration.py --modality T1
python src/atlas_registration.py --modality T2
python src/atlas_registration.py --modality T1GD
python src/atlas_registration.py --modality FLAIR
```

### Preprocessing

```bash
# Preprocess raw features (imputation + scaling) per modality
python src/preprocessing.py --input outputs/features_raw.csv --output outputs/features_processed.csv
```

### Multimodal Dataset Construction

```bash
# Merge four modality CSVs into a single multimodal table
python src/multimodal_lobewise/build_multimodal_dataset.py

# Preprocess the merged dataset (imputation + scaling, leakage-corrected)
python src/multimodal_lobewise/preprocess_multimodal.py
```

### Training

```bash
# Train XGBoost classifier with Optuna hyperparameter tuning
python src/train.py

# Train XGBoost baseline (multimodal pipeline)
python src/multimodal_lobewise/train_multimodal_model.py

# Train leakage-corrected SVM (RBF kernel)
python src/multimodal_lobewise/train_svm_model.py
```

### Baseline Comparison

```bash
# Compare LR, RF, XGB, SVM across spatial, clinical, and combined feature sets
python src/baseline_comparison.py
```

### Radiogenomic Analysis

```bash
# Spatial feature vs molecular marker associations
python src/radiogenomic_analysis.py
```

### Survival Analysis

```bash
# Cox PH + Random Survival Forest survival analysis
python src/survival_analysis.py

# Nested Cox analysis for incremental prognostic value
python src/nested_cox_analysis.py

# NRI/IDI reclassification analysis
python src/nri_idi_analysis.py

# NRI/IDI sensitivity analysis
python src/nri_sensitivity_analysis.py
```

### SHAP Analysis

```bash
# SHAP + XGBoost feature importance
python src/multimodal_lobewise/analyze_feature_importance.py

# SVM permutation importance
python src/multimodal_lobewise/analyze_svm_results.py

# SHAP stability evaluation
python src/multimodal_lobewise/shap_stability_evaluation.py
```

### Statistical Comparison

```bash
# Pairwise DeLong AUC comparisons
python src/statistical_comparison.py

# Bootstrap 95% CIs for test metrics
python src/compute_metric_bootstrap_cis.py
```

### Experiments

```bash
# Robustness: Feature selection
python experiments/robustness/feature_selection/run_experiment.py

# Robustness: Proportional hazards diagnostics
python experiments/robustness/proportional_hazards/run_ph_robustness.py

# Robustness: Representation ablation
python experiments/robustness/representation_ablation/run_ablation.py

# Robustness: Nested repeated CV validation
python experiments/robustness/reproducibility/run_validation.py

# Statistics: Bootstrap CI for MGMT AUC
python experiments/statistics/bootstrap/run_experiment.py

# Statistics: Cohort exclusion audit
python experiments/statistics/cohort_audit/run_audit.py
```

### Figures

```bash
# Generate anatomical overlay figure
python src/generate_anatomical_overlay.py
```

---

## Results

### Model Performance

| Model               | Feature Set   | ROC-AUC | 95% CI         |
| ------------------- | ------------- | ------- | -------------- |
| SVM (RBF)           | Spatial (64)  | 0.632   | [0.589, 0.675] |
| SVM (RBF)           | Clinical (6)  | 0.772   | [0.734, 0.810] |
| SVM (RBF)           | Combined (70) | 0.799   | [0.763, 0.835] |
| XGBoost             | Spatial (64)  | 0.562   | --             |
| Logistic Regression | Clinical (6)  | 0.772   | --             |

### Key Findings

- **Temporal and frontal enhancing tumour burden** are the strongest spatial survival signatures across modalities. T1GD temporal EN ratio is the top SVM feature by permutation importance.
- **T1-weighted imaging carries the most predictive signal** among the four sequences, followed by T1GD, FLAIR, and T2.
- **Spatial features add independent prognostic value** over clinical-molecular covariates: the combined model achieves higher C-index than the clinical-only model (nested Cox likelihood ratio test p < 0.05).
- **MGMT promoter methylation** is the strongest molecular predictor, with logistic regression achieving AUC = 0.76 from spatial features alone (5-fold CV).
- **Calibration is adequate** for the combined model (Brier score < 0.20) but reveals some overconfidence at high-risk thresholds.
- **SHAP analysis** confirms enhancing tumour subregions in the temporal and frontal lobes dominate model predictions, consistent with the permutation importance ranking.
- **Multimodal fusion adds limited independent information** because same-region measurements across sequences are highly correlated (mean cross-modality same-subregion r = 0.46).

---

## Reproducibility

- **Random seeds:** All pipelines use `random_seed = 42` for train-test splits, cross-validation, and bootstrap resampling.
- **Cross-validation:** 5-fold stratified CV for model selection, 10-fold stratified CV for final evaluation, repeated stratified CV (5x10) for nested validation.
- **Bootstrap:** 5,000 bootstrap replicates for confidence intervals on AUC, C-index, NRI, and IDI.
- **Deterministic preprocessing:** Median imputation and standard scaling are fitted on training data only and serialized for reuse.
- **Checkpointing:** Trained models and preprocessing artefacts are serialized (`.pkl`) for reproducible inference.
- **Self-contained experiments:** Each experiment in `experiments/` produces its own `results/` directory with all outputs (CSV, JSON, PNG, Markdown report) independent of other experiments.
- **Configuration:** All hyperparameters and paths are centralised in `config.json`.

---

## License

This repository is licensed under the [Apache License 2.0](LICENSE).

---

## Acknowledgements

- **UCSF-PDGM** -- Glioblastoma dataset with MRI and clinical metadata used for all analyses.
- **SRI24 Atlas** -- Stereotactic brain atlas used for lobe parcellation and spatial feature extraction.
- **LPBA40** -- Labeled brain parcellation atlas providing the region definitions aggregated into the 4-lobe atlas.
- **BraTS** -- The Tumour segmentation labels (NC=1, ED=2, EN=4) follow the BraTS convention.
- **lifelines** -- Survival analysis library for Cox proportional hazards modelling.
- **scikit-survival** -- Random Survival Forests and time-dependent AUC computation.
- **SHAP** -- Model-agnostic explainability framework used for feature importance analysis.
- **Optuna** -- Hyperparameter optimisation framework for XGBoost tuning.
