# Lobewise Subregion Modelling for Population-Level Probabilistic Risk Atlas Generation in Glioblastoma Using Multimodal MRI

**Spatial survival modelling · Multimodal MRI integration · Probabilistic prognostic atlas · Cross-modality consensus**

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Problem Statement](#2-problem-statement)
- [3. Core Contributions](#3-core-contributions)
- [4. Dataset](#4-dataset)
- [5. Spatial Modelling Pipeline](#5-spatial-modelling-pipeline)
- [6. Spatial Features](#6-spatial-features)
- [7. Population-Level Risk Atlas](#7-population-level-risk-atlas)
- [8. Key Findings](#8-key-findings)
- [9. Repository Structure](#9-repository-structure)
- [10. Running the Pipeline](#10-running-the-pipeline)
- [11. Outputs](#11-outputs)
- [12. Limitations](#12-limitations)
- [13. Future Work](#13-future-work)
- [14. Citation](#14-citation)

---

## 1. Project Overview

Glioblastoma (GBM) is the most aggressive primary brain tumour in adults, with a median overall survival (OS) of approximately 12–15 months under standard therapy. Prognosis is influenced by molecular subtype, extent of resection, patient age, and—critically—tumour location. GBM exhibits pronounced spatial heterogeneity: tumours variably involve different lobes and tissue compartments (enhancing core, necrotic core, peritumoural oedema), and the spatial pattern of involvement carries prognostic information that conventional volumetric metrics alone may not capture.

Multimodal MRI (T1-weighted, T2-weighted, T1-weighted gadolinium-enhanced, FLAIR) provides complementary views of tumour morphology, each sequence highlighting different tissue compartments. Integrating these modalities through a common spatial reference frame enables systematic comparison of how each modality contributes to the spatial characterisation of prognostic risk.

This project aggregates lobewise tumour subregion involvement across approximately 500 patients from the UCSF-PDGM dataset and constructs a **population-level probabilistic risk atlas**. By mapping the spatial distribution of tumour burden in high-risk versus low-risk patients, we identify reproducible lobewise patterns associated with poor overall survival and quantify the cross-modality agreement of these spatial associations.

---

## 2. Problem Statement

> Across approximately 500 patients, aggregate which lobe–subregion combinations most frequently appear in high-risk versus low-risk patients and generate a population-level probabilistic risk atlas.

Patients are stratified into two risk groups based on overall survival:

| Risk Group | Definition | Count |
|-----------|------------|-------|
| High-risk | OS ≤ 12 months | ~220 |
| Low-risk | OS > 12 months | ~280 |

For each of four MRI modalities (T1, T2, T1GD, FLAIR), we extract 12 lobe–subregion occupancy features describing tumour involvement across frontal, temporal, parietal, and occipital lobes, each divided into enhancing tumour (EN), necrotic core (NC), and peritumoural oedema (ED) compartments. Statistical tests compare high-risk versus low-risk distributions for each feature, producing a quantitative spatial risk profile that is further aggregated across modalities to identify consensus prognostic regions.

---

## 3. Core Contributions

| Contribution | Description |
|-------------|-------------|
| **Multimodal MRI integration** | Parallel analysis of T1, T2, T1GD, and FLAIR sequences through a unified spatial framework |
| **Atlas-guided spatial modelling** | SRI24 parcellation registered to each patient's native space for consistent lobe assignment |
| **Lobewise subregion aggregation** | 12 spatial features per modality capturing lobe-level tumour compartment involvement |
| **Population-level probabilistic risk modelling** | Risk probability, odds ratios, and effect sizes computed at cohort scale |
| **Cross-modality consensus analysis** | Systematic quantification of agreement across modalities using Cohen's d, prevalence, and consistency metrics |
| **Statistical spatial association analysis** | Mann–Whitney U tests, Cohen's d, and odds ratios for each lobe–subregion–modality combination |
| **Consensus ranking of prognostic regions** | Multimodal consensus scores identifying the most reproducible spatial risk patterns |

---

## 4. Dataset

### UCSF-PDGM

We use the **UCSF Preoperative Diffuse Glioma MRI (UCSF-PDGM)** dataset (Calabrese et al., 2022), comprising:

- **~500 patients** with histopathologically confirmed glioblastoma
- **Survival metadata** including overall survival in days/months
- **Automated tumour segmentations** with BraTS-compatible label maps (label 1 = necrotic core, label 2 = peritumoural oedema, label 4 = enhancing tumour)
- **Multimodal MRI volumes**: T1-weighted, T2-weighted, T1-weighted gadolinium-enhanced (T1GD), and FLAIR sequences

**Important:** The pipeline operates on extracted feature CSV files. Raw MRI volumes are required only for the feature extraction stage (`atlas_registration.py`); once features are saved, the atlas construction and population-level analysis can be run independently of the original imaging data.

### SRI24 Atlas

The [SRI24 atlas](https://www.nitrc.org/projects/sri24/) provides a cortical parcellation (~116 regions) in a common stereotaxic space. We construct a 4-lobe atlas (frontal, temporal, parietal, occipital) from the SRI24 parcellation labels and register it to each patient's native MRI space.

---

## 5. Spatial Modelling Pipeline

The pipeline consists of seven sequential stages:

### Stage 1: Atlas Construction

The SRI24 tzo116 parcellation is mapped to a 4-lobe representation by grouping constituent cortical regions into frontal, temporal, parietal, and occipital labels. A cached lobe atlas (`sri24_4lobe_atlas.nii.gz`) is written to disk for reuse.

```
SRI24 ~116 regions → 4-lobe atlas {frontal, temporal, parietal, occipital}
```

### Stage 2: Atlas-to-Patient Registration

The lobe atlas is registered from SRI24 space to each patient's native MRI space. ANTs (SyN) registration is used on Linux/macOS; an affine fallback is available on Windows. Transforms are cached per patient to avoid recomputation.

### Stage 3: Lobe / Subregion Feature Extraction

For each patient and each MRI modality, tumour segmentation labels are intersected with the registered lobe atlas. Twelve occupancy ratios are computed:

```
{lobe}_{subregion}_ratio = voxels(lobe ∩ subregion) / total_voxels(lobe)
```

where `lobe ∈ {frontal, temporal, parietal, occipital}` and `subregion ∈ {ed, en, nc}`.

### Stage 4: Population Aggregation

Feature data from all patients is assembled into modality-specific dataframes, aligned by `patient_id`, and restricted to patients present across all four modalities for consistent cross-modality comparison.

### Stage 5: High-Risk vs Low-Risk Statistical Analysis

For each of the 12 features × 4 modalities, we compute:

| Metric | Description |
|--------|-------------|
| **Group means** | Mean occupancy ratio in high-risk and low-risk groups |
| **Cohen's d** | Standardised effect size (positive = higher in high-risk) |
| **Mann–Whitney U** | Non-parametric test of distribution difference |
| **Odds ratio** | Ratio of the odds of involvement in high-risk vs low-risk |
| **Prevalence** | Percentage of patients with non-negligible involvement (>1% occupancy) |

### Stage 6: Cross-Modality Consensus Analysis

Feature-level statistics are compared across all four modality pairs using Pearson and Spearman correlations. Modality consistency is defined as the proportion of modalities in which a feature shows a significant (p < 0.05) positive Cohen's d (risk-associated).

### Stage 7: Probabilistic Risk Atlas Generation

For each lobe–subregion combination, the conditional risk probability is computed as:

```
risk_probability(feature) = n_high_risk_involved / (n_high_risk_involved + n_low_risk_involved)
```

This yields a population-level map of how likely a patient is to be high-risk given tumour involvement in a specific lobe–subregion.

---

## 6. Spatial Features

### Lobe–Subregion Occupancy Ratios (12 features)

| Lobe | ED (Oedema) | EN (Enhancing) | NC (Necrotic Core) |
|------|:-----------:|:--------------:|:------------------:|
| **Frontal** | `frontal_ed_ratio` | `frontal_en_ratio` | `frontal_nc_ratio` |
| **Temporal** | `temporal_ed_ratio` | `temporal_en_ratio` | `temporal_nc_ratio` |
| **Parietal** | `parietal_ed_ratio` | `parietal_en_ratio` | `parietal_nc_ratio` |
| **Occipital** | `occipital_ed_ratio` | `occipital_en_ratio` | `occipital_nc_ratio` |

Each ratio represents the fraction of voxels within a given lobe that are occupied by the specified tissue compartment. Values range from 0 (no involvement) to 1 (full lobe occupancy).

### Atlas Resolution

The current atlas is a **coarse 4-lobe representation**. This resolution was chosen for interpretability and computational efficiency, but it is an important limitation: spatial variation within a lobe (e.g., anterior vs posterior frontal) is not captured, and hemispheric laterality is not modelled.

---

## 7. Population-Level Risk Atlas

### Probabilistic Map

For each feature, the risk probability P(high-risk | involvement) quantifies the proportion of involved patients who belong to the high-risk group:

| Feature | Risk Probability | Modalities Consistent |
|---------|:---------------:|:--------------------:|
| `frontal_en_ratio` | 0.57 | 4 / 4 |
| `frontal_nc_ratio` | 0.56 | 4 / 4 |
| `occipital_en_ratio` | 0.54 | 3 / 4 |
| `temporal_en_ratio` | 0.51 | 4 / 4 |

*(Values shown are approximate; exact values vary slightly by modality.)*

### Consensus Score

The multimodal consensus score combines effect size and cross-modality reproducibility:

```
consensus_score = mean(Cohen's d) × modality_consistency
```

Lobe–subregion features are ranked by this score to identify the most reproducible spatial prognostic patterns.

### Key Atlas Components

| Component | Description |
|-----------|-------------|
| `occurrence_frequency.csv` | Prevalence, odds ratio, and group-specific involvement percentages |
| `statistical_tests.csv` | Group means, Cohen's d, Mann–Whitney U, p-values |
| `risk_probability_map.csv` | Conditional risk probability per feature |
| `modality_agreement.csv` | Cross-modality summary statistics and consistency scores |
| `consensus_rankings.csv` | Features ranked by multimodal consensus score |
| `multimodal_summary.csv` | Per-feature modality-wise significance counts |

### Visualisations

- **Correlation heatmaps** — Pearson r between modality pairs for Cohen's d, odds ratio, and prevalence
- **Consensus barplot** — Ranked features by consensus score
- **Modality agreement heatmap** — Cohen's d and significance per feature × modality

---

## 8. Key Findings

The following findings are based on the current coarse-resolution 4-lobe analysis:

1. **Frontal enhancing tumour ratio (`frontal_en_ratio`) is the strongest prognostic feature** across all four modalities, with the highest Cohen's d (~0.41) and perfect cross-modality consistency (4/4 modalities significant at p < 0.05). This confirms that enhancing tumour burden in the frontal lobe is robustly associated with high-risk status.

2. **Temporal enhancing tumour ratio (`temporal_en_ratio`) shows a moderate but consistent signal** (Cohen's d ~0.26, 4/4 modalities significant). Temporal lobe involvement is also associated with increased risk, though the effect size is smaller than for frontal EN involvement.

3. **Strong cross-modality agreement**: Four lobe–subregion features (`frontal_en_ratio`, `frontal_nc_ratio`, `frontal_ed_ratio`, `temporal_en_ratio`, `temporal_nc_ratio`) show 100% modality consistency (significant risk association in all four MRI sequences). Pearson correlations between modalities for Cohen's d and prevalence are high, indicating that spatial patterns are largely modality-invariant at this scale.

4. **Modality redundancy at coarse lobewise scale**: The high cross-modality correlation of Cohen's d values suggests that at the 4-lobe resolution, the four MRI sequences provide largely redundant spatial information. Finer-grained spatial analysis may reveal modality-specific patterns not captured here.

5. **Parietal and occipital features** do not show statistically significant associations in any modality, suggesting that tumour involvement in these lobes does not discriminate between high-risk and low-risk groups at this cohort scale.

---

## 9. Repository Structure

```
gbm_lobewise_prognostic_atlas/
├── data/
│   ├── raw/
│   │   ├── ucsf_with_clinical.csv        # Clinical metadata + OS
│   │   └── UCSF-PDGM-metadata_v5.csv     # Raw metadata
│   └── SRI24/                            # SRI24 atlas files (not committed)
├── src/
│   ├── atlas_registration.py             # Stage 1–3: Atlas construction, registration, feature extraction
│   ├── preprocessing.py                  # Risk label assignment, imputation, filtering
│   ├── train.py                          # Modality-specific XGBoost training (legacy)
│   └── lobewise_population_analysis.py   # Stage 4–7: Population atlas, statistics, consensus
├── outputs/
│   ├── features_raw_t1.csv               # Extracted features per modality
│   ├── features_raw_t2.csv
│   ├── features_raw_t1gd.csv
│   ├── features_raw_flair.csv
│   ├── sri24_4lobe_atlas.nii.gz          # Cached lobe atlas in SRI24 space
│   ├── transforms/                       # Per-patient ANTs transform cache
│   └── population_atlas/
│       ├── t1/                           # Per-modality statistical outputs
│       │   ├── statistical_tests.csv
│       │   ├── occurrence_frequency.csv
│       │   ├── risk_probability_map.csv
│       │   └── clean_population_features.csv
│       ├── t2/                           # (same structure)
│       ├── t1gd/                         # (same structure)
│       ├── flair/                        # (same structure)
│       └── combined/                     # Cross-modality consensus outputs
│           ├── modality_agreement.csv
│           ├── consensus_rankings.csv
│           ├── multimodal_summary.csv
│           ├── correlation_matrix.csv
│           ├── correlation_heatmap_cohens_d.png
│           ├── correlation_heatmap_odds_ratio.png
│           ├── correlation_heatmap_prevalence.png
│           ├── consensus_barplot.png
│           └── modality_agreement_heatmap.png
├── docs/
│   ├── atlas_registration.md             # Registration and feature extraction documentation
│   └── preprocessing.md                  # Preprocessing documentation
├── config.json                           # All tunable pipeline parameters
├── requirements.txt
└── README.md
```

---

## 10. Running the Pipeline

### Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate     # Linux/macOS
.venv\Scripts\Activate.ps1    # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

> **Note on ANTs:** `antspy` is required for atlas-to-patient registration and is not available via pip on Windows. On Linux/macOS, install separately: `pip install antspy`. On Windows, set `atlas.use_ants_registration: false` in `config.json` to use the affine fallback.

### Stage A — Atlas Registration and Feature Extraction

```bash
# Full batch for each modality (T1, T2, T1GD, FLAIR)
python src/atlas_registration.py --modality T1
python src/atlas_registration.py --modality T2
python src/atlas_registration.py --modality T1GD
python src/atlas_registration.py --modality FLAIR

# Single patient
python src/atlas_registration.py --patient UCSF-PDGM-0004 --modality T1

# Dry run (lists patients without processing)
python src/atlas_registration.py --modality T1 --dry-run
```

This step produces `outputs/features_raw_{modality}.csv`.

### Stage B — Preprocessing

```bash
python src/preprocessing.py
```

This assigns binary risk labels (OS ≤ 12 months = high-risk) and applies quality filtering.

### Stage C — Population-Level Probabilistic Risk Atlas

```bash
python src/lobewise_population_analysis.py
```

This is the **core population analysis pipeline**. It operates directly on the extracted feature CSVs and does **not** require reprocessing the original UCSF imaging data. The script:

1. Loads features across all four modalities
2. Intersects patient IDs for consistent cross-modality comparison
3. Computes per-modality statistical tests (Cohen's d, Mann–Whitney U, odds ratios, prevalence)
4. Generates probabilistic risk maps
5. Computes cross-modality consensus metrics and rankings
6. Produces visualisations (correlation heatmaps, consensus barplots, agreement heatmaps)

---

## 11. Outputs

### Per-Modality Outputs (`outputs/population_atlas/{modality}/`)

| File | Contents |
|------|----------|
| `statistical_tests.csv` | High-risk vs low-risk group means, mean difference, Cohen's d, Mann–Whitney U, p-value |
| `occurrence_frequency.csv` | Overall, high-risk, and low-risk involvement prevalence; odds ratio; p-value; Cohen's d |
| `risk_probability_map.csv` | Conditional probability P(high-risk \| involvement) per feature |
| `clean_population_features.csv` | Intersected patient data with risk labels |

### Cross-Modality Consensus Outputs (`outputs/population_atlas/combined/`)

| File | Contents |
|------|----------|
| `modality_agreement.csv` | Per-feature: mean Cohen's d, standard deviation, mean p-value, odds ratio, prevalence, modality consistency |
| `consensus_rankings.csv` | Features ranked by consensus score (mean Cohen's d × modality consistency) |
| `multimodal_summary.csv` | Per-feature: number of modalities showing significant risk/protective/non-significant association |
| `correlation_matrix.csv` | Pairwise modality correlations (Pearson r and Spearman r) for three metrics |

### Visualisations

| Output | Description |
|--------|-------------|
| `correlation_heatmap_cohens_d.png` | Cross-modality Pearson r for Cohen's d |
| `correlation_heatmap_odds_ratio.png` | Cross-modality Pearson r for odds ratio |
| `correlation_heatmap_prevalence.png` | Cross-modality Pearson r for prevalence |
| `consensus_barplot.png` | Ranked features by multimodal consensus score |
| `modality_agreement_heatmap.png` | Cohen's d per feature × modality with significance markers |

---

## 13. Future Work

Several extensions are planned to address the limitations above and to deepen the spatial analysis:

- **Voxelwise probabilistic atlas**: Move beyond lobewise aggregation to voxel-level probabilistic mapping, enabling identification of focal prognostic hotspots within lobes.
- **116-region atlas**: Utilise the full SRI24 tzo116 parcellation (~116 cortical regions) for finer-grained spatial analysis.
- **Hemispheric modelling**: Separate left and right hemisphere involvement to capture lateralisation effects.
- **Graph-based spatial modelling**: Represent tumour spatial configuration as graphs (e.g., lobe-to-lobe spread patterns, connectivity-informed burden distribution).
- **Radiomic-spatial integration**: Combine occupancy ratios with first-order and higher-order radiomic features (texture, shape, intensity) within each lobe–subregion.
- **Covariate-adjusted spatial analysis**: Incorporate molecular, clinical, and demographic covariates to isolate the independent prognostic contribution of spatial tumour patterns.

---

## 14. Citation

If you use this work, please cite the associated publication (reference forthcoming upon acceptance):

```bibtex
@misc{lobewise2026,
  title        = {Lobewise Subregion Modelling for Population-Level Probabilistic Risk
                  Atlas Generation in Glioblastoma Using Multimodal MRI},
  author       = {Krishna Rathore, Anuj Saha},
  year         = {2026},
  howpublished = {GitHub repository},
  note         = {Available: https://github.com/AnujSaha0111/GBM-Lobewise-Subregion-Modelling}
}
```

---

## License

This project is released under the [LICENSE](LICENSE) file included in this repository.
