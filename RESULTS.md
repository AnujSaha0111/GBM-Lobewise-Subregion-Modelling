# Lobewise Subregion Modelling (T1, T2, FLAIR, T1GD) — All Patients

## 1. Overview

This document presents the results of a population-level spatial survival analysis of glioblastoma (GBM) using multimodal MRI. A cohort of approximately 500 patients from the UCSF-PDGM dataset was analysed through an atlas-guided lobewise subregion modelling framework. Tumour involvement was quantified across four cortical lobes (frontal, temporal, parietal, occipital) and three tissue compartments (enhancing tumour [EN], necrotic core [NC], peritumoural oedema [ED]), yielding 12 spatial features per patient per modality. The association between lobewise tumour burden and overall survival (OS) was evaluated by comparing high-risk (OS ≤ 12 months) and low-risk (OS > 12 months) patient groups across four MRI sequences:

| Modality | MRI Sequence |
|----------|-------------|
| T1 | T1-weighted |
| T2 | T2-weighted |
| T1GD | T1-weighted gadolinium-enhanced |
| FLAIR | Fluid-attenuated inversion recovery |

---

## 2. Cohort Statistics

After filtering for data completeness across all four modalities and quality control criteria, the analytic cohort comprised:

| Metric | Value |
|--------|-------|
| Total patients | 498 |
| High-risk (OS ≤ 12 mo) | 218 |
| Low-risk (OS > 12 mo) | 280 |
| Modalities analysed | T1, T2, T1GD, FLAIR |
| Spatial features per modality | 12 lobe–subregion occupancy ratios |
| Atlas resolution | 4 lobes (frontal, temporal, parietal, occipital) |

All 498 patients had complete feature data across all four modalities, enabling consistent cross-modality comparison.

---

## 3. Spatial Population Analysis

The pipeline aggregated lobe–subregion occupancy ratios across the entire patient cohort and evaluated their association with overall survival. For each of the 12 features, we computed group means, Cohen's d (standardised effect size), Mann–Whitney U statistics, odds ratios, and involvement prevalence.

The 12 spatial features are defined as:

```
{lobe}_{subregion}_ratio = voxels(lobe ∩ subregion) / total_voxels(lobe)
```

where `lobe ∈ {frontal, temporal, parietal, occipital}` and `subregion ∈ {ed, en, nc}`.

The complete feature set is:

| Index | Feature | Description |
|-------|---------|-------------|
| 1 | `frontal_ed_ratio` | Oedema burden in frontal lobe |
| 2 | `frontal_en_ratio` | Enhancing tumour burden in frontal lobe |
| 3 | `frontal_nc_ratio` | Necrotic core burden in frontal lobe |
| 4 | `temporal_ed_ratio` | Oedema burden in temporal lobe |
| 5 | `temporal_en_ratio` | Enhancing tumour burden in temporal lobe |
| 6 | `temporal_nc_ratio` | Necrotic core burden in temporal lobe |
| 7 | `parietal_ed_ratio` | Oedema burden in parietal lobe |
| 8 | `parietal_en_ratio` | Enhancing tumour burden in parietal lobe |
| 9 | `parietal_nc_ratio` | Necrotic core burden in parietal lobe |
| 10 | `occipital_ed_ratio` | Oedema burden in occipital lobe |
| 11 | `occipital_en_ratio` | Enhancing tumour burden in occipital lobe |
| 12 | `occipital_nc_ratio` | Necrotic core burden in occipital lobe |

---

## 4. Strongest Prognostic Spatial Features

The table below summarises the mean Cohen's d across the four modalities and the significance trend for each feature. Features are ordered by mean effect size.

| Feature | Mean Cohen's d | Consistent (p < 0.05) | Trend |
|---------|:-------------:|:---------------------:|-------|
| `frontal_en_ratio` | 0.409 | 4 / 4 modalities | Risk-associated |
| `temporal_en_ratio` | 0.255 | 4 / 4 modalities | Risk-associated |
| `frontal_nc_ratio` | 0.184 | 4 / 4 modalities | Risk-associated |
| `frontal_ed_ratio` | 0.086 | 4 / 4 modalities | Risk-associated |
| `temporal_nc_ratio` | 0.085 | 4 / 4 modalities | Risk-associated |
| `parietal_en_ratio` | 0.117 | 0 / 4 modalities | Non-significant |
| `parietal_nc_ratio` | 0.091 | 0 / 4 modalities | Non-significant |
| `parietal_ed_ratio` | 0.050 | 0 / 4 modalities | Non-significant |
| `occipital_en_ratio` | 0.007 | 3 / 4 modalities | Weak / marginal |
| `occipital_ed_ratio` | 0.028 | 0 / 4 modalities | Non-significant |
| `occipital_nc_ratio` | 0.002 | 0 / 4 modalities | Non-significant |
| `temporal_ed_ratio` | −0.012 | 0 / 4 modalities | Non-significant |

**Interpretation.** Enhancing tumour burden within frontal regions demonstrated the strongest and most reproducible association with poor overall survival across all MRI modalities, with a mean Cohen's d of 0.41 and statistical significance in all four sequences. This finding is consistent with prior literature reporting frontal lobe involvement as an adverse prognostic factor in GBM, potentially due to proximity to eloquent cortex limiting safe resection.

Temporal lobe enhancing tumour burden showed a moderate but consistent effect (Cohen's d ≈ 0.26, significant across all modalities). Frontal necrotic core and oedema burden also showed consistent but weaker associations.

Parietal and occipital lobe features did not reach statistical significance at the p < 0.05 threshold in any modality, suggesting that tumour involvement in these lobes does not reliably discriminate between high-risk and low-risk groups at this cohort scale.

---

## 5. Cross-Modality Consensus Analysis

### Modality Agreement

The four MRI sequences showed near-perfect agreement in their spatial risk rankings. Of the 12 lobe–subregion features, six (`frontal_en_ratio`, `frontal_nc_ratio`, `frontal_ed_ratio`, `temporal_en_ratio`, `temporal_nc_ratio`, `occipital_en_ratio`) showed a positive Cohen's d direction in all four modalities. Five features showed 100% modality consistency (significant risk association in all four modalities) at p < 0.05.

### Cross-Modality Correlations

Pearson correlations between modalities for Cohen's d, odds ratio, and prevalence were uniformly near-perfect (r > 0.99 across all modality pairs):

| Metric | Mean Pearson r | Min–Max |
|--------|:-------------:|:-------:|
| Cohen's d | > 0.999 | 0.999–1.000 |
| Odds ratio | > 0.99 | 0.988–1.000 |
| Prevalence | > 0.999 | 0.999–1.000 |

Spearman rank correlations were similarly high (ρ ≥ 0.97 for all comparisons).

### Interpretation

The extremely high cross-modality agreement suggests that **coarse lobewise tumour burden dominates modality-specific variation** at the current atlas resolution. At the 4-lobe scale, all four MRI sequences capture essentially the same spatial information: a given lobe's involvement by a given tissue compartment is consistent enough across sequences that the between-modality differences in contrast weighting do not substantially alter the spatial risk profile. Modality-specific prognostic patterns may emerge at higher spatial resolution where fine-grained tissue contrast differences become relevant.

---

## 6. Population-Level Probabilistic Risk Atlas

The generated atlas quantifies the probability that a patient belongs to the high-risk group given the presence of tumour involvement in a specific lobe–subregion (involvement defined as >1% occupancy of the lobe). The atlas represents population-level associations between tumour spatial distribution and survival risk, computed across the entire 498-patient cohort.

### Observed Risk Probability Patterns

| Feature | Risk Probability (T2) | Modalities Consistent |
|---------|:--------------------:|:--------------------:|
| `frontal_en_ratio` | 0.57 | 4 / 4 |
| `frontal_nc_ratio` | 0.56 | 4 / 4 |
| `occipital_en_ratio` | 0.54 | 3 / 4 |
| `frontal_ed_ratio` | 0.49 | 4 / 4 |
| `temporal_en_ratio` | 0.51 | 4 / 4 |

**Dominant high-risk zones.** The frontal lobe—particularly its enhancing tumour compartment—emerges as the dominant region associated with poor prognosis. Patients with frontal EN involvement have an approximately 57% probability of being high-risk (OS ≤ 12 months).

**Secondary risk regions.** Temporal EN involvement shows a moderate risk probability (~0.51), while temporal NC and frontal NC show weaker but consistently positive associations.

**Low-signal regions.** Occipital and parietal lobe involvement do not substantially shift risk probability away from the cohort baseline (44% high-risk prevalence).

**Important caveat.** The current atlas is a coarse-resolution regional map rather than a voxelwise probabilistic brain map. It identifies lobe-level associations and should not be interpreted as a fine-grained anatomical risk localisation.

---

## 7. Statistical Observations

### Effect Size Consistency

`frontal_en_ratio` showed the most stable and positive Cohen's d across all four modalities (d = 0.408–0.410), with the lowest cross-modality standard deviation (σ = 0.001). This reproducibility across T1, T2, T1GD, and FLAIR sequences indicates that the frontal EN association is robust to differences in MRI contrast mechanisms.

### Prevalence Statistics

The most prevalent feature across the cohort was `frontal_ed_ratio` (oedema in the frontal lobe), present in approximately 70% of patients. Despite its high prevalence, its discriminative power was modest (Cohen's d ≈ 0.09), suggesting that frontal oedema is nearly ubiquitous in GBM regardless of survival outcome.

### Odds Ratio Trends

`frontal_en_ratio` exhibited the highest odds ratio (~2.3), indicating that patients with frontal EN involvement are more than twice as likely to belong to the high-risk group compared to those without frontal EN involvement. This pattern was consistent across all modalities.

### Modality Consistency

| Consistency | Features |
|:-----------:|----------|
| 4 / 4 (100%) | `frontal_en_ratio`, `frontal_nc_ratio`, `frontal_ed_ratio`, `temporal_en_ratio`, `temporal_nc_ratio` |
| 3 / 4 (75%) | `occipital_en_ratio` |
| 0 / 4 (0%) | `parietal_ed_ratio`, `parietal_en_ratio`, `parietal_nc_ratio`, `occipital_ed_ratio`, `occipital_nc_ratio`, `temporal_ed_ratio` |

No feature showed a consistently protective (negative) association across modalities, suggesting that at the lobewise scale, tumour involvement patterns are primarily risk-associated or non-significant, rather than protective.

---

## 8. Visualisations Generated

The following figures are produced by `lobewise_population_analysis.py` and saved to `outputs/population_atlas/combined/`:

| Figure | Description |
|--------|-------------|
| `consensus_barplot.png` | Horizontal bar chart ranking the 12 features by multimodal consensus score (mean Cohen's d × modality consistency). Positive scores indicate risk association; features are colour-coded by effect direction. |
| `modality_agreement_heatmap.png` | Heatmap of Cohen's d values across the 12 features (columns) and 4 modalities (rows). Significance markers indicate risk-associated (\*), protective (x), or non-significant (+/−) patterns, providing an at-a-glance view of cross-modality consistency. |
| `correlation_heatmap_cohens_d.png` | Pairwise Pearson correlation of Cohen's d values between modalities. Near-diagonal values (r > 0.999) confirm near-identical spatial effect size rankings. |
| `correlation_heatmap_odds_ratio.png` | Pairwise Pearson correlation of odds ratio values between modalities. |
| `correlation_heatmap_prevalence.png` | Pairwise Pearson correlation of involvement prevalence between modalities. |

---

## 9. Key Conclusions

1. **Frontal enhancing tumour burden is the strongest prognostic spatial pattern.** The `frontal_en_ratio` feature shows the largest and most consistent effect size (Cohen's d ≈ 0.41, p < 10⁻⁸, odds ratio ≈ 2.3) across all four MRI modalities, establishing frontal EN involvement as a robust lobewise marker of poor overall survival.

2. **Spatial prognostic patterns are highly reproducible across MRI modalities.** Near-perfect correlations (r > 0.99) for Cohen's d, odds ratio, and prevalence between all modality pairs indicate that the lobewise risk profile does not depend on the choice of MRI sequence.

3. **Coarse lobewise spatial burden dominates modality-specific effects.** The extremely high cross-modality agreement suggests that at the 4-lobe resolution, the spatial information content is largely redundancy across sequences. Modality-specific prognostic signatures—if they exist—likely require finer spatial granularity to be detected.

4. **The framework successfully generates a multimodal probabilistic spatial risk atlas at regional resolution.** Despite the coarse atlas, the pipeline identifies reproducible risk-associated lobe–subregion patterns and quantifies their cross-modality consistency, demonstrating the feasibility of atlas-guided population-level spatial survival modelling in GBM.

---

## 10. Limitations

The following limitations should be considered when interpreting the results:

- **Coarse 4-lobe atlas resolution.** The analysis aggregates tumour involvement over entire cortical lobes, which may mask important sub-lobar spatial patterns. A patient with a small enhancing focus in the anterior frontal lobe is treated identically to one with diffuse frontal EN involvement.
- **No voxelwise localisation.** Voxel-based lesion-symptom mapping (VLSM) or voxel-wise probabilistic mapping—which can identify focal prognostic hotspots at millimetre scale—is not performed.
- **No intra-lobe spatial heterogeneity modelling.** The distribution of tumour burden within a lobe (focal vs diffuse, depth from cortex, relationship to sulcal boundaries) is not characterised.
- **No texture or radiomic feature integration.** Only occupancy ratios are used; first-order intensity statistics, texture features, shape descriptors, and higher-order radiomic features are excluded from the current analysis.
- **Modality redundancy at coarse scale.** The near-perfect cross-modality correlation suggests that the four MRI sequences provide largely redundant spatial information at the 4-lobe resolution. This does not preclude the existence of modality-specific prognostic patterns at higher spatial resolution.
- **No molecular or clinical covariate adjustment.** The analysis is purely spatial and does not account for MGMT promoter methylation, IDH mutation status, age, KPS, or extent of resection, all of which are known prognostic factors in GBM.

---

## 11. Future Work

Several extensions are planned to address the current limitations and to deepen the spatial analysis:

- **Voxelwise probabilistic atlas generation.** Transition from lobewise aggregation to voxel-level probabilistic mapping to identify focal prognostic hotspots at native MRI resolution.
- **116-region atlas modelling.** Utilise the full SRI24 tzo116 parcellation for fine-grained region-level analysis, enabling identification of prognostic patterns within individual gyral and subcortical structures.
- **Hemispheric asymmetry analysis.** Separate left and right hemisphere involvement to investigate lateralisation effects on prognosis.
- **Graph-based spatial modelling.** Represent tumour spatial configuration as graphs encoding lobe-to-lobe spread patterns and connectivity-informed burden distribution.
- **Integration with radiomics and deep learning.** Combine occupancy ratios with texture, shape, and intensity-based radiomic features, and explore deep learning-based spatial encoding for end-to-end survival prediction.
- **Covariate-adjusted spatial modelling.** Incorporate molecular, clinical, and demographic covariates to estimate the independent prognostic contribution of spatial tumour patterns beyond known clinical factors.
