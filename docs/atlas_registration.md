# Atlas Registration & Feature Extraction

## What This Script Does

`src/atlas_registration.py` is the most critical script in the pipeline. It performs three tasks:

1. **Builds a 4-lobe SRI24 atlas** — Parses the TZO116 parcellation, groups ~116 cortical regions into 4 lobes (frontal, temporal, parietal, occipital), and fills gaps via distance-transform nearest-seed assignment within a dilated supratentorial mask. Result is cached as `outputs/sri24_4lobe_atlas.nii.gz`.

2. **Registers the atlas to each patient's native MRI space** — Uses ANTs registration (SRI24 T1 → patient fixed image), then warps the atlas and brainmask into patient space using nearest-neighbor interpolation. The patient-side fixed image is selected via `--modality`. Transforms are cached in `outputs/transforms/` as `{patient_id}0GenericAffine.mat` and `{patient_id}1Warp.nii.gz`.

3. **Extracts 16 radiomic features** — From the registered atlas + tumor segmentation, computes 4 global volumetric ratios and 12 lobe-level invasion fractions.

---

## Supported Modalities

The pipeline supports 4 fixed-image modalities. Select one at runtime with `--modality`:

| Flag | Fixed Image File | Config Key | Interpolator |
|------|-----------------|------------|--------------|
| `T1` *(default)* | `{ID}_T1.nii.gz` | `t1_suffix` | `linear` |
| `T2` | `{ID}_T2.nii.gz` | `t2_suffix` | `linear` |
| `T1GD` | `{ID}_T1GD.nii.gz` | `t1gd_suffix` | `linear` |
| `FLAIR` | `{ID}_FLAIR.nii.gz` | `flair_suffix` | `linear` |

> **Interpolator rule:** All intensity modalities use `interpolator="linear"`. The atlas (label volume) and brainmask always use `interpolator="nearestNeighbor"` — this is hardcoded and never changes.

Each run writes to its own CSV so results are never overwritten:

```
outputs/features_raw_t1.csv
outputs/features_raw_t2.csv
outputs/features_raw_t1gd.csv
outputs/features_raw_flair.csv
```

---

## CLI Flags

```bash
# Full batch — default modality (T1)
python src/atlas_registration.py

# Use T1GD as fixed image
python src/atlas_registration.py --modality T1GD

# FLAIR dry-run (list patients only, no processing)
python src/atlas_registration.py --modality FLAIR --dry-run

# T2, single patient
python src/atlas_registration.py --modality T2 --patient UCSF-PDGM-0004

# Custom config path
python src/atlas_registration.py --modality T1GD --config path/to/config.json
```

> When `--modality` is set, the script writes `"active_modality": "<MODALITY>"` into the `atlas` section of `config.json` so every run is self-documenting.

---

## `config.json` — Relevant Keys

```json
"data": {
  "t1_suffix":    "_T1.nii.gz",
  "t2_suffix":    "_T2.nii.gz",
  "t1gd_suffix":  "_T1GD.nii.gz",
  "flair_suffix": "_FLAIR.nii.gz",
  "seg_suffix":   "_tumor_segmentation.nii.gz",
  "sri24_dir":    "data/SRI24",
  "sri24_t1_filename": "spgr.nii.gz"
},
"atlas": {
  "use_ants_registration": true,
  "registration_type": "SyN",
  "transforms_cache_dir": "outputs/transforms",
  "lobe_dilation_voxels": 3,
  "active_modality": "T1GD"     ← written automatically on each run
}
```

> **Note:** `sri24_t1_filename` is always the SRI24 T1 template used as the **moving** image in ANTs. This never changes regardless of `--modality`.

---

## Required Input Files

### SRI24 Atlas (place in `data/SRI24/`)

| File | Description |
|------|-------------|
| `tzo116plus.nii.gz` | Label volume with ~116 cortical regions |
| `suptent.nii.gz` | Supratentorial mask |
| `tissues.nii.gz` | Brain tissue mask (brainmask) |
| `SRI24-tzo116plus.txt` | Label ID → region name mapping |
| `spgr.nii.gz` | SRI24 T1 template (moving image for ANTs) |

Download from: https://www.nitrc.org/projects/sri24/

> The SRI24 template filename varies across distributions. Set `data.sri24_t1_filename` in `config.json` to match your download.

### UCSF-PDGM Data (confirmed suffixes from dataset)

| Modality | Path pattern |
|----------|-------------|
| T1 | `UCSF/DATA-IMAGE-STRUCTURAL/{ID}/{ID}_T1.nii.gz` |
| T2 | `UCSF/DATA-IMAGE-STRUCTURAL/{ID}/{ID}_T2.nii.gz` |
| T1GD | `UCSF/DATA-IMAGE-STRUCTURAL/{ID}/{ID}_T1GD.nii.gz` |
| FLAIR | `UCSF/DATA-IMAGE-STRUCTURAL/{ID}/{ID}_FLAIR.nii.gz` |
| Segmentation | `UCSF/DATA-AUTOMATED-SEGMENT/{ID}_tumor_segmentation.nii.gz` |
| Clinical CSV | `data/raw/UCSF-PDGM-metadata_v5.csv` (columns: `ID`, `OS`) |

Patients missing the selected modality file **or** the segmentation are silently skipped during discovery.

---

## Registration Direction

**SRI24 T1 (moving) → Patient Native Space (fixed)**

The patient-side fixed image is the modality chosen by `--modality`. This is required because UCSF-PDGM tumor segmentations are in each patient's native space, NOT aligned to SRI24.

**Transform cache note:** Transforms are keyed by patient ID (`{patient_id}0GenericAffine.mat`, `{patient_id}1Warp.nii.gz`). All modalities share the same cache folder. If you want per-modality caches to avoid cross-contamination between different registration experiments, change `atlas.transforms_cache_dir` in `config.json` (e.g., `outputs/transforms/t1gd`) before running.

---

## 4-Lobe Atlas Algorithm

1. Parse TZO label file; assign each of ~116 regions to a lobe via name prefixes
2. Build seed volume: voxels get values 1 (frontal), 2 (temporal), 3 (parietal), 4 (occipital)
3. Dilate supratentorial mask by `lobe_dilation_voxels` (default: 3)
4. `scipy.ndimage.distance_transform_edt` with `return_indices=True` fills unassigned voxels with nearest seed label
5. Zero out voxels outside the dilated supratentorial mask

---

## Lobe Ratio Denominator

Lobe ratios use **total lobe voxels** as denominator (not tumor-in-lobe):

```
frontal_ed_ratio = ED voxels in frontal lobe / total frontal lobe voxels
```

This represents **lobe invasion fraction** — what proportion of the frontal lobe is infiltrated by edema. Values are small (~0.001) by design.

---

## Output CSV Schema

Each modality writes to `outputs/features_raw_{modality}.csv`:

| Column | Type | Description |
|--------|------|-------------|
| `patient_id` | str | UCSF-PDGM identifier |
| `global_nc_en_ratio` | float | NC voxels / ET voxels |
| `global_ed_en_ratio` | float | ED voxels / ET voxels |
| `global_ed_total_ratio` | float | ED voxels / WT voxels |
| `tumor_burden_index` | float | WT voxels / brain voxels |
| `{lobe}_{sub}_ratio` | float | 12 lobe-level ratios (4 lobes × 3 subregions) |
| `OS_months` | float | Overall survival converted from days |
| `lobe_assignment_reliable` | bool | True if ≥90% of tumor voxels have a lobe label |

---

## QA Column

`lobe_assignment_reliable` = `True` when ≥90% of tumor voxels fall within a non-zero lobe region of the registered atlas. Patients with `False` are dropped in downstream preprocessing.

---

## Fallback Mode

Set `atlas.use_ants_registration: false` in `config.json` to use affine-based nearest-neighbor resampling via NIfTI headers. This assumes SRI24 and patient data share world coordinates (which is generally **not** true for UCSF-PDGM). Use only for testing or when ANTs is unavailable.

---

## Resume + Cache Performance

- Transform cache is scanned **once** at startup; membership checks are O(1) in-memory set lookups.
- CSV resume reads already-processed patient IDs at startup; skipped patients print `[SKIP] Already processed`.
- Crash-safe: each row is flushed to disk immediately after extraction (`csv_file.flush()`).
