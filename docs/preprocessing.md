# Preprocessing

This step converts a raw features CSV (output of `atlas_registration.py`) into a
clean training table with binary risk labels.

---

## What It Does

1. **Drops unreliable rows** — removes patients where `lobe_assignment_reliable = False`
2. **Drops missing OS** — removes patients with null or non-numeric `OS_months`
3. **Assigns binary risk labels:**
   - `risk_label = 1` if `OS_months <= 12` (high-risk)
   - `risk_label = 0` otherwise
4. **Imputes missing feature values** with column medians (tree models only)
5. **Leaves features unscaled** by default — tree models do not require scaling

---

## Supported Modalities

Run preprocessing independently for each modality. Each modality has its own
raw input CSV and produces its own processed output CSV — results are never mixed.

| Modality | Input CSV | Output CSV |
|----------|-----------|------------|
| T1 | `outputs/features_raw_t1.csv` | `outputs/features_processed_t1.csv` |
| T2 | `outputs/features_raw_t2.csv` | `outputs/features_processed_t2.csv` |
| T1GD | `outputs/features_raw_t1gd.csv` | `outputs/features_processed_t1gd.csv` |
| FLAIR | `outputs/features_raw_flair.csv` | `outputs/features_processed_flair.csv` |

---

## CLI Usage

### T1 (default)
```bash
python src/preprocessing.py \
  --input  outputs/features_raw_t1.csv \
  --output outputs/features_processed_t1.csv
```

### T2
```bash
python src/preprocessing.py \
  --input  outputs/features_raw_t2.csv \
  --output outputs/features_processed_t2.csv
```

### T1GD
```bash
python src/preprocessing.py \
  --input  outputs/features_raw_t1gd.csv \
  --output outputs/features_processed_t1gd.csv
```

### FLAIR
```bash
python src/preprocessing.py \
  --input  outputs/features_raw_flair.csv \
  --output outputs/features_processed_flair.csv
```

### All 4 in sequence (PowerShell)
```powershell
foreach ($mod in @("t1","t2","t1gd","flair")) {
    python src/preprocessing.py `
        --input  "outputs/features_raw_$mod.csv" `
        --output "outputs/features_processed_$mod.csv"
}
```

### All 4 in sequence (bash)
```bash
for mod in t1 t2 t1gd flair; do
    python src/preprocessing.py \
        --input  "outputs/features_raw_${mod}.csv" \
        --output "outputs/features_processed_${mod}.csv"
done
```

---

## All CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `config.json` | Path to config.json (reads `os_high_risk_threshold_months`) |
| `--input` | `outputs/features_raw.csv` | Raw features CSV to process |
| `--output` | `outputs/features_processed.csv` | Output path for processed CSV |
| `--scale` | off | Apply StandardScaler (not part of default workflow) |
| `--confirm-scale` | off | Required guard flag when using `--scale` |

---

## Output Schema

The output CSV contains exactly **16 feature columns + 1 label column** (17 total).
`OS_months` and `patient_id` are deliberately excluded from the training table.

| Column | Type | Description |
|--------|------|-------------|
| `global_nc_en_ratio` | float | NC / ET voxel ratio |
| `global_ed_en_ratio` | float | ED / ET voxel ratio |
| `global_ed_total_ratio` | float | ED / WT voxel ratio |
| `tumor_burden_index` | float | WT / brain voxel ratio |
| `{lobe}_{sub}_ratio` × 12 | float | Lobe invasion fractions (4 lobes × 3 subregions) |
| `risk_label` | int (0/1) | `1` = high-risk (OS ≤ 12 mo), `0` = low-risk |

---

## Notes

- `OS_months` is **never** included as a feature — it is only used to derive `risk_label`.
- The 90% lobe-mapping threshold (`lobe_assignment_reliable`) is set in `atlas_registration.py`,
  not here — patients failing it are already flagged before this step.
- Scaling is optional and intended only for comparison workflows (e.g., testing SVM baselines).
  The default XGBoost pipeline does not require it.
- Each modality's processed CSV is independent — you can preprocess and train them in
  any order or in parallel.
