#!/usr/bin/env python3
"""Cohort Exclusion Audit

Reads existing CSVs (no modification to any pipeline code) and produces a
CONSORT-style cohort exclusion audit with exact counts.

Pipeline stages reconstructed from existing files:
  1. UCSF-PDGM-metadata_v5.csv       → original cohort
  2. features_raw_*.csv                → post-atlas registration (modality discovery)
  3. features_raw.csv                  → post-atlas, single reference modality
  4. features_processed.csv            → post-preprocessing (reliable + OS filter)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "experiments" / "statistics" / "cohort_audit" / "results"


def _normalize_id(pid: str) -> str:
    """Zero-pad numeric portion of UCSF-PDGM IDs to 4 digits."""
    import re
    pid = str(pid).strip()
    m = re.match(r"^(UCSF-PDGM-)(\d+)(.*)", pid)
    if m:
        return f"{m.group(1)}{int(m.group(2)):04d}{m.group(3)}"
    return pid


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int) != 0
    values = series.astype(str).str.strip().str.lower()
    return values.isin(["true", "1", "yes", "y"])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    # ── Stage 1: Original cohort from metadata ──────────────────────────
    meta_path = ROOT / "UCSF-PDGM-metadata_v5.csv"
    meta = pd.read_csv(meta_path)
    meta_ids = set(meta["ID"].apply(_normalize_id))
    n_original = len(meta)
    results["stage_1_original_cohort"] = {
        "source": str(meta_path.relative_to(ROOT)),
        "n_patients": n_original,
    }

    # ── Stage 2: Post-atlas registration (feature extraction) ───────────
    # All 4 modality-specific raw feature CSVs have the same patient set.
    mod_files = {
        "T1": ROOT / "outputs" / "features_raw_t1.csv",
        "T1GD": ROOT / "outputs" / "features_raw_t1gd.csv",
        "T2": ROOT / "outputs" / "features_raw_t2.csv",
        "FLAIR": ROOT / "outputs" / "features_raw_flair.csv",
    }
    mod_counts = {}
    mod_ids = {}
    for mod, path in mod_files.items():
        df = pd.read_csv(path)
        mod_counts[mod] = len(df)
        mod_ids[mod] = set(df["patient_id"].apply(_normalize_id))

    # All modalities should have the same patient set (inner-joined later)
    assert len(set.intersection(*mod_ids.values())) == len(mod_ids["T1"]), \
        "Modality CSVs have inconsistent patient sets"
    raw_ids = mod_ids["T1"]
    n_raw = mod_counts["T1"]
    n_lost_atlas = n_original - n_raw

    results["stage_2_post_atlas_registration"] = {
        "modality_patient_counts": mod_counts,
        "all_modalities_consistent": len(set.intersection(*mod_ids.values())) == n_raw,
        "n_patients": n_raw,
        "n_lost_from_original": n_lost_atlas,
    }

    # Identify patients lost during atlas registration
    lost_atlas_ids = sorted(meta_ids - raw_ids)
    results["stage_2_patients_lost"] = lost_atlas_ids

    # ── Stage 3: Preprocessing filters (from features_raw.csv) ──────────
    raw_path = ROOT / "outputs" / "features_raw.csv"
    raw_df = pd.read_csv(raw_path)

    # Filter 1: unreliable lobe assignment
    reliable_mask = _coerce_bool(raw_df["lobe_assignment_reliable"])
    n_unreliable = int((~reliable_mask).sum())
    unreliable_ids = sorted(
        raw_df.loc[~reliable_mask, "patient_id"].apply(_normalize_id).tolist()
    )

    # Filter 2: missing OS_months (NaN after numeric coercion)
    os_numeric = pd.to_numeric(raw_df["OS_months"], errors="coerce")
    n_missing_os = int(os_numeric.isna().sum())
    missing_os_ids = sorted(
        raw_df.loc[os_numeric.isna(), "patient_id"].apply(_normalize_id).tolist()
    )

    # Filter 3: non-positive OS_months (≤ 0)
    os_valid = os_numeric.dropna()
    n_nonpositive_os = int((os_valid <= 0).sum())
    nonpositive_os_ids = sorted(
        raw_df.loc[os_valid[os_valid <= 0].index, "patient_id"]
        .apply(_normalize_id).tolist()
    )

    # Overlap: patients with BOTH unreliable and missing OS
    unreliable_set = set(unreliable_ids)
    missing_os_set = set(missing_os_ids)
    nonpositive_os_set = set(nonpositive_os_ids)
    both_unreliable_and_missing = sorted(unreliable_set & missing_os_set)
    n_both = len(both_unreliable_and_missing)

    # All exclusion reasons combined (OR)
    all_excluded_set = unreliable_set | missing_os_set | nonpositive_os_set
    n_total_excluded_stage3 = len(all_excluded_set)

    # Patients excluded with unreliable lobe assignment only (not missing OS)
    unreliable_only = sorted(unreliable_set - missing_os_set - nonpositive_os_set)
    n_unreliable_only = len(unreliable_only)

    # Patients excluded with missing OS only (not unreliable)
    missing_os_only = sorted(missing_os_set - unreliable_set - nonpositive_os_set)
    n_missing_os_only = len(missing_os_only)

    # Patients excluded with non-positive OS only
    nonpositive_only = sorted(nonpositive_os_set - unreliable_set - missing_os_set)
    n_nonpositive_only = len(nonpositive_only)

    results["stage_3_preprocessing_filters"] = {
        "input_n_patients": n_raw,
        "filter_unreliable_lobe": {
            "description": "lobe_assignment_reliable == False (atlas coverage < 90%)",
            "n_excluded": n_unreliable,
            "patient_ids": unreliable_ids,
        },
        "filter_missing_os": {
            "description": "OS_months is NaN (missing or non-numeric survival)",
            "n_excluded": n_missing_os,
            "patient_ids": missing_os_ids,
        },
        "filter_nonpositive_os": {
            "description": "OS_months <= 0 (non-positive survival)",
            "n_excluded": n_nonpositive_os,
            "patient_ids": nonpositive_os_ids,
        },
        "overlap_unreliable_and_missing_os": {
            "description": "Patients excluded for BOTH unreliable lobe AND missing OS",
            "n_overlap": n_both,
            "patient_ids": both_unreliable_and_missing,
        },
        "unique_exclusions_combined": {
            "n_excluded": n_total_excluded_stage3,
            "n_unreliable_only": n_unreliable_only,
            "n_missing_os_only": n_missing_os_only,
            "n_nonpositive_only": n_nonpositive_only,
            "n_both_unreliable_and_missing_os": n_both,
        },
    }

    # ── Stage 4: Final analysed cohort ──────────────────────────────────
    processed_path = ROOT / "outputs" / "features_processed.csv"
    processed_df = pd.read_csv(processed_path)
    n_final = len(processed_path.read_text().splitlines()) - 1  # subtract header

    # Verify: features_processed should have risk_label column, no patient_id
    has_risk_label = "risk_label" in processed_df.columns
    has_patient_id = "patient_id" in processed_df.columns
    has_lobe_reliable = "lobe_assignment_reliable" in processed_df.columns

    results["stage_4_final_cohort"] = {
        "source": str(processed_path.relative_to(ROOT)),
        "n_patients": n_final,
        "has_risk_label": has_risk_label,
        "has_patient_id": has_patient_id,
        "has_lobe_assignment_reliable": has_lobe_reliable,
    }

    # ── Balance verification ────────────────────────────────────────────
    expected_final = n_raw - n_total_excluded_stage3
    balance_ok = expected_final == n_final

    # Also check: preprocessing.py only filters unreliable + missing OS (not non-positive)
    # In preprocessing.py, the filter is: os_numeric.isna() only
    # Non-positive OS is NOT filtered in preprocessing.py
    # But features_processed.csv has n_final patients. Let's verify.
    # The features_processed.csv is the output of preprocessing.py which was run
    # with specific inputs. Let's check if non-positive OS patients exist in raw.
    n_excluded_by_code = n_unreliable + n_missing_os  # what preprocessing.py actually filters
    expected_by_code = n_raw - n_excluded_by_code

    results["balance_verification"] = {
        "n_original": n_original,
        "n_post_atlas": n_raw,
        "n_lost_atlas_discovery": n_lost_atlas,
        "n_excluded_unreliable": n_unreliable,
        "n_excluded_missing_os": n_missing_os,
        "n_excluded_nonpositive_os": n_nonpositive_os,
        "n_total_excluded_unique": n_total_excluded_stage3,
        "n_final_features_processed": n_final,
        "balance_check_raw_minus_unique_exclusions": {
            "expected_final": expected_final,
            "actual_final": n_final,
            "balanced": balance_ok,
        },
        "balance_check_raw_minus_code_filters": {
            "description": "preprocessing.py filters unreliable + missing OS (NOT non-positive)",
            "n_excluded_by_code": n_excluded_by_code,
            "expected_final": expected_by_code,
            "actual_final": n_final,
            "balanced": expected_by_code == n_final,
        },
    }

    # ── OS_months distribution check ────────────────────────────────────
    os_values = pd.to_numeric(raw_df["OS_months"], errors="coerce").dropna()
    results["os_months_distribution"] = {
        "n_valid_os": int(len(os_values)),
        "min": float(os_values.min()),
        "max": float(os_values.max()),
        "mean": float(os_values.mean()),
        "median": float(os_values.median()),
        "n_nonpositive": int((os_values <= 0).sum()),
        "n_zero": int((os_values == 0).sum()),
        "n_negative": int((os_values < 0).sum()),
    }

    # ── Multimodal pipeline cross-check ─────────────────────────────────
    merged_path = ROOT / "outputs" / "multimodal_lobewise" / "merged_features_with_metadata.csv"
    if merged_path.exists():
        merged_df = pd.read_csv(merged_path)
        n_merged = len(merged_df)
        results["multimodal_pipeline_crosscheck"] = {
            "source": str(merged_path.relative_to(ROOT)),
            "n_patients": n_merged,
            "matches_features_processed": n_merged == n_final,
        }

    # ── Radiogenomic analysis cross-check ───────────────────────────────
    # The radiogenomic analysis loads from features_raw_*.csv, merges all 4
    # modalities (inner join), then filters on lobe_assignment_reliable.
    # It does NOT filter on OS_months directly.
    # Let's simulate this to verify the patient count.
    SPATIAL_COLS = [
        "global_nc_en_ratio", "global_ed_en_ratio", "global_ed_total_ratio",
        "tumor_burden_index",
        "frontal_ed_ratio", "frontal_en_ratio", "frontal_nc_ratio",
        "temporal_ed_ratio", "temporal_en_ratio", "temporal_nc_ratio",
        "parietal_ed_ratio", "parietal_en_ratio", "parietal_nc_ratio",
        "occipital_ed_ratio", "occipital_en_ratio", "occipital_nc_ratio",
    ]

    rad_mod_dfs = {}
    for mod, path in mod_files.items():
        df = pd.read_csv(path)
        rename_map = {col: f"{mod}_{col}" for col in SPATIAL_COLS}
        df = df.rename(columns=rename_map)
        prefixed = [f"{mod}_{col}" for col in SPATIAL_COLS]
        rad_mod_dfs[mod] = df[["patient_id"] + prefixed]

    rad_merged = rad_mod_dfs["T1"][["patient_id"]].copy()
    for mod, df_mod in rad_mod_dfs.items():
        feat_cols = [c for c in df_mod.columns if c != "patient_id"]
        rad_merged = rad_merged.merge(
            df_mod[["patient_id"] + feat_cols],
            on="patient_id", how="inner",
        )

    # Add lobe_assignment_reliable from features_raw.csv
    reliable_col = raw_df[["patient_id", "lobe_assignment_reliable"]].copy()
    rad_merged = rad_merged.merge(reliable_col, on="patient_id", how="left")

    # Filter reliable
    rad_reliable = _coerce_bool(rad_merged["lobe_assignment_reliable"])
    n_rad_before_filter = len(rad_merged)
    n_rad_after_filter = int(rad_reliable.sum())

    results["radiogenomic_analysis_crosscheck"] = {
        "n_before_reliable_filter": n_rad_before_filter,
        "n_after_reliable_filter": n_rad_after_filter,
        "description": (
            "Radiogenomic analysis loads all 4 modalities, inner-joins on "
            "patient_id, then filters on lobe_assignment_reliable only "
            "(NOT on OS_months)."
        ),
    }

    # ── Write JSON output ───────────────────────────────────────────────
    json_path = OUT_DIR / "cohort_exclusion_audit.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Wrote {json_path}")

    # ── Write Markdown report ───────────────────────────────────────────
    md_lines = []
    md_lines.append("# Cohort Exclusion Audit")
    md_lines.append("")
    md_lines.append("Reconstructed from existing data pipeline outputs. No files modified.")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    md_lines.append("## CONSORT-Style Flow Summary")
    md_lines.append("")
    md_lines.append(f"```")
    md_lines.append(f"UCSF-PDGM original cohort                        n = {n_original}")
    md_lines.append(f"")
    md_lines.append(f"  Exclusion 1: No modality images / seg on disk    n = {n_lost_atlas}")
    md_lines.append(f"    (patients discovered by filesystem scan of")
    md_lines.append(f"     DATA-IMAGE-STRUCTURAL + DATA-AUTOMATED-SEG)")
    md_lines.append(f"")
    md_lines.append(f"  Post-atlas cohort                               n = {n_raw}")
    md_lines.append(f"")
    md_lines.append(f"  Exclusion 2: Unreliable lobe assignment          n = {n_unreliable}")
    md_lines.append(f"    (atlas coverage < 90% of whole-tumor voxels)")
    md_lines.append(f"")
    md_lines.append(f"  Exclusion 3: Missing OS_months                  n = {n_missing_os}")
    md_lines.append(f"    (NaN after numeric coercion)")
    md_lines.append(f"")
    if n_nonpositive_os > 0:
        md_lines.append(f"  Exclusion 4: Non-positive OS_months              n = {n_nonpositive_os}")
        md_lines.append(f"    (OS_months <= 0)")
        md_lines.append(f"")
    md_lines.append(f"  Final analysed cohort (features_processed.csv)  n = {n_final}")
    md_lines.append(f"```")
    md_lines.append("")

    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## Detailed Counts")
    md_lines.append("")
    md_lines.append("| Stage | Source | n_patients |")
    md_lines.append("|-------|--------|------------|")
    md_lines.append(f"| 1. Original cohort | UCSF-PDGM-metadata_v5.csv | {n_original} |")
    md_lines.append(f"| 2. Post-atlas registration | features_raw_*.csv | {n_raw} |")
    md_lines.append(f"| 3. Post-preprocessing | features_processed.csv | {n_final} |")
    md_lines.append(f"| Multimodal merge | merged_features_with_metadata.csv | {results.get('multimodal_pipeline_crosscheck', {}).get('n_patients', 'N/A')} |")
    md_lines.append("")

    md_lines.append("## Exclusion Breakdown")
    md_lines.append("")
    md_lines.append("### Stage 1 → 2: Atlas Registration Discovery")
    md_lines.append("")
    md_lines.append(f"- **Lost**: {n_lost_atlas} patients")
    md_lines.append(f"- **Reason**: `discover_patients()` in `atlas_registration.py` scans")
    md_lines.append(f"  `DATA-IMAGE-STRUCTURAL/<pid>/<pid>_<mod>.nii.gz` and")
    md_lines.append(f"  `DATA-AUTOMATED-SEG/<pid>/<pid>_tumor_segmentation.nii.gz`.")
    md_lines.append(f"  Patients missing either file are excluded.")
    if lost_atlas_ids:
        md_lines.append(f"- **Patient IDs**: {', '.join(lost_atlas_ids)}")
    else:
        md_lines.append(f"- **Patient IDs**: (none identified — IDs may differ in raw CSV)")
    md_lines.append("")

    md_lines.append("### Stage 2 → 3: Preprocessing Filters")
    md_lines.append("")
    md_lines.append(f"#### Filter A: Unreliable Lobe Assignment")
    md_lines.append(f"- **Threshold**: < 90% of whole-tumor voxels mapped to SRI24 atlas lobes")
    md_lines.append(f"- **Count**: {n_unreliable} patients excluded")
    md_lines.append(f"- **Computed in**: `atlas_registration.py:440` — `reliable = (mapped_vox / wt_total) >= 0.90`")
    md_lines.append(f"- **Filtered in**: `preprocessing.py:72-74`")
    if unreliable_ids:
        md_lines.append(f"- **Patient IDs**: {', '.join(unreliable_ids)}")
    md_lines.append("")

    md_lines.append(f"#### Filter B: Missing OS_months")
    md_lines.append(f"- **Count**: {n_missing_os} patients excluded")
    md_lines.append(f"- **Computed in**: `atlas_registration.py:468-488` — `load_clinical_os()` converts")
    md_lines.append(f"  OS days to months; NaN if OS column is empty.")
    md_lines.append(f"- **Filtered in**: `preprocessing.py:76-79` — `pd.to_numeric(errors='coerce').isna()`")
    if missing_os_ids:
        md_lines.append(f"- **Patient IDs**: {', '.join(missing_os_ids)}")
    md_lines.append("")

    if n_nonpositive_os > 0:
        md_lines.append(f"#### Filter C: Non-positive OS_months")
        md_lines.append(f"- **Count**: {n_nonpositive_os} patients excluded")
        md_lines.append(f"- **Note**: `preprocessing.py` does NOT explicitly filter non-positive OS.")
        md_lines.append(f"  These patients have OS_months <= 0 but are included in features_raw.csv.")
        md_lines.append(f"  They were excluded during an earlier run of preprocessing.py that")
        md_lines.append(f"  may have had additional filtering, OR they were excluded during the")
        md_lines.append(f"  atlas registration step (non-positive OS → assigned NaN).")
        if nonpositive_os_ids:
            md_lines.append(f"- **Patient IDs**: {', '.join(nonpositive_os_ids)}")
        md_lines.append("")

    md_lines.append(f"#### Overlap Between Filters")
    md_lines.append(f"- Patients with BOTH unreliable lobe AND missing OS: {n_both}")
    if both_unreliable_and_missing:
        md_lines.append(f"- **Patient IDs**: {', '.join(both_unreliable_and_missing)}")
    md_lines.append(f"- Patients excluded for unreliable lobe ONLY: {n_unreliable_only}")
    md_lines.append(f"- Patients excluded for missing OS ONLY: {n_missing_os_only}")
    if n_nonpositive_os > 0:
        md_lines.append(f"- Patients excluded for non-positive OS ONLY: {n_nonpositive_only}")
    md_lines.append("")

    md_lines.append("## Balance Verification")
    md_lines.append("")
    md_lines.append(f"| Calculation | Expected | Actual | Balanced? |")
    md_lines.append(f"|-------------|----------|--------|-----------|")
    bd = results["balance_verification"]
    bc1 = bd["balance_check_raw_minus_unique_exclusions"]
    bc2 = bd["balance_check_raw_minus_code_filters"]
    md_lines.append(f"| Post-atlas minus unique exclusions | {bc1['expected_final']} | {bc1['actual_final']} | **{'YES' if bc1['balanced'] else 'NO'}** |")
    md_lines.append(f"| Post-atlas minus code filters (unreliable + missing OS) | {bc2['expected_final']} | {bc2['actual_final']} | **{'YES' if bc2['balanced'] else 'NO'}** |")
    md_lines.append("")

    md_lines.append("## Cross-Checks")
    md_lines.append("")
    mcp = results.get("multimodal_pipeline_crosscheck", {})
    if mcp:
        md_lines.append(f"- **Multimodal merge** (`merged_features_with_metadata.csv`): {mcp['n_patients']} patients — {'matches' if mcp['matches_features_processed'] else 'DOES NOT MATCH'} features_processed.csv")
    rcp = results["radiogenomic_analysis_crosscheck"]
    md_lines.append(f"- **Radiogenomic analysis** (all 4 modalities inner-joined + reliable filter): {rcp['n_after_reliable_filter']} patients")
    md_lines.append(f"  - {rcp['description']}")
    md_lines.append("")

    osd = results["os_months_distribution"]
    md_lines.append("## OS_months Distribution (features_raw.csv, valid values only)")
    md_lines.append("")
    md_lines.append(f"- n = {osd['n_valid_os']}")
    md_lines.append(f"- Range: [{osd['min']:.2f}, {osd['max']:.2f}] months")
    md_lines.append(f"- Mean: {osd['mean']:.2f}, Median: {osd['median']:.2f}")
    md_lines.append(f"- Non-positive: {osd['n_nonpositive']} (zero: {osd['n_zero']}, negative: {osd['n_negative']})")
    md_lines.append("")

    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## Recommended Manuscript Wording")
    md_lines.append("")
    wording = (
        f"Of the {n_original} patients in the UCSF-PDGM cohort, "
        f"{n_lost_atlas} were excluded during atlas registration "
        f"(missing imaging or segmentation data), and "
        f"{n_total_excluded_stage3} were excluded during preprocessing "
        f"({n_unreliable} with unreliable lobe assignment "
        f"[atlas coverage < 90%], "
        f"{n_missing_os} with missing survival data"
    )
    if n_nonpositive_os > 0:
        wording += f", {n_nonpositive_os} with non-positive survival"
    wording += (
        f"), yielding a final analysed cohort of "
        f"**n = {n_final}** patients."
    )
    md_lines.append(f"> {wording}")
    md_lines.append("")

    md_path = OUT_DIR / "cohort_exclusion_audit.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Wrote {md_path}")

    # ── Summary to stdout ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COHORT EXCLUSION AUDIT SUMMARY")
    print("=" * 60)
    print(f"  Original cohort:       n = {n_original}")
    print(f"  Lost (atlas):          n = {n_lost_atlas}")
    print(f"  Post-atlas:            n = {n_raw}")
    print(f"  Unreliable lobe:       n = {n_unreliable}")
    print(f"  Missing OS:            n = {n_missing_os}")
    print(f"  Non-positive OS:       n = {n_nonpositive_os}")
    print(f"  Unique exclusions:     n = {n_total_excluded_stage3}")
    print(f"  Final analysed:        n = {n_final}")
    print(f"  Balance check:         {'PASS' if bc1['balanced'] else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
