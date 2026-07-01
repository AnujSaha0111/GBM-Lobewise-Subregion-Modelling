#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# ── Constants ──

MODALITIES = ("t1", "t2", "t1gd", "flair")

FEATURE_COLS = [
    "global_nc_en_ratio",
    "global_ed_en_ratio",
    "global_ed_total_ratio",
    "tumor_burden_index",
    *(f"{lb}_{sub}_ratio" for lb in ("frontal", "temporal", "parietal", "occipital")
      for sub in ("ed", "en", "nc")),
]

METADATA_COLS = ["patient_id", "OS_months", "lobe_assignment_reliable"]

DEFAULT_INPUTS = {mod: f"outputs/features_raw_{mod}.csv" for mod in MODALITIES}
DEFAULT_OUTPUT = "outputs/multimodal_lobewise/merged_features.csv"
DEFAULT_METADATA = "outputs/multimodal_lobewise/metadata_processed.csv"
DEFAULT_MERGED_OUTPUT = "outputs/multimodal_lobewise/merged_features_with_metadata.csv"
DEFAULT_SUMMARY_OUTPUT = "outputs/multimodal_lobewise/feature_sets_summary.json"

METADATA_FEATURE_COLS = ["age", "sex", "idh", "mgmt", "who_grade", "eor"]


# ── Helpers ──

def _load_config(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _resolve_path(root: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (root / p)


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int) != 0
    values = series.astype(str).str.strip().str.lower()
    return values.isin(["true", "1", "yes", "y"])


def _load_and_filter(modality: str, path: Path) -> pd.DataFrame:
    """Load one modality CSV, select columns, filter unreliable/missing rows."""
    df = pd.read_csv(path)
    keep_cols = METADATA_COLS + FEATURE_COLS
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Modality '{modality}' missing columns: {missing}")

    df = df[keep_cols].copy()

    # Filter unreliable lobe assignments
    reliable = _coerce_bool_series(df["lobe_assignment_reliable"])
    df = df[reliable].copy()

    # Filter missing / invalid OS_months
    os_numeric = pd.to_numeric(df["OS_months"], errors="coerce")
    df = df[os_numeric.notna()].copy()
    df["OS_months"] = os_numeric[os_numeric.notna()]

    # Rename feature columns with modality prefix, e.g. frontal_nc_ratio -> T1_frontal_nc_ratio
    prefix = modality.upper()
    df = df.rename(columns={col: f"{prefix}_{col}" for col in FEATURE_COLS})

    return df


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Arrange columns: metadata first, then features grouped by modality."""
    meta = ["patient_id", "OS_months", "lobe_assignment_reliable", "risk_label"]
    feature_cols = [c for c in df.columns if c not in meta]
    ordered = meta + feature_cols
    return df[[c for c in ordered if c in df.columns]]


# ── Public Functions ──

def build_multimodal_dataset(
    in_csvs: dict[str, Path],
    out_csv: Path,
    cfg: dict,
    metadata_csv: Path | None = None,
    merged_out_csv: Path | None = None,
    summary_out: Path | None = None,
) -> int:
    """Load, filter, prefix, merge, and save the multimodal dataset.

    Optionally merges processed clinical/molecular metadata and generates
    a feature-sets summary.
    """
    threshold = float(cfg["preprocessing"]["os_high_risk_threshold_months"])

    # Load and filter each modality independently
    frames: dict[str, pd.DataFrame] = {}
    for mod in MODALITIES:
        frames[mod] = _load_and_filter(mod, in_csvs[mod])

    # Start with the first modality as the metadata reference
    ref_mod = MODALITIES[0]
    merged = frames[ref_mod].copy()

    # Assign binary risk label
    merged["risk_label"] = (merged["OS_months"] <= threshold).astype(int)

    # Merge remaining modalities on patient_id (INNER JOIN)
    for mod in MODALITIES[1:]:
        other = frames[mod]
        other_cols = ["patient_id"] + [f"{mod.upper()}_{c}" for c in FEATURE_COLS]
        merged = merged.merge(other[other_cols], on="patient_id", how="inner")

    merged = _reorder_columns(merged)

    # Persist spatial-only dataset
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)

    # Summary statistics
    metadata_cols = {"patient_id", "OS_months", "lobe_assignment_reliable", "risk_label"}
    final_feature_cols = [c for c in merged.columns if c not in metadata_cols]

    print(f"Total patients retained: {len(merged)}")
    print(f"Class distribution:\n{merged['risk_label'].value_counts().to_string()}")

    missing = merged.isnull().sum()
    missing_exist = missing[missing > 0]
    if len(missing_exist) > 0:
        print(f"Missing value counts:\n{missing_exist.to_string()}")
    else:
        print("Missing value counts: None")

    print(f"Final feature dimension: {len(final_feature_cols)}")

    # ── Metadata merge (STEP 2) ──
    if metadata_csv is not None and metadata_csv.exists():
        meta_df = pd.read_csv(metadata_csv)
        meta_df["patient_id"] = meta_df["patient_id"].astype(str).str.strip()

        merged_with_meta = merged.merge(meta_df, on="patient_id", how="left")

        n_pre = len(merged)
        n_post = len(merged_with_meta)
        n_meta_matched = merged_with_meta["age"].notna().sum()

        print(f"\nMetadata merge:")
        print(f"  Patients before merge: {n_pre}")
        print(f"  Patients after merge: {n_post}")
        print(f"  Patients with metadata match: {n_meta_matched}/{n_pre}")
        print(f"  Patients without metadata: {n_pre - n_meta_matched}")

        if merged_out_csv is not None:
            merged_out_csv.parent.mkdir(parents=True, exist_ok=True)
            merged_with_meta.to_csv(merged_out_csv, index=False)
            print(f"Wrote {len(merged_with_meta)} rows -> {merged_out_csv}")

        # ── Feature sets summary (STEP 3 & 4) ──
        if summary_out is not None:
            _write_feature_sets_summary(
                spatial_df=merged,
                merged_df=merged_with_meta,
                spatial_cols=final_feature_cols,
                summary_path=summary_out,
            )

    return 0


def _write_feature_sets_summary(
    spatial_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    spatial_cols: list[str],
    summary_path: Path,
) -> None:
    """Compute and persist summary stats for the three feature sets."""
    n_patients = len(spatial_df)

    risk_counts = spatial_df["risk_label"].value_counts().to_dict()

    # Feature set A: spatial only (64 features, no missing after median imputation upstream)
    a_missing = int(spatial_df[spatial_cols].isnull().sum().sum())
    a_cols = len(spatial_cols)

    # Feature set B: clinical + molecular only
    meta_feature_cols = [c for c in METADATA_FEATURE_COLS if c in merged_df.columns]
    b_cols = len(meta_feature_cols)
    b_missing = int(merged_df[meta_feature_cols].isnull().sum().sum())

    # Feature set C: combined
    combined_cols = spatial_cols + meta_feature_cols
    c_cols = len(combined_cols)
    c_missing = int(merged_df[combined_cols].isnull().sum().sum())

    summary = {
        "feature_set": {
            "A_spatial_only": {
                "number_of_patients": n_patients,
                "number_of_features": a_cols,
                "feature_names": spatial_cols,
                "missingness": a_missing,
                "class_distribution": risk_counts,
            },
            "B_clinical_molecular_only": {
                "number_of_patients": n_patients,
                "number_of_features": b_cols,
                "feature_names": meta_feature_cols,
                "missingness": b_missing,
                "class_distribution": risk_counts,
            },
            "C_combined": {
                "number_of_patients": n_patients,
                "number_of_features": c_cols,
                "feature_names": combined_cols,
                "missingness": c_missing,
                "class_distribution": risk_counts,
            },
        },
        "metadata_merge": {
            "spatial_patients": n_patients,
            "metadata_source_patients": int(
                merged_df["age"].notna().sum()
            ),
            "missing_metadata": int(
                merged_df["age"].isna().sum()
            ),
        },
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote feature sets summary -> {summary_path}")
    print(f"  A (spatial): {a_cols} features, {a_missing} missing")
    print(f"  B (clinical): {b_cols} features, {b_missing} missing")
    print(f"  C (combined): {c_cols} features, {c_missing} missing")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build multimodal lobewise-subregion dataset from modality-specific CSVs.",
    )
    parser.add_argument(
        "--config", default="config.json",
        help="Path to config.json (default: config.json)",
    )
    parser.add_argument(
        "--output", default=None,
        help=f"Spatial-only CSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--metadata", default=None,
        help=f"Processed metadata CSV (default: {DEFAULT_METADATA})",
    )
    parser.add_argument(
        "--merged-output", default=None,
        help=f"CSV with metadata merged (default: {DEFAULT_MERGED_OUTPUT})",
    )
    parser.add_argument(
        "--summary", default=None,
        help=f"Feature sets summary JSON (default: {DEFAULT_SUMMARY_OUTPUT})",
    )
    for mod in MODALITIES:
        parser.add_argument(
            f"--{mod}", default=None,
            help=f"{mod.upper()} features CSV (default: {DEFAULT_INPUTS[mod]})",
        )

    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    cfg = _load_config(_resolve_path(root, args.config))

    in_csvs: dict[str, Path] = {}
    for mod in MODALITIES:
        raw = getattr(args, mod)
        p = _resolve_path(root, raw or DEFAULT_INPUTS[mod])
        if not p.exists():
            raise FileNotFoundError(f"Input CSV for '{mod}' not found: {p}")
        in_csvs[mod] = p

    out_csv = _resolve_path(root, args.output or DEFAULT_OUTPUT)
    metadata_csv = _resolve_path(root, args.metadata or DEFAULT_METADATA)
    merged_out_csv = _resolve_path(root, args.merged_output or DEFAULT_MERGED_OUTPUT)
    summary_out = _resolve_path(root, args.summary or DEFAULT_SUMMARY_OUTPUT)

    return build_multimodal_dataset(
        in_csvs, out_csv, cfg,
        metadata_csv=metadata_csv,
        merged_out_csv=merged_out_csv,
        summary_out=summary_out,
    )


if __name__ == "__main__":
    raise SystemExit(main())
