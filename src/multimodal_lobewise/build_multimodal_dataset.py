#!/usr/bin/env python3
"""Build a multimodal lobewise-subregion dataset by merging modality-specific feature CSVs.

Loads features_raw_{t1,t2,t1gd,flair}.csv, filters unreliable rows, renames feature
columns with modality prefixes, inner-joins on patient_id, and writes the merged dataset
to outputs/multimodal_lobewise/merged_features.csv.
"""

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

def build_multimodal_dataset(in_csvs: dict[str, Path], out_csv: Path, cfg: dict) -> int:
    """Load, filter, prefix, merge, and save the multimodal dataset."""
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

    # Persist
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
    return 0


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
        help=f"Output CSV (default: {DEFAULT_OUTPUT})",
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
    return build_multimodal_dataset(in_csvs, out_csv, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
