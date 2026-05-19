#!/usr/bin/env python3
"""Preprocess features_raw.csv into features_processed.csv.

Filters unreliable rows, assigns binary risk labels, and imputes missing
feature values with column medians.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "global_nc_en_ratio",
    "global_ed_en_ratio",
    "global_ed_total_ratio",
    "tumor_burden_index",
    *(f"{lb}_{sub}_ratio" for lb in ("frontal", "temporal", "parietal", "occipital")
      for sub in ("ed", "en", "nc")),
]

REQUIRED_COLS = [
    "patient_id",
    "OS_months",
    "lobe_assignment_reliable",
    *FEATURE_COLS,
]


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


def _print_id_list(label: str, ids: list[str]) -> None:
    if not ids:
        return
    print(label)
    print(", ".join(ids))


def preprocess(in_csv: Path, out_csv: Path, cfg: dict, scale: bool) -> int:
    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    df = pd.read_csv(in_csv)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input CSV: {missing}")

    start_rows = len(df)
    id_col = "patient_id"

    reliable_mask = _coerce_bool_series(df["lobe_assignment_reliable"])
    dropped_unreliable = df[~reliable_mask]
    df = df[reliable_mask].copy()

    os_numeric = pd.to_numeric(df["OS_months"], errors="coerce")
    missing_os_mask = os_numeric.isna()
    dropped_missing_os = df[missing_os_mask]
    df = df[~missing_os_mask].copy()
    df["OS_months"] = os_numeric[~missing_os_mask]

    threshold = float(cfg["preprocessing"]["os_high_risk_threshold_months"])
    df["risk_label"] = (df["OS_months"] <= threshold).astype(int)

    features = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    medians = features.median()
    if medians.isna().any():
        print("WARNING: Some feature columns are all-NaN after filtering.")
    features = features.fillna(medians)

    if scale:
        scaler = StandardScaler()
        features = pd.DataFrame(
            scaler.fit_transform(features),
            columns=FEATURE_COLS,
            index=features.index,
        )

    out_df = pd.concat([features, df["risk_label"]], axis=1)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    removed = start_rows - len(out_df)
    print(f"Loaded {start_rows} rows from {in_csv}")
    print(f"Kept {len(out_df)} rows after filtering (removed {removed})")
    print(f"  Unreliable lobe assignment: {len(dropped_unreliable)}")
    print(f"  Missing/invalid OS_months: {len(dropped_missing_os)}")
    _print_id_list(
        "Dropped (lobe_assignment_reliable=False):",
        dropped_unreliable["patient_id"].astype(str).tolist(),
    )
    _print_id_list(
        "Dropped (missing or invalid OS_months):",
        dropped_missing_os["patient_id"].astype(str).tolist(),
    )
    print(f"Wrote {len(out_df)} rows -> {out_csv}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess GBM features: filter, label, impute.")
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to config.json (default: config.json)",
    )
    parser.add_argument(
        "--modality",
        choices=MODALITIES,
        type=str.upper,
        default="T1",
        help="Modality for default inputs/outputs (default: T1).",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Input CSV (defaults to modality-specific outputs/features_raw_{mod}.csv)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV (defaults to modality-specific outputs/features_processed_{mod}.csv)",
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        help="Apply StandardScaler to features (off by default)",
    )
    parser.add_argument(
        "--confirm-scale",
        action="store_true",
        help="Confirm scaling even though it is not part of the abstract",
    )
    args = parser.parse_args()

    if args.scale and not args.confirm_scale:
        print(
            "ERROR: --scale is not part of the abstract. "
            "Re-run with --scale --confirm-scale if you really want it."
        )
        return 2

    root = Path(__file__).resolve().parents[1]
    cfg = _load_config(_resolve_path(root, args.config))

    modality = args.modality.upper()
    modality_lc = modality.lower()
    default_in = f"outputs/features_raw_{modality_lc}.csv"
    default_out = f"outputs/features_processed_{modality_lc}.csv"

    in_csv = _resolve_path(root, args.input or default_in)
    out_csv = _resolve_path(root, args.output or default_out)

    return preprocess(in_csv, out_csv, cfg, args.scale)


if __name__ == "__main__":
    raise SystemExit(main())
