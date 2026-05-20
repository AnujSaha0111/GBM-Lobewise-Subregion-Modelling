#!/usr/bin/env python3
"""Preprocess the merged multimodal lobewise-subregion dataset for ML.

Loads merged_features.csv, separates metadata from features, performs median
imputation and standard scaling, then saves the processed dataset and the fitted
preprocessing artifacts.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ── Constants ──

METADATA_COLS = ["patient_id", "risk_label"]
DROP_COLS = {"OS_months", "lobe_assignment_reliable"}

DEFAULT_INPUT = "outputs/multimodal_lobewise/merged_features.csv"
DEFAULT_OUTPUT = "outputs/multimodal_lobewise/processed_features.csv"
DEFAULT_SCALER = "outputs/multimodal_lobewise/scaler.pkl"
DEFAULT_IMPUTER = "outputs/multimodal_lobewise/imputer.pkl"


# ── Helpers ──

def _resolve_path(root: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (root / p)


def _verify_integrity(
    feature_df: pd.DataFrame, raw_columns: list[str],
) -> None:
    """Run strong checks before and after preprocessing."""
    # No duplicate columns
    assert feature_df.columns.is_unique, \
        f"Duplicate column(s) found: {feature_df.columns[feature_df.columns.duplicated()].tolist()}"

    # All feature columns must be numeric
    non_numeric = feature_df.select_dtypes(exclude=["number"]).columns.tolist()
    assert len(non_numeric) == 0, \
        f"Non-numeric feature column(s): {non_numeric}"

    # No metadata leakage — feature columns must be exactly the raw input features
    assert set(feature_df.columns) == set(raw_columns), \
        "Feature columns changed unexpectedly after extraction."

    # No NaN after preprocessing
    assert feature_df.isnull().sum().sum() == 0, \
        f"NaNs remain after preprocessing: {feature_df.isnull().sum().sum()}"


# ── Public Functions ──

def preprocess_multimodal(
    in_csv: Path,
    out_csv: Path,
    scaler_path: Path,
    imputer_path: Path,
) -> int:
    """Load, verify, impute, scale, and save the multimodal dataset."""
    # Load
    df = pd.read_csv(in_csv)
    total_patients = len(df)

    print(f"Loaded {total_patients} rows from {in_csv}")

    # Separate metadata, drop columns, and features
    metadata = df[METADATA_COLS].copy()

    raw_feature_cols = [
        c for c in df.columns
        if c not in METADATA_COLS and c not in DROP_COLS
    ]
    n_features = len(raw_feature_cols)
    features = df[raw_feature_cols]

    # Pre-imputation NaN count
    nan_before = features.isnull().sum().sum()

    # Verify extracted features
    _verify_integrity(features, raw_feature_cols)

    # Median imputation
    imputer = SimpleImputer(strategy="median")
    imputed_array = imputer.fit_transform(features)
    imputed_df = pd.DataFrame(imputed_array, columns=raw_feature_cols, index=features.index)

    # Post-imputation NaN count (should be 0, but we measure before assertion)
    nan_after = imputed_df.isnull().sum().sum()

    # Standard scaling — preserve column names
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(imputed_df)
    scaled_df = pd.DataFrame(scaled_array, columns=raw_feature_cols, index=features.index)

    # Reconstruct final DataFrame
    processed = pd.concat([metadata, scaled_df], axis=1)

    # Post-processing integrity checks
    _verify_integrity(scaled_df, raw_feature_cols)

    # Persist
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(out_csv, index=False)

    for path, obj in [(scaler_path, scaler), (imputer_path, imputer)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(obj, f)

    # Summary
    print(f"Patient count: {total_patients}")
    print(f"Feature count: {n_features}")
    print(f"NaN counts — before: {nan_before}, after: {nan_after}")
    print(f"Class distribution:\n{metadata['risk_label'].value_counts().to_string()}")
    print(f"Feature matrix shape: {scaled_df.shape}")
    print(f"Wrote {out_csv}, {scaler_path}, {imputer_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess the merged multimodal lobewise-subregion dataset for ML.",
    )
    parser.add_argument(
        "--input", default=None,
        help=f"Input CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", default=None,
        help=f"Output CSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--scaler", default=None,
        help=f"Scaler pickle path (default: {DEFAULT_SCALER})",
    )
    parser.add_argument(
        "--imputer", default=None,
        help=f"Imputer pickle path (default: {DEFAULT_IMPUTER})",
    )

    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]

    in_csv = _resolve_path(root, args.input or DEFAULT_INPUT)
    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    out_csv = _resolve_path(root, args.output or DEFAULT_OUTPUT)
    scaler_path = _resolve_path(root, args.scaler or DEFAULT_SCALER)
    imputer_path = _resolve_path(root, args.imputer or DEFAULT_IMPUTER)

    return preprocess_multimodal(in_csv, out_csv, scaler_path, imputer_path)


if __name__ == "__main__":
    raise SystemExit(main())
