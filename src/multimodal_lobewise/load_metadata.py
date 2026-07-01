#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

METADATA_SOURCE = "UCSF-PDGM-metadata_v5.csv"

OUTPUT_COLS = [
    "patient_id",
    "age",
    "sex",
    "idh",
    "mgmt",
    "who_grade",
    "eor",
]

INPUT_MAP = {
    "patient_id": "ID",
    "age": "Age at MRI",
    "sex": "Sex",
    "idh": "IDH",
    "mgmt": "MGMT status",
    "who_grade": "WHO CNS Grade",
    "eor": "EOR",
}


def _normalize_patient_id(pid: str) -> str:
    """Zero-pad the numeric portion of UCSF-PDGM IDs to 4 digits.

    Handles optional _FU… suffix used for follow-up scans.
    Examples:
        UCSF-PDGM-004    -> UCSF-PDGM-0004
        UCSF-PDGM-0105   -> UCSF-PDGM-0105
        UCSF-PDGM-0429_FU003d -> UCSF-PDGM-0429_FU003d
    """
    pid = str(pid).strip()
    m = re.match(r"^(UCSF-PDGM-)(\d+)(.*)", pid)
    if m:
        return f"{m.group(1)}{int(m.group(2)):04d}{m.group(3)}"
    return pid


def _resolve_path(root: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (root / p)


def _encode_sex(val: str) -> int:
    v = str(val).strip().upper()
    if v == "F":
        return 0
    if v == "M":
        return 1
    raise ValueError(f"Unexpected sex value: {val!r}")


def _encode_idh(val: str) -> int:
    v = str(val).strip().lower()
    if v == "wildtype":
        return 0
    return 1


def _encode_mgmt(val: str) -> float:
    v = str(val).strip().lower()
    if v == "positive":
        return 1.0
    if v == "negative":
        return 0.0
    return np.nan


def _encode_who_grade(val: str) -> int:
    return int(str(val).strip())


def _encode_eor(val: str) -> float:
    v = str(val).strip().lower()
    if v == "biopsy":
        return 0.0
    if v == "str":
        return 1.0
    if v == "gtr":
        return 2.0
    return np.nan


def load_metadata(
    in_csv: Path,
    out_csv: Path,
) -> int:
    """Load, encode, and save processed metadata."""
    df = pd.read_csv(in_csv)

    renamed = df.rename(columns={v: k for k, v in INPUT_MAP.items()})

    result = pd.DataFrame()
    result["patient_id"] = renamed["patient_id"].astype(str).str.strip().apply(
        _normalize_patient_id
    )
    result["age"] = pd.to_numeric(renamed["age"], errors="coerce")
    result["sex"] = renamed["sex"].apply(_encode_sex).astype(int)
    result["idh"] = renamed["idh"].apply(_encode_idh).astype(int)

    # Use .loc to avoid chained-assignment warnings
    raw_mgmt = renamed["mgmt"].apply(_encode_mgmt)
    result.loc[:, "mgmt"] = raw_mgmt

    result["who_grade"] = renamed["who_grade"].apply(_encode_who_grade).astype(int)

    raw_eor = renamed["eor"].apply(_encode_eor)
    result.loc[:, "eor"] = raw_eor

    result = result[OUTPUT_COLS]

    print(f"Loaded metadata from {in_csv}")
    print(f"  Total patients: {len(result)}")
    print(f"  Columns: {list(result.columns)}")
    print(f"  Dtypes:\n{result.dtypes.to_string()}")
    print(f"  Missing values:\n{result.isnull().sum().to_string()}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False)
    print(f"Wrote {len(result)} rows -> {out_csv}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load and encode UCSF-PDGM metadata.",
    )
    parser.add_argument(
        "--input", default=None,
        help=f"Metadata CSV (default: {METADATA_SOURCE})",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output CSV path (default: outputs/multimodal_lobewise/metadata_processed.csv)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    in_csv = _resolve_path(root, args.input or METADATA_SOURCE)
    if not in_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {in_csv}")

    out_csv = _resolve_path(
        root,
        args.output or "outputs/multimodal_lobewise/metadata_processed.csv",
    )
    return load_metadata(in_csv, out_csv)


if __name__ == "__main__":
    raise SystemExit(main())
