#!/usr/bin/env python3
"""SRI24 atlas registration and 16-feature extraction for GBM survival
risk stratification.

Builds a 4-lobe atlas from SRI24 parcellation, registers it to each
patient's native MRI space via ANTs (or affine fallback on Windows),
then extracts exactly 16 radiomic features per patient.

Usage:
    python src/atlas_registration.py                            # full batch (default: T1)
    python src/atlas_registration.py --modality T1GD           # use T1GD as fixed image
    python src/atlas_registration.py --modality FLAIR --dry-run
    python src/atlas_registration.py --patient UCSF-PDGM-0004 --modality T2

Supported modalities: T1, T2, T1GD, FLAIR
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import shutil
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

# ── Constants ─────────────────────────────────────────────────────────────
NC_LABEL, ED_LABEL, EN_LABEL = 1, 2, 4
LOBE_IDS = {"frontal": 1, "temporal": 2, "parietal": 3, "occipital": 4}
LOBE_NAMES = list(LOBE_IDS.keys())

LOBE_PREFIXES = {
    "frontal": ("Frontal_", "Precentral_", "Rolandic_Oper_",
                "Supp_Motor_Area_", "Olfactory_", "Rectus_"),
    "temporal": ("Temporal_", "Heschl_", "ParaHippocampal_",
                 "Hippocampus_", "Amygdala_", "Fusiform_"),
    "parietal": ("Postcentral_", "Parietal_", "SupraMarginal_",
                 "Angular_", "Precuneus_", "Paracentral_Lobule_"),
    "occipital": ("Calcarine_", "Cuneus_", "Lingual_", "Occipital_"),
}

FEATURE_COLS = [
    "global_nc_en_ratio", "global_ed_en_ratio",
    "global_ed_total_ratio", "tumor_burden_index",
    *(f"{lb}_{sub}_ratio" for lb in LOBE_NAMES for sub in ("ed", "en", "nc")),
]
assert len(FEATURE_COLS) == 16

OUTPUT_COLS = ["patient_id", *FEATURE_COLS, "OS_months", "lobe_assignment_reliable"]

IS_WINDOWS = platform.system() == "Windows"

# Supported modalities and their config-key suffixes
MODALITY_SUFFIX_KEY: dict[str, str] = {
    "T1":   "t1_suffix",
    "T2":   "t2_suffix",
    "T1GD": "t1gd_suffix",
    "FLAIR": "flair_suffix",
}

# linear for intensity images; nearestNeighbor stays for atlas/brainmask
MODALITY_INTERPOLATOR: dict[str, str] = {
    "T1":    "linear",
    "T2":    "linear",
    "T1GD":  "linear",
    "FLAIR": "linear",
}


# ── Helpers ───────────────────────────────────────────────────────────────
def _sdiv(num: float, den: float) -> float:
    """Safe division: 0.0 on zero denominator."""
    return 0.0 if den == 0 else float(num) / float(den)


def _load_vol(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    """Load NIfTI, squeeze trailing 4th dim if present."""
    img = nib.load(str(path))
    d = np.asarray(img.dataobj)
    if d.ndim == 4 and d.shape[-1] == 1:
        d = d[..., 0]
    if d.ndim != 3:
        raise ValueError(f"Expected 3-D at {path}, got {d.shape}")
    return img, d


def _load_config(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _nib_to_ants(nib_img: nib.Nifti1Image):
    """
    Convert a nibabel NIfTI image to an ANTsPy image.
    Works with both old and new antspyx APIs.
    
    Modern antspyx removed ants.from_nibabel().
    This function handles both old and new versions gracefully.
    """
    import ants  # type: ignore

    # Try modern API first (antspyx >= 0.3.x)
    if hasattr(ants, "from_nibabel"):
        # Old API still present — use it directly
        return ants.from_nibabel(nib_img)

    # Modern API: convert manually via numpy + affine decomposition
    data = np.asarray(nib_img.dataobj, dtype=np.float32)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]

    affine = nib_img.affine

    # Extract spacing (voxel sizes) from affine
    spacing = tuple(float(np.linalg.norm(affine[:3, i])) for i in range(3))

    # Extract origin
    ras2lps = np.diag([-1.0, -1.0, 1.0])
    origin = tuple(float(x) for x in (ras2lps @ affine[:3, 3]))

    # Extract direction cosines (unit column vectors)
    direction = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        col = affine[:3, i]
        norm = np.linalg.norm(col)
        col = ras2lps @ col
        direction[i, :] = col / norm if norm > 0 else col

    return ants.from_numpy(
        data,
        origin=origin,
        spacing=spacing,
        direction=direction,
    )


def _ants_to_numpy(ants_img) -> np.ndarray:
    """Extract numpy array from ANTsPy image (API-agnostic)."""
    if hasattr(ants_img, "numpy"):
        return ants_img.numpy()
    elif hasattr(ants_img, "view"):
        return np.array(ants_img.view())
    else:
        raise RuntimeError("Cannot extract numpy array from ANTsPy image. "
                           "Check antspyx version.")


# ── Section A: Build 4-lobe atlas (run once, cached) ─────────────────────
def _parse_tzo_labels(label_file: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    for line in label_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"\s+", line)
        try:
            out[int(parts[0])] = parts[1]
        except (ValueError, IndexError):
            continue
    return out


def _build_seeds(tzo: np.ndarray, label_map: dict[int, str]) -> np.ndarray:
    seeds = np.zeros(tzo.shape, dtype=np.uint8)
    for idx, name in label_map.items():
        if name == "Background" or "Ventricle" in name:
            continue
        if name.startswith(("Cerebelum_", "Vermis_")):
            continue
        for lobe, prefixes in LOBE_PREFIXES.items():
            if name.startswith(prefixes):
                seeds[tzo == idx] = LOBE_IDS[lobe]
                break
    return seeds


def build_4lobe_atlas(
    sri24_dir: Path, dilation: int, out_path: Path
) -> tuple[nib.Nifti1Image, np.ndarray]:
    """Build or load the filled 4-lobe atlas in SRI24 space."""
    if out_path.exists():
        print(f"  Cached atlas: {out_path}")
        return _load_vol(out_path)

    tzo_path = sri24_dir / "tzo116plus.nii.gz"
    sup_path = sri24_dir / "suptent.nii.gz"
    lbl_path = sri24_dir / "SRI24-tzo116plus.txt"
    for p in (tzo_path, sup_path, lbl_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing SRI24 file: {p}")

    tzo_img, tzo_data = _load_vol(tzo_path)
    _, sup_data = _load_vol(sup_path)
    seeds = _build_seeds(tzo_data.astype(np.int32), _parse_tzo_labels(lbl_path))

    mask = sup_data > 0
    if dilation > 0:
        mask = ndimage.binary_dilation(mask, iterations=dilation)

    nn_idx = ndimage.distance_transform_edt(
        seeds == 0, return_distances=False, return_indices=True)
    filled = seeds.copy()
    filled[mask] = seeds[tuple(ix[mask] for ix in nn_idx)]
    filled[~mask] = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    atlas_img = nib.Nifti1Image(filled, tzo_img.affine, tzo_img.header)
    nib.save(atlas_img, str(out_path))
    print(f"  Built 4-lobe atlas -> {out_path}")
    return atlas_img, filled


# ── Section B: Registration ───────────────────────────────────────────────
def _affine_resample_nn(
    src_img: nib.Nifti1Image, tgt_img: nib.Nifti1Image
) -> np.ndarray:
    """Nearest-neighbor resample via NIfTI affines (Windows fallback)."""
    tgt_shape = tgt_img.shape[:3]
    coords = np.indices(tgt_shape, dtype=np.int32).reshape(3, -1).T
    world = nib.affines.apply_affine(tgt_img.affine, coords)
    src_ijk = nib.affines.apply_affine(np.linalg.inv(src_img.affine), world)
    si = np.rint(src_ijk).astype(np.int32)
    ss = np.array(src_img.shape[:3])
    valid = np.all((si >= 0) & (si < ss), axis=1)
    sd = np.asarray(src_img.dataobj)
    if sd.ndim == 4:
        sd = sd[..., 0]
    out = np.zeros(coords.shape[0], dtype=sd.dtype)
    if valid.any():
        out[valid] = sd[tuple(si[valid].T)]
    return out.reshape(tgt_shape)


def _ants_register(
    patient_img: Path,
    sri24_t1: Path,
    atlas_img: nib.Nifti1Image,
    brainmask_img: nib.Nifti1Image,
    tx_dir: Path,
    reg_type: str,
    cached_affine_ids: set[str],
    cached_syn_ids: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    ANTs registration: SRI24 T1 -> patient native space (fixed = patient_img).
    Applies same transform to atlas (label vol) and brainmask.
    Uses modern antspyx API — no ants.from_nibabel().
    """
    import ants  # type: ignore

    fixed = ants.image_read(str(patient_img))
    moving = ants.image_read(str(sri24_t1))

    # Flat prefix cache: outputs/transforms/UCSF-PDGM-01020GenericAffine.mat
    prefix = tx_dir.name    # patient ID
    flat_dir = tx_dir.parent  # outputs/transforms/
    affine_mat = flat_dir / f"{prefix}0GenericAffine.mat"
    warp_field = flat_dir / f"{prefix}1Warp.nii.gz"

    if reg_type == "SyN" and prefix in cached_syn_ids:
        tx_list = [str(warp_field), str(affine_mat)]
        print("    [CACHE] Reusing SyN transforms")
    elif reg_type != "SyN" and prefix in cached_affine_ids:
        tx_list = [str(affine_mat)]
        print("    [CACHE] Reusing Affine transform")
    else:
        print(f"    [ANTs] Running {reg_type} registration ...", end=" ", flush=True)
        flat_dir.mkdir(parents=True, exist_ok=True)
        reg = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform=reg_type,
            outprefix=str(flat_dir / prefix),
        )
        tx_list = reg["fwdtransforms"]
        print("done")
        if reg_type == "SyN":
            cached_affine_ids.add(prefix)
            cached_syn_ids.add(prefix)
        else:
            cached_affine_ids.add(prefix)

    # Convert nibabel images to ANTsPy (modern API-safe)
    a_ants = _nib_to_ants(atlas_img)
    bm_ants = _nib_to_ants(brainmask_img)

    # Apply transform to atlas — MUST use nearestNeighbor (label volume)
    reg_atlas = ants.apply_transforms(
        fixed=fixed,
        moving=a_ants,
        transformlist=tx_list,
        interpolator="nearestNeighbor",
    )

    # Apply transform to brainmask — also nearest neighbor
    reg_bm = ants.apply_transforms(
        fixed=fixed,
        moving=bm_ants,
        transformlist=tx_list,
        interpolator="nearestNeighbor",
    )

    return _ants_to_numpy(reg_atlas), _ants_to_numpy(reg_bm)


def _ants_available() -> bool:
    """Check if antspyx is importable."""
    try:
        import ants  # noqa: F401
        return True
    except ImportError:
        return False


def _scan_transform_cache(cache_dir: Path) -> tuple[set[str], set[str]]:
    """Scan transform cache once and return (affine_ids, syn_ids)."""
    if not cache_dir.exists():
        return set(), set()

    affine_ids: set[str] = set()
    warp_ids: set[str] = set()

    affine_suffix = "0GenericAffine.mat"
    warp_suffix = "1Warp.nii.gz"

    for p in cache_dir.rglob("*"):
        if not p.is_file():
            continue
        name = p.name
        if name.endswith(affine_suffix):
            affine_ids.add(name[: -len(affine_suffix)])
        elif name.endswith(warp_suffix):
            warp_ids.add(name[: -len(warp_suffix)])

    syn_ids = affine_ids & warp_ids
    return affine_ids, syn_ids


def register_atlas(
    patient_img: Path,
    seg_img: nib.Nifti1Image,
    atlas_img: nib.Nifti1Image,
    brainmask_img: nib.Nifti1Image,
    cfg: dict,
    patient_id: str,
    cached_affine_ids: set[str] | None = None,
    cached_syn_ids: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (registered_atlas, registered_brainmask) in patient native space.

    Priority:
    1. ANTs (if use_ants_registration=true AND antspyx importable)
    2. Affine fallback (always works, Windows-safe)
    """
    use_ants = cfg["atlas"].get("use_ants_registration", True)
    if cached_affine_ids is None:
        cached_affine_ids = set()
    if cached_syn_ids is None:
        cached_syn_ids = set()

    if use_ants:
        if IS_WINDOWS:
            print("    [WARN] Windows detected — antspyx unavailable, "
                  "using affine fallback. Run on Linux/WSL for ANTs.")
        elif not _ants_available():
            print("    [WARN] antspyx not installed — using affine fallback. "
                  "Install with: pip install antspyx")
        else:
            sri24_dir = Path(cfg["data"]["sri24_dir"])
            sri24_t1 = sri24_dir / cfg["data"].get("sri24_t1_filename", "T1.nii.gz")
            if not sri24_t1.exists():
                raise FileNotFoundError(
                    f"SRI24 T1 template not found: {sri24_t1}\n"
                    "Set 'sri24_t1_filename' in config.json to the correct filename."
                )
            tx_dir = Path(cfg["atlas"]["transforms_cache_dir"]) / patient_id
            return _ants_register(
                patient_img, sri24_t1,
                atlas_img, brainmask_img,
                tx_dir, cfg["atlas"]["registration_type"],
                cached_affine_ids, cached_syn_ids,
            )

    # Affine fallback path
    print("    [WARN] Affine fallback (no ANTs registration)")
    ra = _affine_resample_nn(atlas_img, seg_img)
    rb = _affine_resample_nn(brainmask_img, seg_img)
    return ra, rb


# ── Section C: Extract 16 features ───────────────────────────────────────
def extract_features(
    seg_data: np.ndarray,
    atlas_data: np.ndarray,
    brainmask: np.ndarray,
    patient_id: str,
    os_months: float,
) -> dict:
    """Compute exactly 16 features + metadata for one patient."""
    seg = seg_data.astype(np.uint8)
    nc_total = int(np.sum(seg == NC_LABEL))
    ed_total = int(np.sum(seg == ED_LABEL))
    en_total = int(np.sum(seg == EN_LABEL))
    wt_total = nc_total + ed_total + en_total
    brain_vox = int(np.sum(brainmask > 0))

    row: dict = {"patient_id": patient_id}

    # Global features (4)
    row["global_nc_en_ratio"]   = _sdiv(nc_total, en_total)
    row["global_ed_en_ratio"]   = _sdiv(ed_total, en_total)
    row["global_ed_total_ratio"] = _sdiv(ed_total, wt_total)
    row["tumor_burden_index"]   = _sdiv(wt_total, brain_vox)

    # Lobe-wise features (12)
    # Denominator = total lobe voxels (lobe invasion fraction per abstract)
    tumor_mask = seg > 0
    mapped_vox = 0
    for lobe_name, lobe_id in LOBE_IDS.items():
        lobe_mask = atlas_data == lobe_id
        lobe_total = int(np.sum(lobe_mask))
        nc_in = int(np.sum((seg == NC_LABEL) & lobe_mask))
        ed_in = int(np.sum((seg == ED_LABEL) & lobe_mask))
        en_in = int(np.sum((seg == EN_LABEL) & lobe_mask))
        mapped_vox += int(np.sum(tumor_mask & lobe_mask))

        row[f"{lobe_name}_ed_ratio"] = _sdiv(ed_in, lobe_total)
        row[f"{lobe_name}_en_ratio"] = _sdiv(en_in, lobe_total)
        row[f"{lobe_name}_nc_ratio"] = _sdiv(nc_in, lobe_total)

    # QA
    reliable = (_sdiv(mapped_vox, wt_total) >= 0.90) if wt_total > 0 else False
    row["OS_months"] = os_months
    row["lobe_assignment_reliable"] = reliable
    return row


# ── Section D: Discovery + batch runner ───────────────────────────────────
def discover_patients(cfg: dict, root: Path, modality: str) -> list[dict]:
    """Find patients that have both the chosen modality file and segmentation."""
    struct_dir  = root / cfg["data"]["ucsf_root"] / cfg["data"]["structural_subdir"]
    seg_dir     = root / cfg["data"]["ucsf_root"] / cfg["data"]["segmentation_subdir"]
    suffix_key  = MODALITY_SUFFIX_KEY[modality]          # e.g. "t1gd_suffix"
    mod_sfx     = cfg["data"][suffix_key]                # e.g. "_T1GD.nii.gz"
    seg_sfx     = cfg["data"]["seg_suffix"]

    patients = []
    for d in sorted(struct_dir.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        pid     = d.name
        mod_img = d / f"{pid}{mod_sfx}"
        seg     = seg_dir / f"{pid}{seg_sfx}"
        if not mod_img.exists() or not seg.exists():
            continue
        patients.append({"id": pid, "img": mod_img, "seg": seg})
    return patients


def load_clinical_os(csv_path: Path, days_per_month: float) -> dict[str, float]:
    """Read clinical CSV -> {patient_id: OS_months}."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    id_col = "ID" if "ID" in df.columns else "patient_id"
    if id_col not in df.columns:
        raise ValueError(
            f"Expected 'ID' or 'patient_id' column in {csv_path}, "
            f"got: {list(df.columns)}"
        )
    if "OS" not in df.columns:
        raise ValueError(f"Expected 'OS' column in {csv_path}")
    os_map: dict[str, float] = {}
    for _, r in df.iterrows():
        pid = str(r[id_col]).strip()
        os_val = r["OS"]
        if pd.isna(os_val):
            os_map[pid] = float("nan")
        else:
            os_map[pid] = float(os_val) / days_per_month
    return os_map


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument(
        "--modality",
        choices=list(MODALITY_SUFFIX_KEY.keys()),
        default="T1",
        help="Fixed image modality for registration (default: T1). "
             "Choices: T1, T2, T1GD, FLAIR",
    )
    parser.add_argument("--patient", type=str, default=None,
                        help="Run for a single patient ID only.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List discovered patients without processing.")
    args = parser.parse_args()

    modality     = args.modality.upper()
    modality_lc  = modality.lower()          # used in output filename

    root = args.config.resolve().parent
    cfg  = _load_config(args.config.resolve())

    # Persist the chosen modality into config so the run is self-documenting
    cfg["atlas"]["active_modality"] = modality
    with args.config.resolve().open("w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"[CONFIG] active_modality set to '{modality}' in {args.config.resolve().name}")

    # Platform info
    print(f"Platform: {platform.system()} | "
          f"ANTs available: {_ants_available()} | "
          f"use_ants_registration: {cfg['atlas'].get('use_ants_registration', True)} | "
          f"Modality: {modality}")

    # Load clinical OS
    csv_path = root / cfg["data"]["clinical_csv"]
    days_pm  = cfg["preprocessing"]["os_days_per_month"]
    os_map   = load_clinical_os(csv_path, days_pm)

    # Scan transform cache once for resume-aware registration
    tx_cache_dir = root / cfg["atlas"]["transforms_cache_dir"]
    cached_affine_ids, cached_syn_ids = _scan_transform_cache(tx_cache_dir)

    # Discover patients (modality-aware — skips patients missing the file)
    patients = discover_patients(cfg, root, modality)
    if args.patient:
        patients = [p for p in patients if p["id"] == args.patient]
        if not patients:
            print(f"ERROR: patient '{args.patient}' not found.")
            return 1

    print(f"Found {len(patients)} patients with {modality} + segmentation files.")

    if args.dry_run:
        for p in patients:
            os_m   = os_map.get(p["id"], float("nan"))
            in_csv = "yes" if p["id"] in os_map else "NO"
            print(f"  {p['id']}  OS={os_m:.1f}mo  csv={in_csv}")
        print("Dry run complete. No processing performed.")
        return 0

    # Warn about patients missing from CSV
    missing = [p["id"] for p in patients if p["id"] not in os_map]
    if missing:
        print(f"WARNING: {len(missing)} patients have no CSV match: "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}")

    # Build 4-lobe atlas (once, cached)
    sri24_dir = root / cfg["data"]["sri24_dir"]
    atlas_out = root / "outputs" / "sri24_4lobe_atlas.nii.gz"
    atlas_img, _ = build_4lobe_atlas(
        sri24_dir, cfg["atlas"]["lobe_dilation_voxels"], atlas_out)

    # Load brainmask
    tissues_path = sri24_dir / "tissues.nii.gz"
    if not tissues_path.exists():
        raise FileNotFoundError(f"Missing: {tissues_path}")
    bm_img, bm_data = _load_vol(tissues_path)
    bm_img = nib.Nifti1Image(
        (bm_data > 0).astype(np.uint8), bm_img.affine, bm_img.header)

    # Output CSV is modality-specific — T1 results are never overwritten
    out_csv = root / "outputs" / f"features_raw_{modality_lc}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    completed_ids: set[str] = set()
    csv_exists = out_csv.exists()
    if csv_exists:
        import pandas as pd
        try:
            done_df = pd.read_csv(out_csv, usecols=["patient_id"])
            completed_ids = set(done_df["patient_id"].astype(str).tolist())
            print(f"[RESUME] Found {len(completed_ids)} already-processed patients in CSV.")
        except Exception as e:
            print(f"[WARN] Could not read existing CSV: {e}. Starting fresh.")
            csv_exists = False

    # Open CSV in append mode — write header only if file is new
    failed: list[str] = []
    csv_file = out_csv.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLS)
    if not csv_exists:
        writer.writeheader()

    try:
        for i, p in enumerate(patients, 1):
            pid  = p["id"]
            os_m = os_map.get(pid, float("nan"))

            if pid in completed_ids:
                print(f"[{i}/{len(patients)}] {pid} ... [SKIP] Already processed")
                continue

            print(f"[{i}/{len(patients)}] {pid} ...", end=" ", flush=True)

            try:
                seg_img, seg_data = _load_vol(p["seg"])
                reg_atlas, reg_bm = register_atlas(
                    p["img"], seg_img, atlas_img, bm_img, cfg, pid,
                    cached_affine_ids, cached_syn_ids)
                row = extract_features(seg_data, reg_atlas, reg_bm, pid, os_m)
                writer.writerow(row)
                csv_file.flush()
                rel = "OK" if row["lobe_assignment_reliable"] else "UNRELIABLE"
                print(f"done ({rel}) [SAVE] Appended row")
                completed_ids.add(pid)
            except Exception as e:
                print(f"FAILED: {e}")
                failed.append(pid)
    finally:
        csv_file.close()

    print(f"\nDone. CSV -> {out_csv}")
    if failed:
        print(f"Failed patients ({len(failed)}): {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())