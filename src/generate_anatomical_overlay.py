#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import nibabel as nib

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "paper" / "figures"

PATIENT = "UCSF-PDGM-0008"
T1GD_PATH = ROOT / "UCSF" / "DATA-IMAGE-STRUCTURAL" / PATIENT / f"{PATIENT}_T1GD.nii.gz"
SEG_PATH = ROOT / "UCSF" / "DATA-AUTOMATED-SEGMENT" / f"{PATIENT}_tumor_segmentation.nii.gz"
ATLAS_PATH = ROOT / "outputs" / "sri24_4lobe_atlas.nii.gz"

NC_LABEL, ED_LABEL, EN_LABEL = 1, 2, 4
LOBE_IDS = {"frontal": 1, "temporal": 2, "parietal": 3, "occipital": 4}

# Colors
CMAP_MRI = "gray"
COLOR_ET = "#E41A1C"    # red
COLOR_NC = "#FFD700"    # yellow
COLOR_ED = "#377EB8"    # blue
COLOR_ATLAS = plt.cm.Set2_r

LOBE_COLORS = {
    "frontal": "#66C2A5",   # green
    "temporal": "#FC8D62",  # orange
    "parietal": "#8DA0CB",  # blue
    "occipital": "#E78AC3", # pink
}


def find_best_slice(mri_data, seg_data, margin=10):
    """Find axial slice with maximum tumor area, centered in the brain."""
    tumor_mask = seg_data > 0
    slices = np.where(tumor_mask.sum(axis=(0, 1)) > 0)[0]
    if len(slices) == 0:
        return mri_data.shape[2] // 2
    center = int(np.median(slices))
    # avoid edges
    z_max = mri_data.shape[2] - 1 - margin
    return max(margin, min(center, z_max))


def overlay_mask(ax, mask, color, alpha=0.7, label=None):
    """Plot a colored mask overlay on an axis."""
    rgba = np.zeros((*mask.shape, 4))
    c = plt.matplotlib.colors.to_rgba(color)
    rgba[mask] = c
    rgba[mask, 3] = alpha
    ax.imshow(rgba, interpolation="nearest", origin="lower")


def add_lobe_labels(ax, atlas_slice, fontsize=9):
    """Add lobe name labels at the centroid of each lobe region."""
    from scipy import ndimage
    for lobe_name, lobe_id in LOBE_IDS.items():
        mask = atlas_slice == lobe_id
        if not mask.any():
            continue
        cy, cx = ndimage.center_of_mass(mask) if mask.sum() > 0 else (0, 0)
        ax.text(cx, cy, lobe_name, fontsize=fontsize, fontweight="bold",
                color="white", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black",
                          edgecolor="none", alpha=0.6))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {T1GD_PATH.name}...")
    t1gd_img = nib.load(str(T1GD_PATH))
    t1gd_data = np.asarray(t1gd_img.dataobj)

    print(f"Loading {SEG_PATH.name}...")
    seg_img = nib.load(str(SEG_PATH))
    seg_data = np.asarray(seg_img.dataobj)

    print(f"Loading {ATLAS_PATH.name}...")
    atlas_img = nib.load(str(ATLAS_PATH))
    atlas_data = np.asarray(atlas_img.dataobj)

    # Orient atlas to match patient space (flip y-axis)
    if t1gd_img.affine[1, 1] != atlas_img.affine[1, 1]:
        print("Flipping atlas y-axis to match patient orientation...")
        atlas_data = atlas_data[:, ::-1, :]
    if t1gd_img.affine[0, 0] != atlas_img.affine[0, 0]:
        print("Flipping atlas x-axis to match patient orientation...")
        atlas_data = atlas_data[::-1, :, :]

    # Find best axial slice
    z = find_best_slice(t1gd_data, seg_data)
    print(f"Selected axial slice index: {z}")

    # Extract slices
    mri_slice = t1gd_data[:, :, z].T
    seg_slice = seg_data[:, :, z].T
    atlas_slice = atlas_data[:, :, z].T

    # Normalize MRI for display
    p_low, p_high = np.percentile(mri_slice[mri_slice > 0], (1, 99))
    mri_display = np.clip(mri_slice, p_low, p_high)
    mri_display = (mri_display - p_low) / (p_high - p_low + 1e-8)

    # Individual masks
    mask_et = seg_slice == EN_LABEL
    mask_nc = seg_slice == NC_LABEL
    mask_ed = seg_slice == ED_LABEL

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # ── Panel A: T1GD axial ──
    ax_a.imshow(mri_display, cmap=CMAP_MRI, origin="lower")
    ax_a.set_title("A  T1GD Axial Slice", fontsize=12, fontweight="bold", loc="left")
    ax_a.axis("off")

    # ── Panel B: Tumor subregions ──
    ax_b.imshow(mri_display, cmap=CMAP_MRI, origin="lower", alpha=0.3)
    overlay_mask(ax_b, mask_ed, COLOR_ED, alpha=0.6)
    overlay_mask(ax_b, mask_nc, COLOR_NC, alpha=0.7)
    overlay_mask(ax_b, mask_et, COLOR_ET, alpha=0.7)
    ax_b.set_title("B  Tumor Subregions", fontsize=12, fontweight="bold", loc="left")
    ax_b.axis("off")

    legend_elements = [
        mpatches.Patch(color=COLOR_ET, label="Enhancing Tumor (ET)"),
        mpatches.Patch(color=COLOR_NC, label="Necrotic Core (NC)"),
        mpatches.Patch(color=COLOR_ED, label="Peritumoral Edema (ED)"),
    ]
    ax_b.legend(handles=legend_elements, loc="lower right", fontsize=7,
                framealpha=0.85, edgecolor="gray")

    # ── Panel C: SRI24 lobe parcellation ──
    lobe_rgba = np.zeros((*atlas_slice.shape, 4))
    for lobe_name, lobe_id in LOBE_IDS.items():
        mask = atlas_slice == lobe_id
        c = plt.matplotlib.colors.to_rgba(LOBE_COLORS[lobe_name])
        lobe_rgba[mask] = c
        lobe_rgba[mask, 3] = 0.5

    ax_c.imshow(lobe_rgba, interpolation="nearest", origin="lower")
    add_lobe_labels(ax_c, atlas_slice, fontsize=9)
    ax_c.set_title("C  SRI24 Lobe Parcellation", fontsize=12, fontweight="bold", loc="left")
    ax_c.axis("off")

    lobe_legend = [
        mpatches.Patch(color=LOBE_COLORS["frontal"], label="Frontal"),
        mpatches.Patch(color=LOBE_COLORS["temporal"], label="Temporal"),
        mpatches.Patch(color=LOBE_COLORS["parietal"], label="Parietal"),
        mpatches.Patch(color=LOBE_COLORS["occipital"], label="Occipital"),
    ]
    ax_c.legend(handles=lobe_legend, loc="lower right", fontsize=7,
                framealpha=0.85, edgecolor="gray", ncol=2)

    # ── Panel D: Full overlay ──
    ax_d.imshow(mri_display, cmap=CMAP_MRI, origin="lower", alpha=0.3)
    overlay_mask(ax_d, mask_ed, COLOR_ED, alpha=0.5)
    overlay_mask(ax_d, mask_nc, COLOR_NC, alpha=0.6)
    overlay_mask(ax_d, mask_et, COLOR_ET, alpha=0.6)
    # Lobe boundaries
    from scipy.ndimage import sobel
    for lobe_name, lobe_id in LOBE_IDS.items():
        mask = (atlas_slice == lobe_id).astype(float)
        edges = np.hypot(sobel(mask, axis=0), sobel(mask, axis=1)) > 0
        lobe_rgb = plt.matplotlib.colors.to_rgb(LOBE_COLORS[lobe_name])
        edge_overlay = np.zeros((*mask.shape, 4))
        edge_overlay[edges] = (*lobe_rgb, 1.0)
        ax_d.imshow(edge_overlay, interpolation="nearest", origin="lower")
    add_lobe_labels(ax_d, atlas_slice, fontsize=9)
    ax_d.set_title("D  Overlay: MRI + Subregions + Lobes",
                   fontsize=12, fontweight="bold", loc="left")
    ax_d.axis("off")

    plt.tight_layout(pad=1.5)

    # Save
    for fmt, ext, dpi in [("png", "png", 300), ("pdf", "pdf", 300), ("svg", "svg", 150)]:
        path = OUTPUT_DIR / f"anatomical_overlay.{ext}"
        fig.savefig(str(path), dpi=dpi, format=fmt, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"Saved {path}")

    plt.close(fig)
    print("\nDone.")


if __name__ == "__main__":
    main()
