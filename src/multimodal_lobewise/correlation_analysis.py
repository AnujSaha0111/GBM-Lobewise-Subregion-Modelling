#!/usr/bin/env python3
"""Analyse correlation structure and redundancy across multimodal lobewise features.

Computes Pearson and Spearman correlation matrices, identifies highly-correlated
pairs (|r| > 0.80, |r| > 0.90), generates clustered and modality-block heatmaps,
and quantifies within- versus cross-modality redundancy.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

# ── Constants ──

METADATA_COLS = {"patient_id", "risk_label"}
LOBES = ["frontal", "temporal", "parietal", "occipital"]
MODALITIES = ["T1", "T2", "T1GD", "FLAIR"]

DEFAULT_INPUT = "outputs/multimodal_lobewise/processed_features.csv"
DEFAULT_OUTPUT_DIR = "outputs/multimodal_lobewise"

CORRELATION_THRESHOLDS = [0.80, 0.90]


# ── Helpers ──

def _resolve_path(root: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (root / p)


def parse_feature_name(name: str) -> dict[str, str]:
    """Parse a feature name into modality, lobe, and subregion.

    Examples:
        T1GD_frontal_en_ratio   -> modality=T1GD, lobe=frontal,  subregion=en
        T2_global_nc_en_ratio   -> modality=T2,   lobe=global,   subregion=nc_en
    """
    modality, rest = name.split("_", 1)

    for lobe in LOBES:
        prefix = f"{lobe}_"
        if rest.startswith(prefix):
            subregion = rest[len(prefix):].replace("_ratio", "")
            return {"modality": modality, "lobe": lobe, "subregion": subregion}

    if rest.startswith("global_"):
        subregion = rest.replace("global_", "").replace("_ratio", "")
        return {"modality": modality, "lobe": "global", "subregion": subregion}

    if rest == "tumor_burden_index":
        return {"modality": modality, "lobe": "global", "subregion": "tumor_burden"}

    return {"modality": modality, "lobe": "unknown", "subregion": rest}


def _pair_metadata(name_a: str, name_b: str) -> dict[str, bool]:
    """Compare parsed components of two feature names."""
    pa = parse_feature_name(name_a)
    pb = parse_feature_name(name_b)
    return {
        "same_modality": pa["modality"] == pb["modality"],
        "same_lobe": pa["lobe"] == pb["lobe"],
        "same_subregion": pa["subregion"] == pb["subregion"],
    }


# ── Correlation Matrix Computation ──

def compute_correlation_matrices(
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Pearson and Spearman correlation matrices."""
    pearson = X.corr(method="pearson")
    spearman = X.corr(method="spearman")
    return pearson, spearman


# ── High-Correlation Pairs ──

def _extract_upper_tri_pairs(
    corr_matrix: pd.DataFrame, corr_type: str, threshold: float,
) -> list[dict]:
    """Extract upper-triangle pairs with |r| > threshold."""
    pairs = []
    names = corr_matrix.columns
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            val = corr_matrix.iloc[i, j]
            if abs(val) > threshold:
                meta = _pair_metadata(names[i], names[j])
                pairs.append({
                    "feature_a": names[i],
                    "feature_b": names[j],
                    "correlation_type": corr_type,
                    "correlation_value": round(float(val), 6),
                    **meta,
                })
    return pairs


def find_high_correlation_pairs(
    pearson: pd.DataFrame, spearman: pd.DataFrame,
    thresholds: list[float],
) -> pd.DataFrame:
    """Combine all pairs exceeding any threshold for either correlation type."""
    all_pairs: list[dict] = []
    for corr_mat, ctype in [(pearson, "pearson"), (spearman, "spearman")]:
        for thr in thresholds:
            all_pairs.extend(_extract_upper_tri_pairs(corr_mat, ctype, thr))

    if not all_pairs:
        return pd.DataFrame(columns=[
            "feature_a", "feature_b", "correlation_type",
            "correlation_value", "same_lobe", "same_subregion", "same_modality",
        ])

    df = pd.DataFrame(all_pairs)
    df = df.sort_values("correlation_value", ascending=False, key=abs)
    return df.reset_index(drop=True)


# ── Heatmaps ──

def plot_clustered_heatmap(corr_matrix: pd.DataFrame, save_path: Path) -> None:
    """Clustered Pearson correlation heatmap using hierarchical clustering.

    Features are reordered by linkage clustering so that highly-correlated
    groups appear as contiguous blocks, revealing the internal dependency
    structure across modalities, lobes, and subregions.
    """
    # Hierarchical clustering on the distance transform of the correlation matrix
    dist = 1 - corr_matrix.abs()
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="average")

    g = sns.clustermap(
        corr_matrix,
        row_linkage=link,
        col_linkage=link,
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        figsize=(14, 13),
        linewidths=0.0,
        xticklabels=False,
        yticklabels=False,
        dendrogram_ratio=(0.08, 0.08),
        cbar_pos=(0.01, 0.82, 0.03, 0.15),
    )
    g.ax_heatmap.set_xlabel("Features")
    g.ax_heatmap.set_ylabel("Features")
    g.fig.suptitle("Clustered Pearson Correlation — Multimodal Features",
                   fontsize=12, y=1.01)
    g.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(g.fig)


def plot_modality_block_heatmap(corr_matrix: pd.DataFrame, save_path: Path) -> None:
    """Modality-block heatmap where features are grouped by modality.

    Each 16×16 diagonal block shows within-modality correlations; off-diagonal
    blocks show cross-modality correlations. Strong diagonal blocks indicate
    modality-internal redundancy, while bright off-diagonal blocks indicate
    redundant signal shared across different MRI sequences.
    """
    # Build modality-based ordering
    parsed = {col: parse_feature_name(col) for col in corr_matrix.columns}
    mod_order = {m: i for i, m in enumerate(MODALITIES)}

    def sort_key(col: str) -> tuple:
        p = parsed[col]
        return (mod_order.get(p["modality"], 99), p["lobe"], p["subregion"])

    ordered_cols = sorted(corr_matrix.columns, key=sort_key)
    ordered = corr_matrix.loc[ordered_cols, ordered_cols]

    # Modality boundaries for separator lines
    mod_boundaries: list[int] = []
    current = None
    for i, col in enumerate(ordered_cols):
        mod = parsed[col]["modality"]
        if mod != current:
            if current is not None:
                mod_boundaries.append(i)
            current = mod
    mod_boundaries.append(len(ordered_cols))

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        ordered, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        xticklabels=False, yticklabels=False,
        linewidths=0.0,
        cbar_kws={"shrink": 0.8},
    )

    # Modality block separators
    for b in mod_boundaries[:-1]:
        ax.axvline(b, color="black", linewidth=1.2)
        ax.axhline(b, color="black", linewidth=1.2)

    # Modality labels placed at the centre of each block
    block_starts = [0] + mod_boundaries[:-1]
    midpoints = [(s + e) / 2 for s, e in zip(block_starts, mod_boundaries)]
    ax.set_xticks(midpoints)
    ax.set_xticklabels(MODALITIES, rotation=0, fontsize=9)
    ax.set_yticks(midpoints)
    ax.set_yticklabels(MODALITIES, rotation=0, fontsize=9)

    ax.set_xlabel("Modality")
    ax.set_ylabel("Modality")
    ax.set_title("Modality-Block Pearson Correlation", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ── Grouped Summaries ──

def compute_grouped_summaries(
    pearson: pd.DataFrame, spearman: pd.DataFrame,
) -> dict[str, object]:
    """Compute within-modality, cross-modality, and same-subregion cross-modality summaries."""
    names = list(pearson.columns)
    parsed = {n: parse_feature_name(n) for n in names}

    within: list[float] = []
    cross: list[float] = []
    same_sub_cross: list[float] = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pi, pj = parsed[names[i]], parsed[names[j]]
            r_pearson = pearson.iloc[i, j]
            r_spearman = spearman.iloc[i, j]
            r_avg = float(np.mean([abs(r_pearson), abs(r_spearman)]))

            if pi["modality"] == pj["modality"]:
                within.append(r_avg)
            else:
                cross.append(r_avg)
                if pi["subregion"] == pj["subregion"]:
                    same_sub_cross.append(r_avg)

    def _summarise(values: list[float], label: str) -> dict[str, float]:
        if not values:
            return {"label": label, "count": 0, "mean": 0.0, "std": 0.0, "max": 0.0}
        return {
            "label": label,
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "max": float(np.max(values)),
        }

    return {
        "within_modality": _summarise(within, "within_modality"),
        "cross_modality": _summarise(cross, "cross_modality"),
        "same_subregion_cross_modality": _summarise(
            same_sub_cross, "same_subregion_cross_modality",
        ),
    }


# ── Printing ──

def print_summary(
    high_pairs: pd.DataFrame, summary: dict[str, object],
) -> None:
    """Print top redundant patterns and cross-modality findings."""
    print("Top redundant feature groups (highest |r|, Pearson):")
    pearson_pairs = high_pairs[high_pairs["correlation_type"] == "pearson"]
    top = pearson_pairs.drop_duplicates(subset=["feature_a", "feature_b"]).head(15)
    for _, row in top.iterrows():
        tag = ""
        if row["same_modality"]:
            tag = " [same modality]"
        elif row["same_subregion"]:
            tag = " [same subregion]"
        print(f"  r={row['correlation_value']:+.4f}{tag}")
        print(f"    {row['feature_a']}")
        print(f"    {row['feature_b']}")
    print()

    print("Strongest cross-modality redundancy patterns:")
    cross_pairs = pearson_pairs[
        ~pearson_pairs["same_modality"]
    ].drop_duplicates(subset=["feature_a", "feature_b"])
    top_cross = cross_pairs.head(10)
    for _, row in top_cross.iterrows():
        print(f"  r={row['correlation_value']:+.4f}")
        print(f"    {row['feature_a']}")
        print(f"    {row['feature_b']}")
    print()

    print("Grouped correlation summary (mean |r| across Pearson & Spearman):")
    for key in ["within_modality", "cross_modality", "same_subregion_cross_modality"]:
        s = summary[key]
        print(f"  {s['label']}:  {s['count']} pairs,  "
              f"mean |r| = {s['mean']:.4f}  (std = {s['std']:.4f},  max = {s['max']:.4f})")
    print()


# ── Pipeline ──

def run_correlation_analysis(in_csv: Path, output_dir: Path) -> int:
    """Full correlation analysis pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    print(f"Loaded {len(df)} patients, {len(feature_cols)} features")
    print()

    pearson, spearman = compute_correlation_matrices(X)

    # High-correlation pairs
    high_pairs = find_high_correlation_pairs(pearson, spearman, CORRELATION_THRESHOLDS)
    high_pairs.to_csv(output_dir / "high_correlation_pairs.csv", index=False)
    print(f"Wrote {output_dir / 'high_correlation_pairs.csv'}  ({len(high_pairs)} pairs)")

    # Heatmaps
    plot_clustered_heatmap(pearson, output_dir / "correlation_heatmap.png")
    print(f"Wrote {output_dir / 'correlation_heatmap.png'}")

    plot_modality_block_heatmap(pearson, output_dir / "modality_block_heatmap.png")
    print(f"Wrote {output_dir / 'modality_block_heatmap.png'}")

    # Grouped summary
    summary = compute_grouped_summaries(pearson, spearman)
    with (output_dir / "correlation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {output_dir / 'correlation_summary.json'}")
    print()

    # Console
    print_summary(high_pairs, summary)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyse correlation structure and redundancy across multimodal features.",
    )
    parser.add_argument(
        "--input", default=None,
        help=f"Input CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]

    in_csv = _resolve_path(root, args.input or DEFAULT_INPUT)
    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    output_dir = _resolve_path(root, args.output_dir or DEFAULT_OUTPUT_DIR)

    return run_correlation_analysis(in_csv, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
