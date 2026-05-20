#!/usr/bin/env python3
"""Statistical comparison of multimodal lobewise-subregion features between risk groups.

Loads the processed multimodal dataset, splits patients by risk_label, and for
every feature computes group means, standard deviations, Mann-Whitney U p-values,
Benjamini-Hochberg corrected p-values, and Cohen's d effect sizes. Results are
visualised with a volcano plot and boxplots of the top significant features.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ── Constants ──

METADATA_COLS = {"patient_id", "risk_label"}
TARGET_COL = "risk_label"
LABEL_NAMES = ["low-risk", "high-risk"]
LOBES = ["frontal", "temporal", "parietal", "occipital"]

DEFAULT_INPUT = "outputs/multimodal_lobewise/processed_features.csv"
DEFAULT_OUTPUT_DIR = "outputs/multimodal_lobewise"


# ── Helpers ──

def _resolve_path(root: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (root / p)


def parse_feature_name(name: str) -> dict[str, str]:
    """Parse a feature name into modality, lobe, and subregion.

    Examples:
        T1GD_frontal_en_ratio   -> modality=T1GD, lobe=frontal,  subregion=en
        T2_global_nc_en_ratio   -> modality=T2,   lobe=global,   subregion=nc_en
        FLAIR_tumor_burden_index -> modality=FLAIR, lobe=global, subregion=tumor_burden
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


def cohens_d(
    high: pd.Series, low: pd.Series,
) -> float:
    """Cohen's d effect size: standardised mean difference between two groups.

    d = (mean_high - mean_low) / pooled_std

    Interpretation (Cohen, 1988):
        |d| ~ 0.2  small effect
        |d| ~ 0.5  medium effect
        |d| ~ 0.8  large effect

    A positive d means the feature is higher in the high-risk group.
    """
    n1, n2 = len(high), len(low)
    s1, s2 = high.var(ddof=1), low.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((high.mean() - low.mean()) / pooled_std)


# ── Core Analysis ──

def compute_feature_statistics(
    X: pd.DataFrame, y: pd.Series,
) -> pd.DataFrame:
    """For every feature compute group means, std, Mann-Whitney p, Cohen's d, and FDR correction.

    The Mann-Whitney U test (non-parametric) is used because feature
    distributions are often skewed and the test does not assume normality.
    Benjamini-Hochberg FDR correction controls the expected proportion of
    false discoveries among the rejected hypotheses.
    """
    high_mask = y == 1
    low_mask = y == 0

    X_high = X[high_mask]
    X_low = X[low_mask]

    rows = []
    for col in X.columns:
        h = X_high[col].dropna()
        l = X_low[col].dropna()

        # Mann-Whitney U (two-sided)
        stat, p_val = mannwhitneyu(h, l, alternative="two-sided")

        rows.append({
            "feature_name": col,
            "mean_high_risk": float(h.mean()),
            "mean_low_risk": float(l.mean()),
            "std_high_risk": float(h.std(ddof=1)),
            "std_low_risk": float(l.std(ddof=1)),
            "p_value": float(p_val),
            "cohens_d": cohens_d(h, l),
        })

    result = pd.DataFrame(rows)

    # Benjamini-Hochberg FDR correction
    _, corrected, _, _ = multipletests(result["p_value"], method="fdr_bh")
    result["fdr_corrected_p"] = corrected

    # Parse feature names
    parsed = result["feature_name"].apply(parse_feature_name).apply(pd.Series)
    result = pd.concat([result, parsed], axis=1)

    return result.sort_values("p_value")


# ── Plotting ──

def plot_volcano(stats_df: pd.DataFrame, save_path: Path, alpha: float = 0.05) -> None:
    """Volcano plot: Cohen's d (x) vs -log10 FDR-corrected p-value (y).

    Points above the horizontal threshold line are statistically significant
    after FDR correction. Features to the right of zero are elevated in the
    high-risk group; features to the left are elevated in the low-risk group.
    """
    df = stats_df.copy()
    df["neg_log10_p"] = -np.log10(df["fdr_corrected_p"].clip(lower=1e-300))
    df["significant"] = df["fdr_corrected_p"] < alpha

    fig, ax = plt.subplots(figsize=(9, 7))

    for sig, color, label in [
        (False, "grey", "Not significant"),
        (True, "red", f"FDR < {alpha}"),
    ]:
        subset = df[df["significant"] == sig]
        ax.scatter(
            subset["cohens_d"], subset["neg_log10_p"],
            c=color, alpha=0.6, edgecolors="none", s=30, label=label,
        )

    # FDR threshold line
    threshold_y = -np.log10(alpha)
    ax.axhline(y=threshold_y, color="blue", linestyle="--", linewidth=0.8,
               label=f"FDR = {alpha}")

    # Annotate top features
    top = df.nsmallest(10, "p_value")
    for _, row in top.iterrows():
        ax.annotate(
            row["feature_name"],
            (row["cohens_d"], row["neg_log10_p"]),
            fontsize=6, alpha=0.8,
            arrowprops=dict(arrowstyle="-", color="grey", lw=0.5),
        )

    ax.set_xlabel("Cohen's d (high-risk vs low-risk)")
    ax.set_ylabel("-log10(FDR-corrected p-value)")
    ax.set_title("Volcano Plot — Multimodal Lobewise Features")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_top_feature_boxplots(
    X: pd.DataFrame, y: pd.Series, stats_df: pd.DataFrame,
    save_path: Path, n_features: int = 8,
) -> None:
    """Boxplot grid of the *n_features* most statistically significant features."""
    top_features = stats_df.nsmallest(n_features, "p_value")["feature_name"].tolist()
    n_cols = 4
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows))
    axes = axes.flatten()

    for i, feat in enumerate(top_features):
        ax = axes[i]
        data = [X.loc[y == 0, feat].dropna(), X.loc[y == 1, feat].dropna()]
        bp = ax.boxplot(data, labels=LABEL_NAMES, patch_artist=True, widths=0.5)
        bp["boxes"][0].set_facecolor("#4C72B0")
        bp["boxes"][1].set_facecolor("#DD8452")
        ax.set_title(feat, fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(alpha=0.2, axis="y")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Top Significant Features — High-Risk vs Low-Risk", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ── Reporting ──

def print_summary(stats_df: pd.DataFrame) -> None:
    """Print top features by significance and by effect size."""
    print("Top 10 statistically significant features (lowest p-value):")
    top_sig = stats_df.nsmallest(10, "p_value")[
        ["feature_name", "p_value", "fdr_corrected_p", "cohens_d"]
    ]
    for _, row in top_sig.iterrows():
        print(f"  {row['feature_name']:40s}  p={row['p_value']:.2e}  "
              f"fdr={row['fdr_corrected_p']:.2e}  d={row['cohens_d']:+.3f}")
    print()

    print("Top 10 features by absolute effect size (|Cohen's d|):")
    top_es = stats_df.reindex(stats_df["cohens_d"].abs().sort_values(ascending=False).index).head(10)
    for _, row in top_es.iterrows():
        print(f"  {row['feature_name']:40s}  d={row['cohens_d']:+.3f}  "
              f"p={row['p_value']:.2e}  fdr={row['fdr_corrected_p']:.2e}")
    print()

    n_sig = (stats_df["fdr_corrected_p"] < 0.05).sum()
    print(f"Features significant at FDR < 0.05: {n_sig} / {len(stats_df)}")


# ── Pipeline ──

def run_statistical_analysis(in_csv: Path, output_dir: Path) -> int:
    """Load, test, correct, plot, and export the statistical analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    y = df[TARGET_COL].astype(int)
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    print(f"Loaded {len(df)} patients, {len(feature_cols)} features")
    print(f"  High-risk (1): {(y == 1).sum()}")
    print(f"  Low-risk  (0): {(y == 0).sum()}")
    print()

    stats_df = compute_feature_statistics(X, y)

    # Save full statistics table
    out_cols = [
        "feature_name", "modality", "lobe", "subregion",
        "mean_high_risk", "mean_low_risk",
        "std_high_risk", "std_low_risk",
        "p_value", "fdr_corrected_p", "cohens_d",
    ]
    stats_df[out_cols].to_csv(output_dir / "feature_statistics.csv", index=False)
    print(f"Wrote {output_dir / 'feature_statistics.csv'}")

    # Plots
    plot_volcano(stats_df, output_dir / "volcano_plot.png")
    print(f"Wrote {output_dir / 'volcano_plot.png'}")

    plot_top_feature_boxplots(X, y, stats_df, output_dir / "top_feature_boxplots.png")
    print(f"Wrote {output_dir / 'top_feature_boxplots.png'}")
    print()

    # Console summary
    print_summary(stats_df)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Statistical comparison of multimodal lobewise features between risk groups.",
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

    return run_statistical_analysis(in_csv, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
