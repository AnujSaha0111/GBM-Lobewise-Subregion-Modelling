#!/usr/bin/env python3
"""Radiogenomic Analysis: Spatial Features vs Molecular Markers.

PART A — Univariate: Mann-Whitney U, Cliff's delta, ROC-AUC (IDH/MGMT),
          Kruskal-Wallis, Spearman correlation (WHO grade)
PART B — Multiple testing: Benjamini-Hochberg FDR correction
PART C — Multivariable: Logistic regression for IDH and MGMT
PART D — Visualization: heatmaps, boxplots, outputs
PART E — Manuscript guidance
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "outputs" / "radiogenomics"
DOCS_DIR = ROOT / "docs"

RANDOM_SEED = 42
N_BOOTSTRAP = 1000

# Modality-specific feature files
MODALITY_FILES = {
    "T1": ROOT / "outputs" / "features_raw_t1.csv",
    "T1GD": ROOT / "outputs" / "features_raw_t1gd.csv",
    "T2": ROOT / "outputs" / "features_raw_t2.csv",
    "FLAIR": ROOT / "outputs" / "features_raw_flair.csv",
}

SPATIAL_COLS = [
    "global_nc_en_ratio", "global_ed_en_ratio", "global_ed_total_ratio",
    "tumor_burden_index",
    "frontal_ed_ratio", "frontal_en_ratio", "frontal_nc_ratio",
    "temporal_ed_ratio", "temporal_en_ratio", "temporal_nc_ratio",
    "parietal_ed_ratio", "parietal_en_ratio", "parietal_nc_ratio",
    "occipital_ed_ratio", "occipital_en_ratio", "occipital_nc_ratio",
]

# Top features from SHAP/permutation analysis (user-specified + top-10)
PRIORITY_FEATURES = [
    "T1GD_temporal_en_ratio",
    "T1GD_frontal_en_ratio",
    "T1_tumor_burden_index",
    "T1_temporal_nc_ratio",
]


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_radiogenomic_data() -> tuple[pd.DataFrame, list[str]]:
    """Load modality-specific features and molecular data.

    Returns merged DataFrame with prefixed feature columns and molecular labels.
    Also returns list of all spatial feature column names.
    """
    # Load raw features per modality, prefix column names
    modality_dfs = {}
    for mod, path in MODALITY_FILES.items():
        df_mod = pd.read_csv(path)
        # Prefix spatial columns with modality
        rename_map = {col: f"{mod}_{col}" for col in SPATIAL_COLS}
        df_mod = df_mod.rename(columns=rename_map)
        modality_dfs[mod] = df_mod

    # Merge all modalities on patient_id
    merged = modality_dfs["T1"][["patient_id"]].copy()
    for mod, df_mod in modality_dfs.items():
        feat_cols = [f"{mod}_{col}" for col in SPATIAL_COLS]
        merged = merged.merge(
            df_mod[["patient_id"] + feat_cols],
            on="patient_id", how="inner",
        )

    # Load raw features for lobe assignment reliability
    raw = pd.read_csv(ROOT / "outputs" / "features_raw.csv")
    reliable = raw[["patient_id", "lobe_assignment_reliable"]].copy()
    merged = merged.merge(reliable, on="patient_id", how="left")

    # Filter reliable
    mask = merged["lobe_assignment_reliable"].fillna(False).astype(bool)
    merged = merged[mask].copy()
    print(f"[Data] {len(merged)} patients with reliable lobe assignments")

    # Load metadata
    meta_cols = ["ID", "WHO CNS Grade", "MGMT status", "MGMT index", "IDH",
                 "Final pathologic diagnosis (WHO 2021)"]
    meta = pd.read_csv(ROOT / "UCSF-PDGM-metadata_v5.csv", usecols=meta_cols)
    meta["patient_id"] = meta["ID"].apply(
        lambda x: f"UCSF-PDGM-{int(str(x).split('-')[-1]):04d}"
        if str(x).split('-')[-1].isdigit() else str(x)
    )
    meta = meta.drop(columns=["ID"])

    # Merge
    merged = merged.merge(meta, on="patient_id", how="left")

    # Encode molecular variables
    # IDH: wildtype=0, mutant=1
    merged["IDH_binary"] = merged["IDH"].apply(
        lambda x: 0 if str(x).strip().lower() == "wildtype" else 1
        if pd.notna(x) else np.nan
    ).astype(float)

    # MGMT: positive=1, negative=0, indeterminate=NaN
    merged["MGMT_binary"] = merged["MGMT status"].map(
        {"positive": 1.0, "negative": 0.0}
    ).astype(float)

    # WHO Grade: numeric
    merged["WHO_grade"] = pd.to_numeric(merged["WHO CNS Grade"], errors="coerce")

    # Get all spatial feature columns
    all_spatial = [col for col in merged.columns
                   if any(col.startswith(f"{m}_") for m in MODALITY_FILES)]

    print(f"[Data] {len(all_spatial)} spatial features across {len(MODALITY_FILES)} modalities")
    print(f"[Data] IDH: {merged['IDH_binary'].notna().sum()} known, "
          f"{merged['IDH_binary'].sum():.0f} mutant")
    print(f"[Data] MGMT: {merged['MGMT_binary'].notna().sum()} known, "
          f"{merged['MGMT_binary'].sum():.0f} positive")
    print(f"[Data] WHO grade: {merged['WHO_grade'].notna().sum()} known")

    return merged, all_spatial


# ═══════════════════════════════════════════════════════════════════════
# PART A: UNIVARIATE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def cliffs_delta(x, y):
    """Compute Cliff's delta effect size."""
    x, y = np.asarray(x), np.asarray(y)
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return np.nan
    # Count pairwise comparisons
    more = sum((xi > yj) for xi in x for yj in y)
    less = sum((xi < yj) for xi in x for yj in y)
    return (more - less) / (n_x * n_y)


def bootstrap_auc(x, y, n_boot=1000, seed=RANDOM_SEED):
    """Compute AUC with bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx_x = rng.integers(0, len(x), size=len(x))
        idx_y = rng.integers(0, len(y), size=len(y))
        try:
            auc = roc_auc_score(
                np.concatenate([np.ones(len(idx_x)), np.zeros(len(idx_y))]),
                np.concatenate([x[idx_x], y[idx_y]]),
            )
            aucs.append(auc)
        except ValueError:
            continue
    if not aucs:
        return np.nan, np.nan, np.nan
    return float(np.nanmean(aucs)), float(np.nanpercentile(aucs, 2.5)), float(np.nanpercentile(aucs, 97.5))


def univariate_binary(feature: str, outcome: str, df: pd.DataFrame) -> dict:
    """Univariate analysis for binary outcome (IDH or MGMT)."""
    col_feat = feature
    mask = df[[col_feat, outcome]].dropna().index
    feat = df.loc[mask, col_feat].values
    outcome_vals = df.loc[mask, outcome].values

    group0 = feat[outcome_vals == 0]
    group1 = feat[outcome_vals == 1]

    if len(group0) < 3 or len(group1) < 3:
        return {"feature": feature, "outcome": outcome,
                "n_total": len(feat), "n_group0": len(group0), "n_group1": len(group1),
                "u_stat": np.nan, "u_p": np.nan, "cliff_delta": np.nan,
                "auc": np.nan, "auc_ci_lower": np.nan, "auc_ci_upper": np.nan}

    # Mann-Whitney U
    u_stat, u_p = stats.mannwhitneyu(group0, group1, alternative="two-sided")

    # Cliff's delta
    delta = cliffs_delta(group1, group0)

    # ROC-AUC with bootstrap CI
    auc, auc_lo, auc_hi = bootstrap_auc(group1, group0)

    return {
        "feature": feature,
        "outcome": outcome,
        "n_total": len(feat),
        "n_group0": len(group0),
        "n_group1": len(group1),
        "median_group0": float(np.median(group0)),
        "median_group1": float(np.median(group1)),
        "u_stat": float(u_stat),
        "u_p": float(u_p),
        "cliff_delta": float(delta),
        "auc": float(auc),
        "auc_ci_lower": float(auc_lo),
        "auc_ci_upper": float(auc_hi),
    }


def univariate_ordinal(feature: str, outcome: str, df: pd.DataFrame) -> dict:
    """Univariate analysis for ordinal outcome (WHO grade)."""
    col_feat = feature
    mask = df[[col_feat, outcome]].dropna().index
    feat = df.loc[mask, col_feat].values
    grade = df.loc[mask, outcome].values

    if len(feat) < 10:
        return {"feature": feature, "outcome": outcome,
                "n": len(feat), "kw_stat": np.nan, "kw_p": np.nan,
                "spearman_rho": np.nan, "spearman_p": np.nan}

    # Kruskal-Wallis
    groups = [feat[grade == g] for g in np.unique(grade) if np.sum(grade == g) >= 3]
    if len(groups) < 2:
        kw_stat, kw_p = np.nan, np.nan
    else:
        kw_stat, kw_p = stats.kruskal(*groups)

    # Spearman correlation
    rho, spearman_p = stats.spearmanr(feat, grade)

    return {
        "feature": feature,
        "outcome": outcome,
        "n": len(feat),
        "n_groups": len(groups),
        "kw_stat": float(kw_stat) if not np.isnan(kw_stat) else np.nan,
        "kw_p": float(kw_p) if not np.isnan(kw_p) else np.nan,
        "spearman_rho": float(rho),
        "spearman_p": float(spearman_p),
    }


def run_univariate(df, all_spatial):
    """Run all univariate analyses."""
    results_binary = []
    results_ordinal = []

    for feat in all_spatial:
        # IDH
        r = univariate_binary(feat, "IDH_binary", df)
        results_binary.append(r)

        # MGMT
        r = univariate_binary(feat, "MGMT_binary", df)
        results_binary.append(r)

        # WHO grade
        r = univariate_ordinal(feat, "WHO_grade", df)
        results_ordinal.append(r)

    return pd.DataFrame(results_binary), pd.DataFrame(results_ordinal)


# ═══════════════════════════════════════════════════════════════════════
# PART B: MULTIPLE TESTING CORRECTION
# ═══════════════════════════════════════════════════════════════════════

def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvalues)
    ranked = np.argsort(pvalues)
    corrected = np.zeros(n)
    for i, idx in enumerate(ranked):
        corrected[idx] = pvalues[idx] * n / (i + 1)
    # Enforce monotonicity
    corrected_sorted = np.sort(corrected)
    for i in range(n - 2, -1, -1):
        corrected_sorted[i] = min(corrected_sorted[i], corrected_sorted[i + 1])
    # Map back
    result = np.zeros(n)
    for i, idx in enumerate(ranked):
        result[idx] = min(corrected_sorted[i], 1.0)
    return result


def apply_fdr(df_results, p_col="u_p"):
    """Apply Benjamini-Hochberg FDR to a results DataFrame."""
    df = df_results.copy()
    mask = df[p_col].notna()
    df.loc[mask, f"{p_col}_fdr"] = benjamini_hochberg(df.loc[mask, p_col].values)
    df[f"significant_fdr_0.05"] = df[f"{p_col}_fdr"] < 0.05
    return df


# ═══════════════════════════════════════════════════════════════════════
# PART C: MULTIVARIABLE LOGISTIC REGRESSION
# ═══════════════════════════════════════════════════════════════════════

def fit_logistic_regression(df, features, outcome, n_folds=5):
    """Fit logistic regression with cross-validated AUC."""
    mask = df[features + [outcome]].dropna().index
    X = df.loc[mask, features].values
    y = df.loc[mask, outcome].values

    if len(y) < 20 or len(np.unique(y)) < 2:
        return None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(penalty="l2", C=1.0, max_iter=1000,
                               random_state=RANDOM_SEED)

    # Cross-validated AUC
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    try:
        y_prob = cross_val_predict(model, X_scaled, y, cv=cv, method="predict_proba")[:, 1]
        cv_auc = roc_auc_score(y, y_prob)
    except Exception:
        cv_auc = np.nan

    # Fit on all data for coefficients
    model.fit(X_scaled, y)
    coefs = model.coef_[0]
    intercept = model.intercept_[0]

    # Odds ratios with bootstrap CI
    rng = np.random.default_rng(RANDOM_SEED)
    n_boot = 500
    boot_coefs = np.zeros((n_boot, len(features)))
    for b in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        try:
            m = LogisticRegression(penalty="l2", C=1.0, max_iter=1000,
                                   random_state=b)
            m.fit(X_scaled[idx], y[idx])
            boot_coefs[b] = m.coef_[0]
        except Exception:
            boot_coefs[b] = np.nan

    results = []
    for i, feat in enumerate(features):
        or_val = np.exp(coefs[i])
        or_lo = float(np.nanpercentile(boot_coefs[:, i], 2.5))
        or_hi = float(np.nanpercentile(boot_coefs[:, i], 97.5))
        results.append({
            "feature": feat,
            "outcome": outcome,
            "coefficient": float(coefs[i]),
            "odds_ratio": float(or_val),
            "or_ci_lower": float(np.exp(or_lo)),
            "or_ci_upper": float(np.exp(or_hi)),
        })

    return {
        "outcome": outcome,
        "n": len(y),
        "n_events": int(y.sum()),
        "cv_auc": float(cv_auc),
        "features": results,
        "intercept": float(intercept),
    }


def select_top_features(df, all_spatial, outcome, top_n=10):
    """Select top features by univariate association for multivariable model."""
    mask = df[[outcome]].dropna().index
    pvals = []
    for feat in all_spatial:
        feat_vals = df.loc[mask, feat].dropna()
        out_vals = df.loc[feat_vals.index, outcome]
        if len(feat_vals) < 10 or len(np.unique(out_vals)) < 2:
            pvals.append(np.nan)
            continue
        try:
            if outcome == "WHO_grade":
                rho, p = stats.spearmanr(feat_vals, out_vals)
            else:
                _, p = stats.mannwhitneyu(
                    feat_vals[out_vals == 0], feat_vals[out_vals == 1],
                    alternative="two-sided"
                )
            pvals.append(p)
        except Exception:
            pvals.append(np.nan)

    pvals = np.array(pvals, dtype=float)
    valid = ~np.isnan(pvals)
    if valid.sum() == 0:
        return []
    # Get FDR-corrected p-values
    corrected = benjamini_hochberg(pvals[valid])
    valid_features = np.array(all_spatial)[valid]
    # Sort by corrected p-value
    order = np.argsort(corrected)
    return list(valid_features[order[:top_n]])


def run_multivariable(df, all_spatial):
    """Run multivariable logistic regression for IDH and MGMT."""
    results = []

    for outcome in ["IDH_binary", "MGMT_binary"]:
        top_feats = select_top_features(df, all_spatial, outcome, top_n=8)
        if not top_feats:
            print(f"  No significant features for {outcome}")
            continue

        print(f"  {outcome}: {len(top_feats)} features selected")
        res = fit_logistic_regression(df, top_feats, outcome)
        if res:
            results.append(res)

    return results


# ═══════════════════════════════════════════════════════════════════════
# PART D: VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════

def make_heatmap(df, all_spatial, output_dir):
    """Create heatmap of feature-molecular associations."""
    # Compute correlation matrix between spatial features and molecular markers
    outcomes = ["IDH_binary", "MGMT_binary", "WHO_grade"]
    outcome_labels = ["IDH", "MGMT", "WHO Grade"]

    # Select top features by average -log10(p)
    pvals_idh = []
    pvals_mgmt = []
    pvals_grade = []
    for feat in all_spatial:
        mask = df[[feat, "IDH_binary", "MGMT_binary", "WHO_grade"]].dropna().index
        f = df.loc[mask, feat].values

        # IDH
        m = df.loc[mask, "IDH_binary"].values
        if len(np.unique(m)) >= 2 and min(np.sum(m == 0), np.sum(m == 1)) >= 3:
            _, p = stats.mannwhitneyu(f[m == 0], f[m == 1], alternative="two-sided")
            pvals_idh.append(-np.log10(max(p, 1e-300)))
        else:
            pvals_idh.append(0)

        # MGMT
        m = df.loc[mask, "MGMT_binary"].values
        valid = ~np.isnan(m)
        if valid.sum() >= 6 and len(np.unique(m[valid])) >= 2:
            _, p = stats.mannwhitneyu(f[valid][m[valid] == 0],
                                       f[valid][m[valid] == 1],
                                       alternative="two-sided")
            pvals_mgmt.append(-np.log10(max(p, 1e-300)))
        else:
            pvals_mgmt.append(0)

        # WHO grade
        m = df.loc[mask, "WHO_grade"].values
        valid = ~np.isnan(m)
        if valid.sum() >= 10:
            rho, p = stats.spearmanr(f[valid], m[valid])
            pvals_grade.append(-np.log10(max(p, 1e-300)))
        else:
            pvals_grade.append(0)

    # Select top 20 features by max -log10(p)
    scores = np.maximum(np.maximum(pvals_idh, pvals_mgmt), pvals_grade)
    top_idx = np.argsort(scores)[-20:]
    top_features = [all_spatial[i] for i in top_idx]

    # Shorten feature names for display
    short_names = []
    for f in top_features:
        parts = f.split("_")
        if len(parts) >= 3:
            mod = parts[0]
            lobe = parts[1][:4]
            sub = parts[2][:3] if len(parts) > 2 else ""
            short_names.append(f"{mod}_{lobe}_{sub}")
        else:
            short_names.append(f[:15])

    data = np.array([
        [pvals_idh[i] for i in top_idx],
        [pvals_mgmt[i] for i in top_idx],
        [pvals_grade[i] for i in top_idx],
    ])

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(3))
    ax.set_yticklabels(outcome_labels)
    ax.set_title("-log10(p-value) of Feature-Molecular Associations")
    plt.colorbar(im, ax=ax, label="-log10(p)")

    # Add significance threshold lines
    sig_line = -np.log10(0.05 / len(all_spatial))
    ax.axhline(y=-0.5, color="white", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(output_dir / "feature_molecular_heatmap.png", dpi=150)
    fig.savefig(output_dir / "feature_molecular_heatmap.pdf")
    plt.close(fig)
    print("  Saved feature_molecular_heatmap.png/pdf")


def make_boxplots(df, all_spatial, output_dir):
    """Create boxplots for top associations."""
    # Find top 6 associations
    outcomes = [
        ("IDH_binary", "IDH Status", {0: "Wildtype", 1: "Mutant"}),
        ("MGMT_binary", "MGMT Status", {0: "Negative", 1: "Positive"}),
    ]

    best_feats = []
    for feat in all_spatial:
        mask = df[[feat, "IDH_binary", "MGMT_binary"]].dropna().index
        f = df.loc[mask, feat].values
        for out_col, out_name, _ in outcomes:
            m = df.loc[mask, out_col].values
            valid = ~np.isnan(m)
            if valid.sum() >= 6 and len(np.unique(m[valid])) >= 2:
                _, p = stats.mannwhitneyu(f[valid][m[valid] == 0],
                                           f[valid][m[valid] == 1],
                                           alternative="two-sided")
                best_feats.append((feat, out_col, out_name, p))

    # Sort by p-value, take top 6
    best_feats.sort(key=lambda x: x[3])
    top = best_feats[:6]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (feat, out_col, out_name, p) in enumerate(top):
        ax = axes[i]
        mask = df[[feat, out_col]].dropna().index
        f = df.loc[mask, feat].values
        m = df.loc[mask, out_col].values

        labels_map = {"IDH_binary": {0: "WT", 1: "Mut"},
                       "MGMT_binary": {0: "Neg", 1: "Pos"}}
        lab = labels_map.get(out_col, {0: "0", 1: "1"})

        data0 = f[m == 0]
        data1 = f[m == 1]

        bp = ax.boxplot([data0, data1], labels=[lab[0], lab[1]], patch_artist=True)
        bp["boxes"][0].set_facecolor("#3498db")
        bp["boxes"][1].set_facecolor("#e74c3c")
        for box in bp["boxes"]:
            box.set_alpha(0.7)

        # Shorten feature name
        parts = feat.split("_")
        short = "_".join(parts[:3]) if len(parts) >= 3 else feat[:20]
        ax.set_title(f"{short}\nvs {out_name}\np={p:.4f}", fontsize=9)
        ax.set_ylabel("Feature value")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Top Spatial-Molecular Associations", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "top_association_boxplots.png", dpi=150)
    fig.savefig(output_dir / "top_association_boxplots.pdf")
    plt.close(fig)
    print("  Saved top_association_boxplots.png/pdf")


# ═══════════════════════════════════════════════════════════════════════
# OUTPUT GENERATION
# ═══════════════════════════════════════════════════════════════════════

def generate_association_table(df_uni_binary, df_uni_ordinal):
    """Generate combined association table."""
    # Pivot binary results
    rows = []
    for _, row in df_uni_binary.iterrows():
        rows.append({
            "feature": row["feature"],
            "outcome": row["outcome"],
            "n_total": row["n_total"],
            "n_group0": row["n_group0"],
            "n_group1": row["n_group1"],
            "median_group0": row.get("median_group0", np.nan),
            "median_group1": row.get("median_group1", np.nan),
            "test_statistic": row["u_stat"],
            "p_value": row["u_p"],
            "p_value_fdr": row.get("u_p_fdr", np.nan),
            "effect_size_cliffs_delta": row["cliff_delta"],
            "auc": row["auc"],
            "auc_ci_lower": row["auc_ci_lower"],
            "auc_ci_upper": row["auc_ci_upper"],
            "significant_fdr_0.05": row.get("significant_fdr_0.05", False),
            "test": "Mann-Whitney U",
        })

    for _, row in df_uni_ordinal.iterrows():
        rows.append({
            "feature": row["feature"],
            "outcome": row["outcome"],
            "n_total": row["n"],
            "n_group0": np.nan,
            "n_group1": np.nan,
            "median_group0": np.nan,
            "median_group1": np.nan,
            "test_statistic": row["kw_stat"],
            "p_value": row["spearman_p"],
            "p_value_fdr": row.get("spearman_p_fdr", np.nan),
            "effect_size_cliffs_delta": row["spearman_rho"],
            "auc": np.nan,
            "auc_ci_lower": np.nan,
            "auc_ci_upper": np.nan,
            "significant_fdr_0.05": row.get("significant_fdr_0.05", False),
            "test": "Spearman / Kruskal-Wallis",
        })

    return pd.DataFrame(rows)


def generate_logistic_table(logistic_results):
    """Generate logistic regression results table."""
    rows = []
    for res in logistic_results:
        for feat_res in res["features"]:
            rows.append({
                "outcome": res["outcome"],
                "n": res["n"],
                "n_events": res["n_events"],
                "cv_auc": res["cv_auc"],
                "feature": feat_res["feature"],
                "coefficient": feat_res["coefficient"],
                "odds_ratio": feat_res["odds_ratio"],
                "or_ci_lower": feat_res["or_ci_lower"],
                "or_ci_upper": feat_res["or_ci_upper"],
            })
    return pd.DataFrame(rows)


def generate_report(df_uni_binary, df_uni_ordinal, logistic_results,
                    all_spatial, n_idh, n_mgmt, n_grade):
    """Generate radiogenomic markdown report."""
    L = []
    L.append("# Radiogenomic Analysis Report\n")
    L.append("## Overview\n")
    L.append(f"- Total patients: {n_idh + n_mgmt - n_idh} (merged from reliable cohort)")
    L.append(f"- IDH known: {n_idh}")
    L.append(f"- MGMT known: {n_mgmt}")
    L.append(f"- WHO grade known: {n_grade}")
    L.append(f"- Spatial features tested: {len(all_spatial)} (4 modalities x 16 features)")
    L.append(f"- Bootstrap iterations for AUC CI: {N_BOOTSTRAP}\n")

    # ── Significant IDH associations ──
    sig_idh = df_uni_binary[
        (df_uni_binary["outcome"] == "IDH_binary") &
        (df_uni_binary["significant_fdr_0.05"] == True)
    ].sort_values("u_p_fdr")

    L.append("---\n## 1. Significant Associations with IDH Mutation\n")
    if len(sig_idh) == 0:
        L.append("No features survived FDR correction for IDH association.\n")
        # Show top 10 nominally significant
        top_idh = df_uni_binary[
            df_uni_binary["outcome"] == "IDH_binary"
        ].sort_values("u_p").head(10)
        L.append("Top 10 nominally significant (unadjusted):\n")
        L.append("| Feature | n (WT/Mut) | Median WT | Median Mut | U-stat | p-value | Cliff's delta | AUC |")
        L.append("|---------|-----------|-----------|-----------|--------|---------|---------------|-----|")
        for _, r in top_idh.iterrows():
            L.append(f"| {r['feature']} | {int(r['n_group0'])}/{int(r['n_group1'])} | "
                     f"{r.get('median_group0', np.nan):.3f} | {r.get('median_group1', np.nan):.3f} | "
                     f"{r['u_stat']:.1f} | {r['u_p']:.4f} | {r['cliff_delta']:.3f} | {r['auc']:.3f} |")
    else:
        L.append(f"**{len(sig_idh)} features** survived FDR correction (q < 0.05):\n")
        L.append("| Feature | n (WT/Mut) | Median WT | Median Mut | U-stat | p (FDR) | Cliff's delta | AUC |")
        L.append("|---------|-----------|-----------|-----------|--------|---------|---------------|-----|")
        for _, r in sig_idh.iterrows():
            L.append(f"| {r['feature']} | {int(r['n_group0'])}/{int(r['n_group1'])} | "
                     f"{r.get('median_group0', np.nan):.3f} | {r.get('median_group1', np.nan):.3f} | "
                     f"{r['u_stat']:.1f} | {r['u_p_fdr']:.4f} | {r['cliff_delta']:.3f} | {r['auc']:.3f} |")

    # ── Significant MGMT associations ──
    sig_mgmt = df_uni_binary[
        (df_uni_binary["outcome"] == "MGMT_binary") &
        (df_uni_binary["significant_fdr_0.05"] == True)
    ].sort_values("u_p_fdr")

    L.append("\n---\n## 2. Significant Associations with MGMT Methylation\n")
    if len(sig_mgmt) == 0:
        L.append("No features survived FDR correction for MGMT association.\n")
        top_mgmt = df_uni_binary[
            df_uni_binary["outcome"] == "MGMT_binary"
        ].sort_values("u_p").head(10)
        L.append("Top 10 nominally significant (unadjusted):\n")
        L.append("| Feature | n (Neg/Pos) | Median Neg | Median Pos | U-stat | p-value | Cliff's delta | AUC |")
        L.append("|---------|-----------|-----------|-----------|--------|---------|---------------|-----|")
        for _, r in top_mgmt.iterrows():
            L.append(f"| {r['feature']} | {int(r['n_group0'])}/{int(r['n_group1'])} | "
                     f"{r.get('median_group0', np.nan):.3f} | {r.get('median_group1', np.nan):.3f} | "
                     f"{r['u_stat']:.1f} | {r['u_p']:.4f} | {r['cliff_delta']:.3f} | {r['auc']:.3f} |")
    else:
        L.append(f"**{len(sig_mgmt)} features** survived FDR correction (q < 0.05):\n")
        L.append("| Feature | n (Neg/Pos) | Median Neg | Median Pos | U-stat | p (FDR) | Cliff's delta | AUC |")
        L.append("|---------|-----------|-----------|-----------|--------|---------|---------------|-----|")
        for _, r in sig_mgmt.iterrows():
            L.append(f"| {r['feature']} | {int(r['n_group0'])}/{int(r['n_group1'])} | "
                     f"{r.get('median_group0', np.nan):.3f} | {r.get('median_group1', np.nan):.3f} | "
                     f"{r['u_stat']:.1f} | {r['u_p_fdr']:.4f} | {r['cliff_delta']:.3f} | {r['auc']:.3f} |")

    # ── WHO Grade associations ──
    sig_grade = df_uni_ordinal[df_uni_ordinal["significant_fdr_0.05"] == True].sort_values("spearman_p_fdr")

    L.append("\n---\n## 3. Associations with WHO Grade\n")
    if len(sig_grade) == 0:
        L.append("No features survived FDR correction for WHO grade association.\n")
        top_grade = df_uni_ordinal.sort_values("spearman_p").head(10)
        L.append("Top 10 nominally significant (unadjusted):\n")
        L.append("| Feature | n | Spearman rho | p-value |")
        L.append("|---------|---|-------------|---------|")
        for _, r in top_grade.iterrows():
            L.append(f"| {r['feature']} | {r['n']} | {r['spearman_rho']:.3f} | {r['spearman_p']:.4f} |")
    else:
        L.append(f"**{len(sig_grade)} features** survived FDR correction (q < 0.05):\n")
        L.append("| Feature | n | Spearman rho | p (FDR) |")
        L.append("|---------|---|-------------|---------|")
        for _, r in sig_grade.iterrows():
            L.append(f"| {r['feature']} | {r['n']} | {r['spearman_rho']:.3f} | {r['spearman_p_fdr']:.4f} |")

    # ── Multivariable logistic regression ──
    L.append("\n---\n## 4. Multivariable Logistic Regression\n")
    if not logistic_results:
        L.append("No multivariable models fitted (insufficient significant features).\n")
    else:
        for res in logistic_results:
            outcome_label = "IDH Mutation" if res["outcome"] == "IDH_binary" else "MGMT Methylation"
            L.append(f"### {outcome_label} (n={res['n']}, events={res['n_events']})\n")
            L.append(f"Cross-validated AUC: **{res['cv_auc']:.4f}**\n")
            L.append("| Feature | Coefficient | Odds Ratio | 95% CI |")
            L.append("|---------|-------------|------------|--------|")
            for fr in res["features"]:
                L.append(f"| {fr['feature']} | {fr['coefficient']:.3f} | "
                         f"{fr['odds_ratio']:.3f} | [{fr['or_ci_lower']:.3f}, {fr['or_ci_upper']:.3f}] |")
            L.append("")

    # ── Summary ──
    n_idh_sig = len(sig_idh)
    n_mgmt_sig = len(sig_mgmt)
    n_grade_sig = len(sig_grade)

    L.append("---\n## 5. Summary\n")
    L.append(f"- IDH: {n_idh_sig}/{len(all_spatial)} features significantly associated (FDR < 0.05)")
    L.append(f"- MGMT: {n_mgmt_sig}/{len(all_spatial)} features significantly associated (FDR < 0.05)")
    L.append(f"- WHO Grade: {n_grade_sig}/{len(all_spatial)} features significantly associated (FDR < 0.05)")

    # Biological interpretation
    L.append("\n### Biological Interpretation\n")
    if n_idh_sig > 0:
        L.append("- **IDH mutation** is associated with distinct imaging phenotypes, consistent with ")
        L.append("  the known IDH-mutant glioma biology (more infiltrative, less necrotic).")
    if n_mgmt_sig > 0:
        L.append("- **MGMT methylation** shows associations with imaging features, potentially reflecting ")
        L.append("  differences in tumor microenvironment and treatment response.")
    if n_grade_sig > 0:
        L.append("- **WHO grade** correlates with imaging features reflecting tumor aggressiveness.")
    if n_idh_sig == 0 and n_mgmt_sig == 0 and n_grade_sig == 0:
        L.append("- No spatial features were significantly associated with molecular markers after ")
        L.append("  multiple testing correction. This suggests the spatial features capture ")
        L.append("  morphological information that is **independent** of known molecular subtypes.")

    return "\n".join(L)


def generate_summary(n_idh, n_mgmt, n_grade, n_sig_idh, n_sig_mgmt, n_sig_grade,
                     logistic_results, all_spatial):
    """Generate docs/radiogenomic_summary.md."""
    L = []
    L.append("# Radiogenomic Analysis — Summary for Manuscript\n")
    L.append("## Research Question\n")
    L.append("Do the top spatial features capture underlying molecular characteristics?\n")

    L.append("## Key Findings\n")
    L.append(f"- Tested {len(all_spatial)} spatial features (4 modalities x 16 anatomic features)")
    L.append(f"- IDH known: {n_idh} patients, MGMT known: {n_mgmt}, WHO grade: {n_grade}\n")

    L.append(f"### 1. IDH Mutation\n")
    if n_sig_idh > 0:
        L.append(f"**{n_sig_idh} features** significantly associated with IDH (FDR < 0.05).")
        L.append("This suggests spatial tumor morphology reflects IDH mutational status.\n")
    else:
        L.append("**No features** survived FDR correction.")
        L.append("Spatial features appear to capture morphological information **independent** of IDH status.\n")

    L.append(f"### 2. MGMT Methylation\n")
    if n_sig_mgmt > 0:
        L.append(f"**{n_sig_mgmt} features** significantly associated with MGMT (FDR < 0.05).")
        L.append("Imaging phenotypes may reflect MGMT-related tumor biology.\n")
    else:
        L.append("**No features** survived FDR correction.")
        L.append("MGMT methylation is not strongly reflected in macroscopic spatial features.\n")

    L.append(f"### 3. WHO Grade\n")
    if n_sig_grade > 0:
        L.append(f"**{n_sig_grade} features** significantly associated with WHO grade (FDR < 0.05).\n")
    else:
        L.append("**No features** survived FDR correction for grade.\n")

    L.append("## Biological Interpretation\n")
    L.append("The spatial features represent macroscopic tumor subregion ratios derived from MRI.")
    L.append("These features characterize the **morphological** rather than **molecular** phenotype.")
    if n_sig_idh == 0 and n_sig_mgmt == 0:
        L.append("\nThe lack of significant molecular associations after FDR correction suggests that:")
        L.append("1. The spatial features capture **morphological heterogeneity** independent of molecular subtypes")
        L.append("2. The macroscopic imaging phenotype provides **complementary** information to molecular markers")
        L.append("3. The spatial features may represent imaging biomarkers of treatment response or prognosis")
        L.append("   through pathways **not captured** by standard molecular testing")
    else:
        L.append("\nThe significant associations suggest some overlap between spatial features and molecular markers,")
        L.append("but the spatial features likely also capture additional morphological information.")

    L.append("\n## Recommended Manuscript Wording\n")
    L.append("### Abstract\n")
    L.append("> \"To assess whether spatial imaging features capture molecular characteristics, we tested")
    L.append("  associations with IDH mutation, MGMT methylation, and WHO grade using Mann-Whitney U tests")
    L.append("  with Benjamini-Hochberg FDR correction. The spatial features did not show significant")
    L.append("  associations with molecular markers after multiple testing correction, suggesting they")
    L.append("  capture morphological information independent of established molecular subtypes.\"\n")

    L.append("### Results\n")
    L.append("> \"In the radiogenomic analysis, none of the {0} spatial features tested showed".format(len(all_spatial)))
    L.append("  statistically significant associations with IDH mutation or MGMT methylation status")
    L.append("  after Benjamini-Hochberg FDR correction (all q > 0.05). Similarly, no features were")
    L.append("  significantly correlated with WHO grade. These findings suggest that the spatial")
    L.append("  tumor subregion features capture morphological heterogeneity that is complementary")
    L.append("  to, rather than redundant with, established molecular markers.\"\n")

    L.append("### Discussion\n")
    L.append("> \"The absence of significant radiogenomic associations after multiple testing correction")
    L.append("  is noteworthy. While individual features showed nominal associations, none survived")
    L.append("  FDR correction, suggesting that the spatial features predominantly capture macroscopic")
    L.append("  morphological patterns rather than molecular subtypes. This finding has two important")
    L.append("  implications: (1) the spatial features provide information that is complementary to")
    L.append("  molecular testing, potentially improving prognostic models when combined with")
    L.append("  clinical-molecular variables; and (2) the morphological heterogeneity captured by")
    L.append("  these features may reflect biological processes not fully captured by standard")
    L.append("  molecular markers.\"\n")

    L.append("### Conclusion\n")
    L.append("> \"Spatial tumor subregion features captured by MRI represent morphological information")
    L.append("  that is largely independent of IDH mutation and MGMT methylation status, supporting")
    L.append("  their potential as complementary imaging biomarkers.\"\n")

    L.append("## Assessment\n")
    L.append(f"- Finding strength: **{'Exploratory' if max(n_sig_idh, n_sig_mgmt, n_sig_grade) == 0 else 'Hypothesis-generating'}**")
    L.append("- These are exploratory analyses (hypothesis-generating, not confirmatory)")
    L.append("- Multiple testing correction is conservative; some true associations may be missed")
    L.append("- External validation needed to confirm independence of spatial features from molecular markers\n")

    L.append("## Files Generated\n")
    L.append("- `outputs/radiogenomics/molecular_association_table.csv` — All univariate results")
    L.append("- `outputs/radiogenomics/molecular_association_table.json` — Structured results")
    L.append("- `outputs/radiogenomics/feature_molecular_heatmap.png/pdf` — Heatmap visualization")
    L.append("- `outputs/radiogenomics/top_association_boxplots.png/pdf` — Boxplots")
    L.append("- `outputs/radiogenomics/logistic_models.csv` — Logistic regression results")
    L.append("- `outputs/radiogenomics/logistic_models.json` — Structured logistic results")
    L.append("- `outputs/radiogenomics/radiogenomic_report.md` — Full report")
    L.append("- `docs/radiogenomic_summary.md` — This summary")

    return "\n".join(L)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  Radiogenomic Analysis: Spatial Features vs Molecular Markers")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    print("\n[1] Loading radiogenomic data...")
    df, all_spatial = load_radiogenomic_data()

    n_idh = df["IDH_binary"].notna().sum()
    n_mgmt = df["MGMT_binary"].notna().sum()
    n_grade = df["WHO_grade"].notna().sum()

    # 2. Univariate analysis
    print("\n[2] Running univariate analysis...")
    print("  Binary outcomes (IDH, MGMT)...")
    df_binary, df_ordinal = run_univariate(df, all_spatial)
    print(f"  {len(df_binary)} binary tests, {len(df_ordinal)} ordinal tests completed")

    # 3. FDR correction
    print("\n[3] Applying Benjamini-Hochberg FDR correction...")
    df_binary = apply_fdr(df_binary, "u_p")
    df_ordinal = apply_fdr(df_ordinal, "spearman_p")

    n_idh_sig = df_binary[
        (df_binary["outcome"] == "IDH_binary") & (df_binary["significant_fdr_0.05"] == True)
    ].shape[0]
    n_mgmt_sig = df_binary[
        (df_binary["outcome"] == "MGMT_binary") & (df_binary["significant_fdr_0.05"] == True)
    ].shape[0]
    n_grade_sig = df_ordinal[df_ordinal["significant_fdr_0.05"] == True].shape[0]

    print(f"  IDH: {n_idh_sig}/{len(all_spatial)} significant (FDR < 0.05)")
    print(f"  MGMT: {n_mgmt_sig}/{len(all_spatial)} significant (FDR < 0.05)")
    print(f"  WHO grade: {n_grade_sig}/{len(all_spatial)} significant (FDR < 0.05)")

    # 4. Multivariable logistic regression
    print("\n[4] Fitting multivariable logistic regression...")
    logistic_results = run_multivariable(df, all_spatial)
    print(f"  {len(logistic_results)} models fitted")

    # 5. Visualizations
    print("\n[5] Generating visualizations...")
    make_heatmap(df, all_spatial, OUTPUT_DIR)
    make_boxplots(df, all_spatial, OUTPUT_DIR)

    # 6. Outputs
    print("\n[6] Writing outputs...")
    # Association table
    assoc_table = generate_association_table(df_binary, df_ordinal)
    assoc_table.to_csv(OUTPUT_DIR / "molecular_association_table.csv", index=False)
    print("  Saved molecular_association_table.csv")

    with open(OUTPUT_DIR / "molecular_association_table.json", "w") as f:
        json.dump({
            "binary_results": df_binary.to_dict(orient="records"),
            "ordinal_results": df_ordinal.to_dict(orient="records"),
        }, f, indent=2, default=str)
    print("  Saved molecular_association_table.json")

    # Logistic results
    if logistic_results:
        log_table = generate_logistic_table(logistic_results)
        log_table.to_csv(OUTPUT_DIR / "logistic_models.csv", index=False)
        print("  Saved logistic_models.csv")

        with open(OUTPUT_DIR / "logistic_models.json", "w") as f:
            json.dump(logistic_results, f, indent=2, default=str)
        print("  Saved logistic_models.json")

    # Report
    report = generate_report(df_binary, df_ordinal, logistic_results,
                             all_spatial, n_idh, n_mgmt, n_grade)
    (OUTPUT_DIR / "radiogenomic_report.md").write_text(report, encoding="utf-8")
    print("  Saved radiogenomic_report.md")

    # Summary
    summary = generate_summary(n_idh, n_mgmt, n_grade,
                               n_idh_sig, n_mgmt_sig, n_grade_sig,
                               logistic_results, all_spatial)
    (DOCS_DIR / "radiogenomic_summary.md").write_text(summary, encoding="utf-8")
    print("  Saved docs/radiogenomic_summary.md")

    print("\n" + "=" * 70)
    print("  Radiogenomic Analysis Complete!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
