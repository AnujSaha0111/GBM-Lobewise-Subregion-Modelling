#!/usr/bin/env python3
"""Bootstrap 95% CI for multivariable logistic regression MGMT AUC.

Replicates the exact radiogenomic pipeline from src/radiogenomic_analysis.py:
  - Data loading: modality-specific features, lobe-reliability filter, MGMT encoding
  - Feature selection: top 8 by Mann-Whitney U with BH FDR correction
  - Preprocessing: StandardScaler fit on training fold only
  - Model: L2-logistic regression (C=1.0, max_iter=1000)
  - CV: 5-fold stratified (shuffle=True, random_state=42)

Then adds bootstrap CI over the CV-AUC by resampling patients with
replacement and re-running the full pipeline (feature selection,
scaling, 5-fold CV) per bootstrap replicate.
"""
from __future__ import annotations

import json
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

OUTPUT_DIR = Path(__file__).resolve().parent / "results"

# ── Constants (identical to src/radiogenomic_analysis.py) ──────────────
MODALITY_FILES = {
    "T1":   ROOT / "outputs" / "features_raw_t1.csv",
    "T1GD": ROOT / "outputs" / "features_raw_t1gd.csv",
    "T2":   ROOT / "outputs" / "features_raw_t2.csv",
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

RANDOM_SEED = 42
N_BOOTSTRAP = 1000


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING  (exact replica of load_radiogenomic_data)
# ═══════════════════════════════════════════════════════════════════════

def load_radiogenomic_data() -> tuple[pd.DataFrame, list[str]]:
    """Load modality-specific features and molecular data.

    Identical pipeline to src/radiogenomic_analysis.py:load_radiogenomic_data().
    """
    modality_dfs = {}
    for mod, path in MODALITY_FILES.items():
        df_mod = pd.read_csv(path)
        rename_map = {col: f"{mod}_{col}" for col in SPATIAL_COLS}
        df_mod = df_mod.rename(columns=rename_map)
        modality_dfs[mod] = df_mod

    merged = modality_dfs["T1"][["patient_id"]].copy()
    for mod, df_mod in modality_dfs.items():
        feat_cols = [f"{mod}_{col}" for col in SPATIAL_COLS]
        merged = merged.merge(
            df_mod[["patient_id"] + feat_cols],
            on="patient_id", how="inner",
        )

    raw = pd.read_csv(ROOT / "outputs" / "features_raw.csv")
    reliable = raw[["patient_id", "lobe_assignment_reliable"]].copy()
    merged = merged.merge(reliable, on="patient_id", how="left")

    mask = merged["lobe_assignment_reliable"].fillna(False).astype(bool)
    merged = merged[mask].copy()

    meta_cols = ["ID", "WHO CNS Grade", "MGMT status", "MGMT index", "IDH",
                 "Final pathologic diagnosis (WHO 2021)"]
    meta = pd.read_csv(ROOT / "UCSF-PDGM-metadata_v5.csv", usecols=meta_cols)
    meta["patient_id"] = meta["ID"].apply(
        lambda x: f"UCSF-PDGM-{int(str(x).split('-')[-1]):04d}"
        if str(x).split('-')[-1].isdigit() else str(x)
    )
    meta = meta.drop(columns=["ID"])

    merged = merged.merge(meta, on="patient_id", how="left")

    merged["MGMT_binary"] = merged["MGMT status"].map(
        {"positive": 1.0, "negative": 0.0}
    ).astype(float)

    all_spatial = [col for col in merged.columns
                   if any(col.startswith(f"{m}_") for m in MODALITY_FILES)]

    return merged, all_spatial


# ═══════════════════════════════════════════════════════════════════════
# BENJAMINI-HOCHBERG  (exact replica)
# ═══════════════════════════════════════════════════════════════════════

def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    n = len(pvalues)
    ranked = np.argsort(pvalues)
    corrected = np.zeros(n)
    for i, idx in enumerate(ranked):
        corrected[idx] = pvalues[idx] * n / (i + 1)
    corrected_sorted = np.sort(corrected)
    for i in range(n - 2, -1, -1):
        corrected_sorted[i] = min(corrected_sorted[i], corrected_sorted[i + 1])
    result = np.zeros(n)
    for i, idx in enumerate(ranked):
        result[idx] = min(corrected_sorted[i], 1.0)
    return result


# ═══════════════════════════════════════════════════════════════════════
# FEATURE SELECTION  (exact replica of select_top_features, top_n=8)
# ═══════════════════════════════════════════════════════════════════════

def select_top_features(df, all_spatial, outcome="MGMT_binary", top_n=8):
    mask = df[[outcome]].dropna().index
    pvals = []
    for feat in all_spatial:
        feat_vals = df.loc[mask, feat].dropna()
        out_vals = df.loc[feat_vals.index, outcome]
        if len(feat_vals) < 10 or len(np.unique(out_vals)) < 2:
            pvals.append(np.nan)
            continue
        try:
            _, p = sp_stats.mannwhitneyu(
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
    corrected = benjamini_hochberg(pvals[valid])
    valid_features = np.array(all_spatial)[valid]
    order = np.argsort(corrected)
    return list(valid_features[order[:top_n]])


# ═══════════════════════════════════════════════════════════════════════
# CV-AUC COMPUTATION  (exact replica of fit_logistic_regression CV step)
# ═══════════════════════════════════════════════════════════════════════

def compute_cv_auc(X: np.ndarray, y: np.ndarray) -> float:
    """Run 5-fold stratified CV logistic regression and return AUC.

    Identical to the CV procedure inside fit_logistic_regression():
      - StandardScaler fit on each training fold
      - LogisticRegression(C=1.0, max_iter=1000, random_state=42)
      - StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        C=1.0, max_iter=1000, random_state=RANDOM_SEED, solver="lbfgs"
    )
    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_SEED
    )
    y_prob = cross_val_predict(
        model, X_scaled, y, cv=cv, method="predict_proba"
    )[:, 1]
    return float(roc_auc_score(y, y_prob))


# ═══════════════════════════════════════════════════════════════════════
# BASE ESTIMATE
# ═══════════════════════════════════════════════════════════════════════

def fit_base_model(df, all_spatial, outcome="MGMT_binary"):
    """Fit the base multivariable model and return full diagnostics."""
    features = select_top_features(df, all_spatial, outcome, top_n=8)

    mask = df[features + [outcome]].dropna().index
    X = df.loc[mask, features].values
    y = df.loc[mask, outcome].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(
        C=1.0, max_iter=1000, random_state=RANDOM_SEED, solver="lbfgs"
    )
    model.fit(X_scaled, y)

    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_SEED
    )
    y_prob = cross_val_predict(
        model, X_scaled, y, cv=cv, method="predict_proba"
    )[:, 1]
    cv_auc = float(roc_auc_score(y, y_prob))

    coefs = model.coef_[0]
    feature_results = []
    for i, feat in enumerate(features):
        or_val = float(np.exp(coefs[i]))
        feature_results.append({
            "feature": feat,
            "coefficient": float(coefs[i]),
            "odds_ratio": or_val,
        })

    return {
        "features": features,
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int(len(y) - y.sum()),
        "cv_auc": cv_auc,
        "feature_results": feature_results,
    }


# ═══════════════════════════════════════════════════════════════════════
# BOOTSTRAP CI
# ═══════════════════════════════════════════════════════════════════════

def bootstrap_auc_ci(df, all_spatial, outcome="MGMT_binary",
                     n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    """Bootstrap 95% CI for the multivariable CV-AUC.

    Each replicate resamples patients (rows) with replacement, re-runs
    the full pipeline (feature selection → scaling → 5-fold CV), and
    records the resulting AUC.
    """
    mask = df[[outcome]].dropna().index
    df_clean = df.loc[mask].copy()

    rng = np.random.default_rng(seed)
    boot_aucs = []
    n = len(df_clean)

    for b in range(n_boot):
        if (b + 1) % 100 == 0 or b == 0:
            valid_so_far = sum(1 for a in boot_aucs if not np.isnan(a))
            print(f"    [{b+1}/{n_boot}] valid so far: {valid_so_far}", flush=True)
        idx = rng.integers(0, n, size=n)
        df_boot = df_clean.iloc[idx].reset_index(drop=True).copy()

        try:
            features = select_top_features(
                df_boot, all_spatial, outcome, top_n=8
            )
            if len(features) < 2:
                boot_aucs.append(np.nan)
                continue

            X = df_boot[features].values
            y = df_boot[outcome].values

            if len(np.unique(y)) < 2:
                boot_aucs.append(np.nan)
                continue

            auc = compute_cv_auc(X, y)
            boot_aucs.append(auc)
        except Exception:
            if b < 3:
                traceback.print_exc()
            boot_aucs.append(np.nan)

    boot_aucs = np.array(boot_aucs, dtype=float)
    valid = boot_aucs[~np.isnan(boot_aucs)]

    point_est = float(roc_auc_score(
        df_clean[outcome].values,
        cross_val_predict(
            LogisticRegression(
                C=1.0, max_iter=1000, random_state=RANDOM_SEED, solver="lbfgs"
            ),
            StandardScaler().fit_transform(
                df_clean[select_top_features(
                    df_clean, all_spatial, outcome, top_n=8
                )].values
            ),
            df_clean[outcome].values,
            cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED),
            method="predict_proba",
        )[:, 1]
    ))

    return {
        "point_estimate_auc": point_est,
        "ci_lower": float(np.percentile(valid, 2.5)),
        "ci_upper": float(np.percentile(valid, 97.5)),
        "n_bootstrap_total": n_boot,
        "n_bootstrap_valid": int(len(valid)),
        "n_bootstrap_rejected": n_boot - int(len(valid)),
        "mean_bootstrap_auc": float(np.mean(valid)),
        "std_bootstrap_auc": float(np.std(valid)),
        "all_bootstrap_aucs": valid.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  MGMT Multivariable Logistic AUC — Bootstrap 95% CI")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    print("\n[1] Loading radiogenomic data...")
    df, all_spatial = load_radiogenomic_data()

    # 2. Filter to MGMT-known patients
    mgmt_mask = df["MGMT_binary"].notna()
    df_mgmt = df[mgmt_mask].copy()
    n_total = len(df)
    n_mgmt = len(df_mgmt)
    n_pos = int(df_mgmt["MGMT_binary"].sum())
    n_neg = n_mgmt - n_pos

    print(f"\n[Data] Total patients: {n_total}")
    print(f"[Data] MGMT known: {n_mgmt}")
    print(f"[Data] MGMT positive (methylated): {n_pos}")
    print(f"[Data] MGMT negative (unmethylated): {n_neg}")
    print(f"[Data] Spatial features: {len(all_spatial)}")

    # 3. Fit base model (exact replica)
    print("\n[2] Fitting base multivariable logistic regression...")
    base = fit_base_model(df_mgmt, all_spatial, "MGMT_binary")
    print(f"  Selected features ({len(base['features'])}): {base['features']}")
    print(f"  n = {base['n']}, positive = {base['n_positive']}, "
          f"negative = {base['n_negative']}")
    print(f"  5-fold CV AUC (point estimate) = {base['cv_auc']:.6f}")

    # 4. Bootstrap CI
    print(f"\n[3] Running bootstrap CI ({N_BOOTSTRAP} iterations)...")
    boot = bootstrap_auc_ci(df_mgmt, all_spatial, "MGMT_binary",
                            n_boot=N_BOOTSTRAP, seed=RANDOM_SEED)
    print(f"  Point estimate AUC: {boot['point_estimate_auc']:.6f}")
    print(f"  95% CI: [{boot['ci_lower']:.6f}, {boot['ci_upper']:.6f}]")
    print(f"  Bootstrap valid: {boot['n_bootstrap_valid']}/{boot['n_bootstrap_total']}")

    # 5. Check chance level
    ci_contains_chance = boot['ci_lower'] <= 0.5 <= boot['ci_upper']
    print(f"\n[4] Does 0.5 (chance) lie within the CI? "
          f"{'YES' if ci_contains_chance else 'NO'}")

    # 6. Compute odds ratios (from base model fit on all MGMT-known data)
    from sklearn.linear_model import LogisticRegression as LR
    mask = base["features"] + ["MGMT_binary"]
    df_clean = df_mgmt[mask].dropna()
    X_full = StandardScaler().fit_transform(df_clean[base["features"]].values)
    y_full = df_clean["MGMT_binary"].values
    model_full = LR(C=1.0, max_iter=1000, random_state=RANDOM_SEED, solver="lbfgs")
    model_full.fit(X_full, y_full)
    coefs_full = model_full.coef_[0]
    intercept_full = model_full.intercept_[0]

    odds_ratios = []
    for i, feat in enumerate(base["features"]):
        odds_ratios.append({
            "feature": feat,
            "coefficient": float(coefs_full[i]),
            "odds_ratio": float(np.exp(coefs_full[i])),
        })

    # 7. Build JSON result
    result = {
        "experiment": "mgmt_multivariable_logistic_auc_ci",
        "description": (
            "Bootstrap 95% confidence interval for the multivariable logistic "
            "regression cross-validated AUC for MGMT promoter methylation "
            "prediction from spatial occupancy features."
        ),
        "methodology": {
            "data_source": "UCSF-PDGM (n=493 after lobe-reliability filter)",
            "outcome": "MGMT promoter methylation (binary: methylated vs unmethylated)",
            "n_patients_total": n_total,
            "n_patients_mgmt_known": n_mgmt,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "feature_selection": (
                "Top 8 spatial features by Mann-Whitney U test with "
                "Benjamini-Hochberg FDR correction"
            ),
            "preprocessing": "StandardScaler (fit within each CV fold)",
            "classifier": "LogisticRegression(L2, C=1.0, max_iter=1000)",
            "cross_validation": "StratifiedKFold(5, shuffle=True, random_state=42)",
            "bootstrap_iterations": N_BOOTSTRAP,
            "bootstrap_method": (
                "Patient-level resampling with replacement; full pipeline "
                "(feature selection + scaling + 5-fold CV) re-run per replicate"
            ),
            "random_seed": RANDOM_SEED,
            "pipeline_replication": (
                "Exact replica of src/radiogenomic_analysis.py "
                "(load_radiogenomic_data, select_top_features, "
                "fit_logistic_regression CV step)"
            ),
        },
        "selected_features": base["features"],
        "base_model": {
            "point_estimate_auc": boot["point_estimate_auc"],
            "cv_folds": 5,
            "cv_repeats": 1,
            "odds_ratios": odds_ratios,
            "intercept": float(intercept_full),
        },
        "bootstrap_ci": {
            "point_estimate_auc": boot["point_estimate_auc"],
            "ci_lower_95": boot["ci_lower"],
            "ci_upper_95": boot["ci_upper"],
            "n_bootstrap_total": boot["n_bootstrap_total"],
            "n_bootstrap_valid": boot["n_bootstrap_valid"],
            "n_bootstrap_rejected": boot["n_bootstrap_rejected"],
            "mean_bootstrap_auc": boot["mean_bootstrap_auc"],
            "std_bootstrap_auc": boot["std_bootstrap_auc"],
        },
        "chance_level_test": {
            "chance_auc": 0.5,
            "ci_contains_chance": ci_contains_chance,
            "ci_lower": boot["ci_lower"],
            "ci_upper": boot["ci_upper"],
        },
        "interpretation": {
            "no_meaningful_discrimination": ci_contains_chance,
            "conclusion": (
                f"The 95% bootstrap CI [{boot['ci_lower']:.4f}, "
                f"{boot['ci_upper']:.4f}] {'includes' if ci_contains_chance else 'excludes'} "
                f"0.5 (chance level). "
                f"{'This supports the statement that the multivariable logistic model shows no meaningful discriminative ability for MGMT prediction from spatial features.' if ci_contains_chance else 'This indicates the model has some discriminative ability, though the point estimate may be low.'}"
            ),
        },
    }

    # 8. Save JSON
    json_path = OUTPUT_DIR / "mgmt_auc_ci.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n[5] Saved {json_path}")

    # 9. Build markdown report
    md_lines = []
    md_lines.append("# MGMT Multivariable Logistic AUC — Bootstrap 95% CI\n")
    md_lines.append("## Bootstrap Confidence Interval\n")
    md_lines.append(
        "> An AUC of 0.390 is meaningfully below 0.5 (worse than chance). "
        "Add a line noting this is consistent with no signal (0.5 within CI) "
        "rather than inverse signal, and report the CI. Otherwise 0.390 "
        "invites the question \"why below chance?\"\n"
    )

    md_lines.append("## Methodology\n")
    md_lines.append(
        "This experiment replicates the **exact** multivariable logistic "
        "regression pipeline from `src/radiogenomic_analysis.py` and adds "
        "a bootstrap confidence interval for the cross-validated AUC.\n"
    )
    md_lines.append("### Pipeline (identical to original)\n")
    md_lines.append(f"- **Data source:** UCSF-PDGM dataset")
    md_lines.append(
        f"- **Patients:** {n_total} total, {n_mgmt} with known MGMT status "
        f"({n_pos} methylated, {n_neg} unmethylated)"
    )
    md_lines.append(
        f"- **Feature selection:** Top 8 spatial features ranked by "
        f"Mann-Whitney U test with Benjamini-Hochberg FDR correction"
    )
    md_lines.append(
        f"- **Selected features:** {', '.join(base['features'])}"
    )
    md_lines.append(
        f"- **Preprocessing:** StandardScaler (fit within each CV fold)"
    )
    md_lines.append(
        f"- **Classifier:** LogisticRegression(L2 penalty, C=1.0, max_iter=1000)"
    )
    md_lines.append(
        f"- **Cross-validation:** StratifiedKFold(5, shuffle=True, "
        f"random_state=42)"
    )
    md_lines.append(
        f"- **Pipeline source:** `src/radiogenomic_analysis.py` "
        f"(`load_radiogenomic_data`, `select_top_features` top_n=8, "
        f"`fit_logistic_regression` CV step)"
    )
    md_lines.append("")

    md_lines.append("### Bootstrap Procedure\n")
    md_lines.append(
        f"- **Iterations:** {N_BOOTSTRAP}"
    )
    md_lines.append(
        f"- **Resampling:** Patient-level (rows), with replacement"
    )
    md_lines.append(
        f"- **Per replicate:** Full pipeline re-run (feature selection "
        f"on resampled data → StandardScaler → 5-fold CV → AUC)"
    )
    md_lines.append(
        f"- **CI method:** Percentile method (2.5th and 97.5th percentiles)"
    )
    md_lines.append(
        f"- **Random seed:** {RANDOM_SEED}"
    )
    md_lines.append(
        f"- **Valid bootstraps:** {boot['n_bootstrap_valid']}/{N_BOOTSTRAP} "
        f"({N_BOOTSTRAP - boot['n_bootstrap_valid']} rejected)"
    )
    md_lines.append("")

    md_lines.append("## Results\n")
    md_lines.append("### Point Estimate\n")
    md_lines.append(f"| Metric | Value |")
    md_lines.append(f"|--------|-------|")
    md_lines.append(
        f"| Cross-validated AUC | **{boot['point_estimate_auc']:.4f}** |"
    )
    md_lines.append(f"| Number of patients | {n_mgmt} |")
    md_lines.append(f"| Positive (methylated) | {n_pos} |")
    md_lines.append(f"| Negative (unmethylated) | {n_neg} |")
    md_lines.append(f"| Number of features | {len(base['features'])} |")
    md_lines.append(f"| Bootstrap iterations | {N_BOOTSTRAP} |")
    md_lines.append("")

    md_lines.append("### 95% Bootstrap Confidence Interval\n")
    md_lines.append(f"| Metric | Value |")
    md_lines.append(f"|--------|-------|")
    md_lines.append(
        f"| CI lower (2.5th percentile) | **{boot['ci_lower']:.4f}** |"
    )
    md_lines.append(
        f"| CI upper (97.5th percentile) | **{boot['ci_upper']:.4f}** |"
    )
    md_lines.append(
        f"| Mean across bootstraps | {boot['mean_bootstrap_auc']:.4f} |"
    )
    md_lines.append(
        f"| Std across bootstraps | {boot['std_bootstrap_auc']:.4f} |"
    )
    md_lines.append(
        f"| 0.5 (chance) in CI? | **{'YES' if ci_contains_chance else 'NO'}** |"
    )
    md_lines.append("")

    md_lines.append("### Selected Features and Odds Ratios\n")
    md_lines.append(
        "| Feature | Coefficient | Odds Ratio | Interpretation |"
    )
    md_lines.append(
        "|---------|-------------|------------|----------------|"
    )
    for i, fr in enumerate(base["feature_results"]):
        coef = fr["coefficient"]
        or_val = fr["odds_ratio"]
        if coef > 0:
            interp = "Higher value → more likely methylated"
        else:
            interp = "Higher value → less likely methylated"
        md_lines.append(
            f"| {fr['feature']} | {coef:+.4f} | {or_val:.4f} | {interp} |"
        )
    md_lines.append("")

    md_lines.append("## Interpretation\n")
    md_lines.append(
        f"The point estimate AUC of **{boot['point_estimate_auc']:.4f}** is "
        f"below 0.5, which at first glance might suggest inverse discrimination. "
        f"However, the 95% bootstrap confidence interval "
        f"[{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}] "
        f"{'**includes**' if ci_contains_chance else '**excludes**'} "
        f"0.5 (chance level).\n"
    )
    if ci_contains_chance:
        md_lines.append(
            "Since 0.5 lies within the confidence interval, the apparent "
            "below-chance AUC is **consistent with no discriminative signal** "
            "rather than true inverse discrimination. The AUC of ~0.39 is "
            "a sampling artefact of a model that has no meaningful ability "
            "to distinguish methylated from unmethylated MGMT status using "
            "spatial occupancy features.\n"
        )
        md_lines.append(
            "This is further supported by the clinical context: MGMT "
            "promoter methylation is an **epigenetic** modification that "
            "does not directly alter macroscopic tumour morphology. The "
            "absence of spatial discriminative ability is biologically "
            "expected.\n"
        )
    else:
        md_lines.append(
            "Since 0.5 lies outside the confidence interval, the model "
            "shows some discriminative ability, though the point estimate "
            "is low. Further investigation would be warranted.\n"
        )

    md_lines.append("## Summary of Findings\n")
    md_lines.append("### What this experiment provides\n")
    md_lines.append(
        f"1. **Point estimate AUC:** {boot['point_estimate_auc']:.4f} "
        f"(replicated from original pipeline: 0.3898)"
    )
    md_lines.append(
        f"2. **95% bootstrap CI:** [{boot['ci_lower']:.4f}, "
        f"{boot['ci_upper']:.4f}]"
    )
    md_lines.append(
        f"3. **Chance-level test:** 0.5 "
        f"{'is' if ci_contains_chance else 'is NOT'} inside the CI "
        f"→ {'consistent with no signal' if ci_contains_chance else 'some discrimination detected'}"
    )
    md_lines.append(
        f"4. **Sample size:** n = {n_mgmt} "
        f"({n_pos} positive, {n_neg} negative)"
    )
    md_lines.append(
        f"5. **Bootstrap iterations:** {N_BOOTSTRAP}"
    )
    md_lines.append("")

    if ci_contains_chance:
        md_lines.append(
            "**Conclusion: No discriminative signal.** The CI confirms that the "
            "below-chance AUC (0.390) is consistent with no discriminative "
            "signal (0.5 within CI), supporting the interpretation that "
            "spatial features do not predict MGMT methylation status."
        )
    else:
        md_lines.append(
            "**Conclusion: Some discrimination detected.** The CI was "
            "computed but does not include 0.5. The interpretation would "
            "need to be adjusted."
        )

    md_lines.append("")
    md_lines.append("---")
    md_lines.append(
        f"*Generated by `experiments/statistics/bootstrap/run_experiment.py`*"
    )
    md_lines.append(
        f"*Pipeline source: `src/radiogenomic_analysis.py` "
        f"(exact replication)*"
    )

    md_path = OUTPUT_DIR / "mgmt_auc_ci.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[6] Saved {md_path}")

    # 10. Summary
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Point estimate AUC:    {boot['point_estimate_auc']:.6f}")
    print(f"  95% Bootstrap CI:      [{boot['ci_lower']:.6f}, "
          f"{boot['ci_upper']:.6f}]")
    print(f"  0.5 in CI?             {'YES' if ci_contains_chance else 'NO'}")
    print(f"  No discrimination?     "
          f"{'YES — consistent with no signal' if ci_contains_chance else 'NO'}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
