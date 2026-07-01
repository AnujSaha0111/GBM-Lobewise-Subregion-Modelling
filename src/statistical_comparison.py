#!/usr/bin/env python3

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ──
ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "outputs" / "multimodal_lobewise" / "merged_features_with_metadata.csv"
COMP_DIR = ROOT / "outputs" / "multimodal_lobewise_svm_comparison"
BASELINE_DIR = ROOT / "outputs" / "baselines"
OUTPUT_DIR = ROOT / "outputs" / "statistical_comparisons"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_BOOTSTRAP = 5000
ALPHA = 0.05

SPATIAL_FEATURE_PREFIXES = ("T1_", "T2_", "T1GD_", "FLAIR_")
METADATA_FEATURE_COLS = ["age", "sex", "idh", "mgmt", "who_grade", "eor"]

# SVM best params from the feature-set comparison experiment (optimised per set)
SVM_FS_PARAMS = {
    "spatial": {"C": 100, "gamma": 0.001},
    "clinical": {"C": 1, "gamma": 0.001},
    "combined": {"C": 1, "gamma": "scale"},
}

# Model keys used in the baseline experiment
BASELINE_MODELS = [
    {"key": "lr",  "name": "Logistic Regression", "color": "#1b9e77",
     "estimator": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE, class_weight="balanced")},
    {"key": "rf",  "name": "Random Forest",        "color": "#d95f02",
     "estimator": RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")},
    {"key": "xgb", "name": "XGBoost",              "color": "#7570b3",
     "estimator": XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE)},
    {"key": "svm", "name": "SVM (RBF)",           "color": "#e7298a",
     "estimator": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)},
]

FEATURE_SET_NAMES = {
    "spatial": "Spatial Only",
    "clinical": "Clinical + Molecular",
    "combined": "Combined",
}

FS_KEYS = ["spatial", "clinical", "combined"]

# ── Data ──
def load_and_split():
    df = pd.read_csv(INPUT_CSV)
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    y = df["risk_label"].astype(int)

    spatial_cols = [c for c in df.columns if c.startswith(SPATIAL_FEATURE_PREFIXES)]
    meta_cols = [c for c in METADATA_FEATURE_COLS if c in df.columns]
    combined_cols = spatial_cols + meta_cols

    feature_map = {
        "spatial": spatial_cols,
        "clinical": meta_cols,
        "combined": combined_cols,
    }

    X_train_dict = {}
    X_test_dict = {}
    y_train_dict = {}
    y_test_dict = {}
    y_test_vals = None

    for key in feature_map:
        cols = feature_map[key]
        X = df[cols].apply(pd.to_numeric, errors="coerce")
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
        )
        X_train_dict[key] = X_tr
        X_test_dict[key] = X_te
        y_train_dict[key] = y_tr
        y_test_dict[key] = y_te
        if y_test_vals is None:
            y_test_vals = y_te.values

    return feature_map, X_train_dict, X_test_dict, y_train_dict, y_test_dict, y_test_vals


def load_baseline_best_params():
    """Read best_params for each (model_key, fs_key) from saved JSON files."""
    params = {}
    for mc in BASELINE_MODELS:
        for fs_key in FS_KEYS:
            path = BASELINE_DIR / mc["key"] / f"metrics_{fs_key}.json"
            if path.exists():
                with path.open("r") as f:
                    data = json.load(f)
                bp = data.get("best_params", {})
                params[(mc["key"], fs_key)] = bp
    return params


# ── Paired bootstrap ──
def paired_bootstrap_auc(y_true, prob_a, prob_b, n_bootstrap, random_state):
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    diffs = []
    aucs_a = []
    aucs_b = []
    for bs_idx in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        pa = prob_a[idx]
        pb = prob_b[idx]
        if len(np.unique(yb)) < 2:
            continue
        aa = roc_auc_score(yb, pa)
        ab = roc_auc_score(yb, pb)
        aucs_a.append(aa)
        aucs_b.append(ab)
        diffs.append(ab - aa)
    diffs = np.array(diffs)
    aucs_a = np.array(aucs_a)
    aucs_b = np.array(aucs_b)

    delta = float(np.mean(diffs))
    ci_l = float(np.percentile(diffs, 2.5))
    ci_u = float(np.percentile(diffs, 97.5))

    # Two-sided p-value from bootstrap distribution
    # Under H0: delta = 0, we count how often |diff| >= |observed delta|
    obs_delta = delta  # same as mean of diffs
    p_val = float(np.mean(np.abs(diffs - obs_delta) >= np.abs(obs_delta)))
    # If all diffs are on one side, p = 0 but report as 1/n_bootstrap
    if p_val == 0:
        p_val = 1.0 / n_bootstrap

    return {
        "delta_auc": delta,
        "ci_lower": ci_l,
        "ci_upper": ci_u,
        "p_value": p_val,
    }


# ── FDR correction ──
def benjamini_hochberg(p_values):
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    ranks = np.arange(1, n + 1)
    adjusted = sorted_p * n / ranks
    # Enforce monotonicity
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    # Restore original order
    result = np.empty(n)
    result[sorted_indices] = adjusted
    return list(result)


# ── Main ──
def main():
    print("=" * 60)
    print("  STATISTICAL COMPARISON OF ROC-AUCs")
    print("=" * 60)

    feature_map, X_tr, X_te, y_tr, y_te, y_test_vals = load_and_split()
    print(f"\nTest set: {len(y_test_vals)} samples ({y_test_vals.sum()} high-risk)")

    all_results = []  # dummy

    # ══════════════════════════════════════════════════════════
    #  1. Feature-set comparisons (SVM on different feature sets)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 50}")
    print("  FEATURE-SET COMPARISONS (SVM)")
    print(f"{'=' * 50}")

    svm_probs = {}
    for fs_key in FS_KEYS:
        params = SVM_FS_PARAMS[fs_key]
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
        ])
        pipe.set_params(**{f"svm__{k}": v for k, v in params.items()})
        pipe.fit(X_tr[fs_key], y_tr[fs_key])
        svm_probs[fs_key] = pipe.predict_proba(X_te[fs_key])[:, 1]
        auc_val = roc_auc_score(y_te[fs_key], svm_probs[fs_key])
        print(f"  {FEATURE_SET_NAMES[fs_key]:25s} fitted (AUC={auc_val:.3f})", flush=True)

    fs_pairs = [
        ("spatial", "clinical"),
        ("spatial", "combined"),
        ("clinical", "combined"),
    ]

    fs_results = []
    for fs_a, fs_b in fs_pairs:
        name_a = FEATURE_SET_NAMES[fs_a]
        name_b = FEATURE_SET_NAMES[fs_b]
        # Use clinical test set labels (all same, from same split)
        y_true = y_te[fs_a].values
        prob_a = svm_probs[fs_a]
        prob_b = svm_probs[fs_b]
        print(f"    entering bootstrap: {name_a} vs {name_b}", flush=True)

        comp = paired_bootstrap_auc(y_true, prob_a, prob_b, N_BOOTSTRAP, RANDOM_STATE)
        auc_a = float(roc_auc_score(y_true, prob_a))
        auc_b = float(roc_auc_score(y_true, prob_b))
        print(f"    bootstrap complete: delta={comp['delta_auc']:.4f}", flush=True)
        comp["model_a"] = name_a
        comp["model_b"] = name_b
        comp["comparison"] = f"{name_a} vs {name_b}"
        comp["group"] = "Feature Set"
        comp["auc_a"] = auc_a
        comp["auc_b"] = auc_b
        fs_results.append(comp)
        print(f"  {name_a:30s} vs {name_b:25s}  "
              f"Delta AUC = {comp['delta_auc']:.4f}  "
              f"p = {comp['p_value']:.4f}")

    # FDR correction for feature-set comparisons
    fs_p_raw = [r["p_value"] for r in fs_results]
    fs_p_adj = benjamini_hochberg(fs_p_raw)
    for r, adj_p in zip(fs_results, fs_p_adj):
        r["p_value_adjusted"] = adj_p
        r["significant"] = "Yes" if adj_p < ALPHA else "No"
    all_results.extend(fs_results)

    # ══════════════════════════════════════════════════════════
    #  2. Baseline classifier comparisons
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 50}")
    print("  BASELINE CLASSIFIER COMPARISONS")
    print(f"{'=' * 50}")

    # We use the same test set labels for all models within a feature set
    # (all models share the same train/test split per feature set)
    # Read best_params from saved baseline JSONs
    bp_store = load_baseline_best_params()

    baseline_pairs = [
        ("lr", "rf", "Logistic Regression", "Random Forest"),
        ("lr", "xgb", "Logistic Regression", "XGBoost"),
        ("lr", "svm", "Logistic Regression", "SVM (RBF)"),
        ("rf", "xgb", "Random Forest", "XGBoost"),
        ("rf", "svm", "Random Forest", "SVM (RBF)"),
        ("xgb", "svm", "XGBoost", "SVM (RBF)"),
    ]

    baseline_results = []

    for fs_key in FS_KEYS:
        fs_name = FEATURE_SET_NAMES[fs_key]
        print(f"\n  --- {fs_name} ---")
        y_true = y_te[fs_key].values

        # Fit all 4 models with their best params for this feature set
        model_probs = {}
        for mc in BASELINE_MODELS:
            key = mc["key"]
            name = mc["name"]
            estimator = mc["estimator"]
            bp = bp_store.get((key, fs_key), {})

            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", estimator),
            ])
            if bp:
                pipe.set_params(**bp)
            # Fit on training data
            pipe.fit(X_tr[fs_key], y_tr[fs_key])
            prob = pipe.predict_proba(X_te[fs_key])[:, 1]
            model_probs[key] = prob
            auc_val = float(roc_auc_score(y_true, prob))
            print(f"    {name:25s} AUC = {auc_val:.4f}")

        # All pairwise comparisons
        for ka, kb, na, nb in baseline_pairs:
            prob_a = model_probs[ka]
            prob_b = model_probs[kb]

            comp = paired_bootstrap_auc(y_true, prob_a, prob_b, N_BOOTSTRAP, RANDOM_STATE)
            auc_a = float(roc_auc_score(y_true, prob_a))
            auc_b = float(roc_auc_score(y_true, prob_b))
            comp["model_a"] = f"{na} ({fs_name})"
            comp["model_b"] = f"{nb} ({fs_name})"
            comp["comparison"] = f"{na} vs {nb} [{fs_name}]"
            comp["group"] = f"Baseline [{fs_name}]"
            comp["auc_a"] = auc_a
            comp["auc_b"] = auc_b
            baseline_results.append(comp)

    # FDR correction for all baseline comparisons together
    bl_p_raw = [r["p_value"] for r in baseline_results]
    bl_p_adj = benjamini_hochberg(bl_p_raw)
    for r, adj_p in zip(baseline_results, bl_p_adj):
        r["p_value_adjusted"] = adj_p
        r["significant"] = "Yes" if adj_p < ALPHA else "No"
    all_results.extend(baseline_results)

    # ══════════════════════════════════════════════════════════
    #  3. Outputs
    # ══════════════════════════════════════════════════════════

    # Build dataframe
    df = pd.DataFrame(all_results)
    display_cols = [
        "group", "comparison", "auc_a", "auc_b", "delta_auc",
        "ci_lower", "ci_upper", "p_value", "p_value_adjusted", "significant",
    ]
    df_display = df[display_cols].round(4)

    # Feature-set CSV
    fs_df = pd.DataFrame(fs_results)
    fs_out = fs_df[display_cols].round(4)
    fs_out.to_csv(OUTPUT_DIR / "feature_set_auc_comparisons.csv", index=False)
    print(f"\nSaved feature_set_auc_comparisons.csv")
    print("\nFeature-set comparisons:")
    print(fs_out.to_string(index=False))

    # Baseline CSV
    bl_df = pd.DataFrame(baseline_results)
    bl_out = bl_df[display_cols].round(4)
    bl_out.to_csv(OUTPUT_DIR / "baseline_auc_comparisons.csv", index=False)
    print(f"\nSaved baseline_auc_comparisons.csv")
    print("\nBaseline comparisons (first 6 rows):")
    print(bl_out.head(6).to_string(index=False))

    # ══════════════════════════════════════════════════════════
    #  Forest plot
    # ══════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot all comparisons; colour by significance
    y_positions = list(range(len(all_results)))
    y_labels = []
    colors = []
    x_vals = []
    ci_lowers = []
    ci_uppers = []

    for i, r in enumerate(all_results):
        y_labels.append(r["comparison"])
        colors.append("#d62728" if r["significant"] == "Yes" else "#1f77b4")
        x_vals.append(r["delta_auc"])
        ci_lowers.append(r["ci_lower"])
        ci_uppers.append(r["ci_upper"])

    # Reverse so first comparison is at top
    y_positions = y_positions[::-1]
    y_labels = y_labels[::-1]
    colors = colors[::-1]
    x_vals = x_vals[::-1]
    ci_lowers = ci_lowers[::-1]
    ci_uppers = ci_uppers[::-1]

    ax.errorbar(x_vals, y_positions, fmt="o", color="black", ecolor="black",
                capsize=3, capthick=1, elinewidth=1.2, markersize=6)
    for i, (x, y, c) in enumerate(zip(x_vals, y_positions, colors)):
        ax.plot(x, y, "o", color=c, markersize=8, zorder=5)

    ax.axvline(0, color="gray", linestyle="--", lw=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Delta AUC", fontsize=11)
    ax.set_title("Pairwise AUC Comparisons with 95% CI", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
               markersize=8, label="Significant (FDR-adjusted p < 0.05)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4",
               markersize=8, label="Not significant"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "auc_difference_forest_plot.png", dpi=150)
    plt.close(fig)
    print("Saved auc_difference_forest_plot.png")

    # ══════════════════════════════════════════════════════════
    #  Report
    # ══════════════════════════════════════════════════════════

    # Answers
    def find_comparison(name_a_contains, name_b_contains, group_contains=None):
        for r in all_results:
            if group_contains and group_contains not in r.get("group", ""):
                continue
            if name_a_contains in r["model_a"] and name_b_contains in r["model_b"]:
                return r
            if name_a_contains in r["model_b"] and name_b_contains in r["model_a"]:
                return r
        return None

    def fmt_comp(r):
        if r is None:
            return "N/A"
        sig = "significant" if r["significant"] == "Yes" else "not significant"
        return (f"Delta AUC = {r['delta_auc']:.4f}  "
                f"95% CI = [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]  "
                f"p = {r['p_value']:.4f}  "
                f"FDR-adjusted p = {r['p_value_adjusted']:.4f}  "
                f"({sig})")

    # Find specific comparisons
    spatial_vs_clinical = find_comparison("Spatial Only", "Clinical + Molecular", "Feature Set")
    spatial_vs_combined = find_comparison("Spatial Only", "Combined", "Feature Set")
    clinical_vs_combined = find_comparison("Clinical + Molecular", "Combined", "Feature Set")
    xgb_vs_svm_combined = find_comparison("XGBoost", "SVM", "Baseline [Combined]")

    # Best model per feature set from our computed AUCs
    best_per_fs = {}
    for fs_key in FS_KEYS:
        fs_name = FEATURE_SET_NAMES[fs_key]
        best = None
        best_auc = -1
        for mc in BASELINE_MODELS:
            # Compute AUC for this model on this feature set
            for r in all_results:
                if f"({fs_name})" in r["model_a"] and mc["name"] in r["model_a"]:
                    auc_val = r["auc_a"]
                elif f"({fs_name})" in r["model_b"] and mc["name"] in r["model_b"]:
                    auc_val = r["auc_b"]
                else:
                    continue
                if auc_val > best_auc:
                    best_auc = auc_val
                    best = mc["name"]
        best_per_fs[fs_name] = (best, best_auc)


    # Practical meaningfulness assessment
    meaningful_deltas = [r for r in all_results if abs(r["delta_auc"]) >= 0.02]
    report += f"Of {len(all_results)} comparisons, {len(meaningful_deltas)} show a delta AUC >= 0.02 (a commonly used threshold for practical relevance).\n\n"

    if spatial_vs_clinical and abs(spatial_vs_clinical["delta_auc"]) >= 0.02:
        report += f"- Clinical vs Spatial: {'improvement' if spatial_vs_clinical['delta_auc'] > 0 else 'decrease'} of {abs(spatial_vs_clinical['delta_auc']):.3f} — practically meaningful.\n"
    if spatial_vs_combined and abs(spatial_vs_combined["delta_auc"]) >= 0.02:
        report += f"- Combined vs Spatial: {'improvement' if spatial_vs_combined['delta_auc'] > 0 else 'decrease'} of {abs(spatial_vs_combined['delta_auc']):.3f} — practically meaningful.\n"

    for r in all_results:
        report += (f"| {r['comparison']} | {r['delta_auc']:.4f} | "
                   f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}] | "
                   f"{r['p_value']:.4f} | {r['p_value_adjusted']:.4f} | "
                   f"{r['significant']} |\n")

    print(f"\n{'=' * 60}")
    print(f"  All outputs in: {OUTPUT_DIR}")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
