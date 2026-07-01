#!/usr/bin/env python3

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ──
ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = ROOT / "outputs" / "multimodal_lobewise" / "merged_features_with_metadata.csv"
OUTPUT_DIR = ROOT / "outputs" / "multimodal_lobewise_svm_comparison"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_BOOTSTRAP = 5000
N_PERMUTATION = 5000
ALPHA = 0.05
N_SPLITS = 5
N_REPEATS = 10

PARAM_GRID = {
    "svm__C": [0.01, 0.1, 1, 10, 100],
    "svm__gamma": ["scale", 0.0001, 0.001, 0.01, 0.1],
}

SPATIAL_FEATURE_PREFIXES = ("T1_", "T2_", "T1GD_", "FLAIR_")
METADATA_FEATURE_COLS = ["age", "sex", "idh", "mgmt", "who_grade", "eor"]

FEATURE_SETS = {
    "spatial": {
        "name": "A_spatial_only",
        "output_prefix": "spatial",
    },
    "metadata": {
        "name": "B_clinical_molecular_only",
        "output_prefix": "metadata",
    },
    "combined": {
        "name": "C_combined",
        "output_prefix": "combined",
    },
}

LABEL_NAMES = ["low-risk", "high-risk"]


# ── Helpers ──

def load_and_split():
    df = pd.read_csv(INPUT_CSV)
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    y = df["risk_label"].astype(int)

    spatial_cols = [c for c in df.columns
                    if c.startswith(SPATIAL_FEATURE_PREFIXES)]
    meta_cols = [c for c in METADATA_FEATURE_COLS if c in df.columns]
    combined_cols = spatial_cols + meta_cols

    feature_map = {
        "spatial": spatial_cols,
        "metadata": meta_cols,
        "combined": combined_cols,
    }

    patient_ids_train = {}
    patient_ids_test = {}
    X_train_dict = {}
    X_test_dict = {}
    y_train_dict = {}
    y_test_dict = {}

    for key in feature_map:
        cols = feature_map[key]
        X = df[cols].apply(pd.to_numeric, errors="coerce")

        X_tr, X_te, y_tr, y_te, ids_tr, ids_te = train_test_split(
            X, y, df["patient_id"],
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        X_train_dict[key] = X_tr
        X_test_dict[key] = X_te
        y_train_dict[key] = y_tr
        y_test_dict[key] = y_te
        patient_ids_train[key] = ids_tr
        patient_ids_test[key] = ids_te

    return feature_map, X_train_dict, X_test_dict, y_train_dict, y_test_dict


def train_with_tuning(X_train, y_train):
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True,
                    class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=PARAM_GRID,
        scoring="roc_auc",
        cv=skf,
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_, gs.best_score_, pd.DataFrame(gs.cv_results_)


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred).tolist()
    return {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": cm,
    }, y_proba, y_pred


def bootstrap_auc_ci(y_true, y_prob, n_bootstraps, random_state):
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    boot_aucs = []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        pb = y_prob[idx]
        if len(np.unique(yb)) < 2:
            continue
        boot_aucs.append(roc_auc_score(yb, pb))
    boot_aucs = np.array(boot_aucs)
    return {
        "mean_auc": float(np.mean(boot_aucs)),
        "ci_lower": float(np.percentile(boot_aucs, 2.5)),
        "ci_upper": float(np.percentile(boot_aucs, 97.5)),
        "n_valid_bootstrap": len(boot_aucs),
    }


def permutation_test(y_true, y_prob, observed_auc, n_permutations, random_state):
    rng = np.random.default_rng(random_state)
    perm_aucs = np.empty(n_permutations)
    for i in range(n_permutations):
        y_shuffled = y_true.copy()
        rng.shuffle(y_shuffled)
        perm_aucs[i] = roc_auc_score(y_shuffled, y_prob)
    n_exceed = int(np.sum(perm_aucs >= observed_auc))
    p_value = (n_exceed + 1.0) / (n_permutations + 1.0)
    return {
        "permutation_p_value": float(p_value),
        "n_exceed": n_exceed,
        "n_permutations": n_permutations,
    }


def repeated_stability(X, y):
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True,
                    class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    rskf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE,
    )
    rows = []
    for repeat_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True,
                        class_weight="balanced", random_state=RANDOM_STATE)),
        ])
        pipe.fit(X_tr, y_tr)
        y_pr = pipe.predict(X_te)
        y_prb = pipe.predict_proba(X_te)[:, 1]
        rows.append({
            "repeat": repeat_idx + 1,
            "fold": (repeat_idx % N_SPLITS) + 1,
            "roc_auc": float(roc_auc_score(y_te, y_prb)),
            "accuracy": float(accuracy_score(y_te, y_pr)),
            "precision": float(precision_score(y_te, y_pr, zero_division=0)),
            "recall": float(recall_score(y_te, y_pr, zero_division=0)),
            "f1_score": float(f1_score(y_te, y_pr, zero_division=0)),
        })

    results_df = pd.DataFrame(rows)
    summary = {}
    for col in ["roc_auc", "accuracy", "precision", "recall", "f1_score"]:
        vals = results_df[col].values
        summary[col] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return summary, results_df


def bootstrap_paired_difference(
    y_true, prob_a, prob_c, n_bootstraps, random_state,
):
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    diffs = []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        pa = prob_a[idx]
        pc = prob_c[idx]
        if len(np.unique(yb)) < 2:
            continue
        auc_a = roc_auc_score(yb, pa)
        auc_c = roc_auc_score(yb, pc)
        diffs.append(auc_c - auc_a)
    diffs = np.array(diffs)
    delta = float(np.mean(diffs))
    ci_l = float(np.percentile(diffs, 2.5))
    ci_u = float(np.percentile(diffs, 97.5))
    if ci_l > 0:
        p_value = float(np.mean(diffs <= 0))
    elif ci_u < 0:
        p_value = float(np.mean(diffs >= 0))
    else:
        p_value = 2.0 * min(float(np.mean(diffs >= 0)), float(np.mean(diffs <= 0)))
    return {
        "delta_auc": delta,
        "ci_lower": ci_l,
        "ci_upper": ci_u,
        "p_value": p_value,
        "n_valid_bootstrap": len(diffs),
    }


# ── Main ──

def main():
    print("=" * 60)
    print("  SVM FEATURE SET COMPARISON")
    print("=" * 60)
    print()

    feature_map, X_tr, X_te, y_tr, y_te = load_and_split()

    all_metrics = {}
    all_probas = {}
    all_preds = {}
    all_best_params = {}
    all_auc_sig = {}
    all_cv_summary = {}
    all_cv_results = {}
    all_best_cv_auc = {}

    roc_curves = {}

    for key in ["spatial", "metadata", "combined"]:
        prefix = FEATURE_SETS[key]["output_prefix"]
        print(f"\n{'-' * 50}")
        print(f"  Feature Set: {FEATURE_SETS[key]['name']} ({key})")
        print(f"  Features: {X_tr[key].shape[1]}")
        print(f"{'-' * 50}")

        # Train with tuning
        print("  GridSearchCV ...")
        model, best_params, best_cv_auc, cv_results_df = train_with_tuning(
            X_tr[key], y_tr[key],
        )
        all_best_params[key] = best_params
        all_best_cv_auc[key] = best_cv_auc
        all_cv_results[key] = cv_results_df
        print(f"    Best params: {best_params}")
        print(f"    Best CV AUC: {best_cv_auc:.4f}")

        # Evaluate
        metrics, y_proba, y_pred = evaluate(model, X_te[key], y_te[key])
        all_metrics[key] = metrics
        all_probas[key] = y_proba
        all_preds[key] = y_pred
        print(f"    Test ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"    Test Accuracy: {metrics['accuracy']:.4f}")

        # Bootstrap CI
        print("  Bootstrap CI ...")
        boot_result = bootstrap_auc_ci(
            y_te[key].values, y_proba, N_BOOTSTRAP, RANDOM_STATE,
        )

        # Permutation test
        print("  Permutation test ...")
        perm_result = permutation_test(
            y_te[key].values, y_proba, metrics["roc_auc"], N_PERMUTATION, RANDOM_STATE + 1,
        )

        auc_sig = {
            "observed_auc": metrics["roc_auc"],
            **boot_result,
            **perm_result,
            "n_test_samples": len(y_te[key]),
            "alpha": ALPHA,
        }
        all_auc_sig[key] = auc_sig

        # ROC curve data
        fpr, tpr, _ = roc_curve(y_te[key].values, y_proba)
        roc_curves[key] = {"fpr": fpr, "tpr": tpr, "auc": metrics["roc_auc"]}

        # Repeated stability
        print("  RepeatedStratifiedKFold stability ...")
        cv_summary, cv_results_df = repeated_stability(
            pd.concat([X_tr[key], X_te[key]]),
            pd.concat([y_tr[key], y_te[key]]),
        )
        all_cv_summary[key] = cv_summary
        all_cv_results[key + "_stability"] = cv_results_df
        auc_cv = cv_summary["roc_auc"]
        print(f"    CV ROC-AUC: {auc_cv['mean']:.4f} ± {auc_cv['std']:.4f}")

    # ── Statistical comparison: Spatial vs Combined ──
    print(f"\n{'-' * 50}")
    print("  COMPARISON: Spatial vs Combined")
    print(f"{'-' * 50}")
    comparison = bootstrap_paired_difference(
        y_te["combined"].values,
        all_probas["spatial"],
        all_probas["combined"],
        N_BOOTSTRAP,
        RANDOM_STATE + 2,
    )
    print(f"    Delta AUC (Combined - Spatial): {comparison['delta_auc']:.4f}")
    print(f"    95% CI: [{comparison['ci_lower']:.4f}, {comparison['ci_upper']:.4f}]")
    print(f"    p-value: {comparison['p_value']:.4f}")

    # ── Save all outputs ──
    for key in ["spatial", "metadata", "combined"]:
        prefix = FEATURE_SETS[key]["output_prefix"]

        # Metrics
        path = OUTPUT_DIR / f"metrics_{prefix}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(all_metrics[key], f, indent=2)

        # AUC significance
        path = OUTPUT_DIR / f"auc_significance_{prefix}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(all_auc_sig[key], f, indent=2)

        # CV summary
        path = OUTPUT_DIR / f"cv_{prefix}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(all_cv_summary[key], f, indent=2)

    # Model comparison CSV
    rows = []
    for key in ["spatial", "metadata", "combined"]:
        m = all_metrics[key]
        auc_sig = all_auc_sig[key]
        n_feat = X_tr[key].shape[1]
        rows.append({
            "Feature Set": FEATURE_SETS[key]["name"],
            "No. Features": n_feat,
            "ROC-AUC": f"{m['roc_auc']:.4f}",
            "95% CI": f"[{auc_sig['ci_lower']:.4f}, {auc_sig['ci_upper']:.4f}]",
            "Accuracy": f"{m['accuracy']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall": f"{m['recall']:.4f}",
            "F1": f"{m['f1_score']:.4f}",
            "Permutation p-value": f"{auc_sig['permutation_p_value']:.4f}",
        })
    comparison_df = pd.DataFrame(rows)
    comp_path = OUTPUT_DIR / "model_comparison.csv"
    comparison_df.to_csv(comp_path, index=False)
    print(f"\nWrote {comp_path}")
    print(comparison_df.to_string(index=False))

    # ── Figures ──

    # 1. Comparison ROC curves
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = {"spatial": "blue", "metadata": "green", "combined": "red"}
    labels = {
        "spatial": f"Spatial (AUC = {roc_curves['spatial']['auc']:.3f})",
        "metadata": f"Clinical (AUC = {roc_curves['metadata']['auc']:.3f})",
        "combined": f"Combined (AUC = {roc_curves['combined']['auc']:.3f})",
    }
    for key in ["spatial", "metadata", "combined"]:
        ax.plot(
            roc_curves[key]["fpr"], roc_curves[key]["tpr"],
            color=colors[key], lw=2, label=labels[key],
        )
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Chance")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — SVM (RBF) by Feature Set", fontsize=13)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "comparison_roc_curves.png", dpi=150)
    plt.close(fig)
    print("Saved comparison_roc_curves.png")

    # 2. AUC comparison bar plot with error bars
    auc_vals = []
    ci_lowers = []
    ci_uppers = []
    set_labels_plot = ["Spatial", "Clinical", "Combined"]
    for key in ["spatial", "metadata", "combined"]:
        auc_vals.append(all_metrics[key]["roc_auc"])
        ci_lowers.append(all_auc_sig[key]["ci_lower"])
        ci_uppers.append(all_auc_sig[key]["ci_upper"])
    auc_vals = np.array(auc_vals)
    ci_lowers = np.array(ci_lowers)
    ci_uppers = np.array(ci_uppers)
    yerr_lower = auc_vals - ci_lowers
    yerr_upper = ci_uppers - auc_vals

    fig, ax = plt.subplots(figsize=(8, 6))
    x_pos = np.arange(len(set_labels_plot))
    bars = ax.bar(x_pos, auc_vals, width=0.5, color=["steelblue", "seagreen", "coral"],
                  edgecolor="black", linewidth=1)
    ax.errorbar(
        x_pos, auc_vals,
        yerr=[yerr_lower, yerr_upper],
        fmt="none", ecolor="black", capsize=5, capthick=1.5, elinewidth=1.5,
    )
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance (0.5)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(set_labels_plot, fontsize=12)
    ax.set_ylabel("ROC-AUC", fontsize=12)
    ax.set_title("ROC-AUC by Feature Set with 95% CI", fontsize=13)
    ax.set_ylim([0.0, 1.0])
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    for i, (x, v) in enumerate(zip(x_pos, auc_vals)):
        ax.text(x, v + 0.015, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "auc_comparison_barplot.png", dpi=150)
    plt.close(fig)
    print("Saved auc_comparison_barplot.png")

    # 3. Delta AUC plot
    delta = comparison["delta_auc"]
    d_ci_l = comparison["ci_lower"]
    d_ci_u = comparison["ci_upper"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(0, delta, width=0.4, color="mediumpurple", edgecolor="black",
           linewidth=1.2)
    ax.errorbar(0, delta, yerr=[[delta - d_ci_l], [d_ci_u - delta]],
                fmt="none", ecolor="black", capsize=5, capthick=1.5, elinewidth=1.5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks([0])
    ax.set_xticklabels(["Combined − Spatial"], fontsize=12)
    ax.set_ylabel("Delta AUC", fontsize=12)
    ax.set_title(
        f"Incremental Value of Clinical + Molecular Features\n"
        f"Delta AUC = {delta:.4f}  [{d_ci_l:.4f}, {d_ci_u:.4f}]  "
        f"p = {comparison['p_value']:.4f}",
        fontsize=12,
    )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "delta_auc_plot.png", dpi=150)
    plt.close(fig)
    print("Saved delta_auc_plot.png")

    # ── Write conclusions.md ──
    spatial_auc = all_metrics["spatial"]["roc_auc"]
    meta_auc = all_metrics["metadata"]["roc_auc"]
    combined_auc = all_metrics["combined"]["roc_auc"]
    delta_auc = comparison["delta_auc"]
    p_val = comparison["p_value"]
    sig_str = "statistically significant" if p_val < ALPHA else "not statistically significant"

    print(f"\n{'=' * 60}")
    print("  COMPARISON COMPLETE")
    print(f"  All outputs in: {OUTPUT_DIR}")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
