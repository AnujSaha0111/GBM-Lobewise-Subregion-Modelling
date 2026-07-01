#!/usr/bin/env python3

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ──
ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = ROOT / "outputs" / "multimodal_lobewise" / "merged_features_with_metadata.csv"
OUTPUT_DIR = ROOT / "outputs" / "multimodal_lobewise_svm_comparison"
CAL_DIR = OUTPUT_DIR / "calibration"
CAL_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_BINS = 10

SPATIAL_FEATURE_PREFIXES = ("T1_", "T2_", "T1GD_", "FLAIR_")
METADATA_FEATURE_COLS = ["age", "sex", "idh", "mgmt", "who_grade", "eor"]

FEATURE_SETS = {
    "spatial": {
        "name": "Spatial Only",
        "prefix": "spatial",
        "best_params": {"svm__C": 100, "svm__gamma": 0.001},
    },
    "clinical": {
        "name": "Clinical Only",
        "prefix": "clinical",
        "best_params": {"svm__C": 1, "svm__gamma": 0.001},
    },
    "combined": {
        "name": "Combined",
        "prefix": "combined",
        "best_params": {"svm__C": 1, "svm__gamma": "scale"},
    },
}


# ── Data ──

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
        "clinical": meta_cols,
        "combined": combined_cols,
    }

    X_train_dict = {}
    X_test_dict = {}
    y_train_dict = {}
    y_test_dict = {}

    for key in feature_map:
        cols = feature_map[key]
        X = df[cols].apply(pd.to_numeric, errors="coerce")

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        X_train_dict[key] = X_tr
        X_test_dict[key] = X_te
        y_train_dict[key] = y_tr
        y_test_dict[key] = y_te

    return feature_map, X_train_dict, X_test_dict, y_train_dict, y_test_dict


# ── Calibration metrics ──

def brier(y_true, y_prob):
    return float(brier_score_loss(y_true, y_prob))


def calibration_intercept_slope(y_true, y_prob):
    logit_p = np.clip(y_prob, 1e-15, 1 - 1e-15)
    logit_p = np.log(logit_p / (1 - logit_p))
    logit_p = logit_p.reshape(-1, 1)
    model = LogisticRegression(solver="lbfgs", random_state=RANDOM_STATE)
    model.fit(logit_p, y_true)
    intercept = float(model.intercept_[0])
    slope = float(model.coef_[0][0])
    return intercept, slope


def hosmer_lemeshow(y_true, y_prob, n_bins=10):
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df["bin"] = pd.qcut(df["y_prob"], q=n_bins, duplicates="drop", labels=False)

    obs = df.groupby("bin")["y_true"].sum().values
    total = df.groupby("bin")["y_true"].count().values
    exp = df.groupby("bin")["y_prob"].sum().values

    valid = total > 0
    if not np.all(valid):
        n_bins_actual = valid.sum()
    else:
        n_bins_actual = n_bins

    h = np.sum((obs[valid] - exp[valid]) ** 2 /
               (exp[valid] * (1 - exp[valid] / total[valid])))

    df_h = n_bins_actual - 2
    p_value = 1.0 - chi2.cdf(h, df_h) if df_h > 0 else 1.0
    return float(h), float(p_value), int(df_h)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    total = len(y_prob)
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() == 0:
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return float(ece / total)


def maximum_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    max_err = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() == 0:
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()
        max_err = max(max_err, abs(bin_acc - bin_conf))
    return float(max_err)


def compute_all_calibration_metrics(y_true, y_prob, n_bins=10):
    bs = brier(y_true, y_prob)
    intercept, slope = calibration_intercept_slope(y_true, y_prob)
    hl_stat, hl_p, hl_df = hosmer_lemeshow(y_true, y_prob, n_bins)
    ece = expected_calibration_error(y_true, y_prob, n_bins)
    mce = maximum_calibration_error(y_true, y_prob, n_bins)
    return {
        "brier_score": bs,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "hosmer_lemeshow_statistic": hl_stat,
        "hosmer_lemeshow_df": hl_df,
        "hosmer_lemeshow_p_value": hl_p,
        "ece": ece,
        "mce": mce,
        "n_bins": n_bins,
        "n_samples": len(y_true),
    }


# ── Main ──

def main():
    print("=" * 60)
    print("  SVM CALIBRATION ANALYSIS")
    print("=" * 60)

    feature_map, X_tr, X_te, y_tr, y_te = load_and_split()

    all_original_metrics = {}
    all_y_prob = {}
    all_models = {}

    calib_results = []

    for key in ["spatial", "clinical", "combined"]:
        info = FEATURE_SETS[key]
        print(f"\n{'-' * 50}")
        print(f"  {info['name']}")
        print(f"{'-' * 50}")

        # Build pipeline with best params
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel="rbf", probability=True,
                class_weight="balanced", random_state=RANDOM_STATE,
            )),
        ])
        pipeline.set_params(**info["best_params"])
        print(f"    Params: {info['best_params']}")
        pipeline.fit(X_tr[key], y_tr[key])
        all_models[key] = pipeline

        y_prob = pipeline.predict_proba(X_te[key])[:, 1]
        all_y_prob[key] = y_prob
        y_true = y_te[key].values

        # Original calibration metrics
        metrics = compute_all_calibration_metrics(y_true, y_prob, N_BINS)
        all_original_metrics[key] = metrics
        print(f"    Brier:       {metrics['brier_score']:.4f}")
        print(f"    Intercept:   {metrics['calibration_intercept']:.4f}")
        print(f"    Slope:       {metrics['calibration_slope']:.4f}")
        print(f"    HL p-value:  {metrics['hosmer_lemeshow_p_value']:.4f}")
        print(f"    ECE:         {metrics['ece']:.4f}")
        print(f"    MCE:         {metrics['mce']:.4f}")

        # Recalibration: Platt and Isotonic
        # Platt scaling: logistic regression on SVM decision values
        train_preds = pipeline.predict(X_tr[key])
        train_proba = pipeline.predict_proba(X_tr[key])[:, 1]
        train_dec = pipeline.decision_function(X_tr[key])

        platt_calib = LogisticRegression(solver="lbfgs",
                                         random_state=RANDOM_STATE)
        platt_calib.fit(train_dec.reshape(-1, 1), y_tr[key])

        iso_calib = IsotonicRegression(out_of_bounds="clip")
        iso_calib.fit(train_proba, y_tr[key])

        # Compare recalibrated probabilities
        for method_name, calc_prob in [
            ("original", lambda: y_prob),
            ("platt", lambda: platt_calib.predict_proba(
                pipeline.decision_function(X_te[key]).reshape(-1, 1))[:, 1]),
            ("isotonic", lambda: iso_calib.transform(
                pipeline.predict_proba(X_te[key])[:, 1])),
        ]:
            prob = calc_prob()
            m = compute_all_calibration_metrics(y_true, prob, N_BINS)
            calib_results.append({
                "model": info["name"],
                "recalibration": method_name,
                **m,
            })

        # Generate individual calibration curve figure (3 panels)
        fname = f"calibration_curve_{info['prefix']}.png"
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

        ref_probs = {
            "Original": y_prob,
            "Platt": platt_calib.predict_proba(
                pipeline.decision_function(X_te[key]).reshape(-1, 1))[:, 1],
            "Isotonic": iso_calib.transform(
                pipeline.predict_proba(X_te[key])[:, 1]),
        }
        for idx, (r_name, r_prob) in enumerate(ref_probs.items()):
            ax = axes[idx]
            prob_true, prob_pred = calibration_curve(
                y_true, r_prob, n_bins=N_BINS, strategy="uniform",
            )
            ax.plot(prob_pred, prob_true, marker="o", lw=2, color="blue")
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("Mean Predicted Probability")
            if idx == 0:
                ax.set_ylabel("Observed Proportion")
            ax.set_title(f"{r_name}", fontsize=11)
            ax.grid(alpha=0.3)
            bs_val = brier_score_loss(y_true, r_prob)
            ax.text(0.55, 0.12, f"Brier = {bs_val:.4f}",
                    transform=ax.transAxes, fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.7))

        fig.suptitle(f"Calibration Curves — {info['name']}", fontsize=13)
        plt.tight_layout()
        fig.savefig(CAL_DIR / fname, dpi=150)
        plt.close(fig)
        print(f"    Saved {fname}")

    # ── Combined calibration comparison figure ──
    print(f"\n{'-' * 50}")
    print("  Combined calibration comparison")
    print(f"{'-' * 50}")
    colors = {"spatial": "blue", "clinical": "green", "combined": "red"}
    line_styles = {"original": "-", "platt": "--", "isotonic": ":"}
    method_labels = {"original": "Original", "platt": "Platt", "isotonic": "Isotonic"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for idx, (method_key, method_label) in enumerate(method_labels.items()):
        ax = axes[idx]
        for key in ["spatial", "clinical", "combined"]:
            info = FEATURE_SETS[key]
            y_true = y_te[key].values

            if method_key == "original":
                prob = all_y_prob[key]
            else:
                pipeline = all_models[key]
                if method_key == "platt":
                    calib = LogisticRegression(solver="lbfgs",
                                               random_state=RANDOM_STATE)
                    calib.fit(pipeline.decision_function(X_tr[key]).reshape(-1, 1),
                              y_tr[key])
                    prob = calib.predict_proba(
                        pipeline.decision_function(X_te[key]).reshape(-1, 1))[:, 1]
                else:
                    calib = IsotonicRegression(out_of_bounds="clip")
                    calib.fit(pipeline.predict_proba(X_tr[key])[:, 1],
                              y_tr[key])
                    prob = calib.transform(
                        pipeline.predict_proba(X_te[key])[:, 1])

            prob_true, prob_pred = calibration_curve(
                y_true, prob, n_bins=N_BINS, strategy="uniform",
            )
            ax.plot(prob_pred, prob_true, marker="o", lw=2,
                    color=colors[key], label=info["name"])
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("Mean Predicted Probability")
            if idx == 0:
                ax.set_ylabel("Observed Proportion")
            ax.set_title(method_label, fontsize=12)
            ax.legend(fontsize=8, loc="lower right")
            ax.grid(alpha=0.3)

    fig.suptitle("Calibration Comparison Across Feature Sets", fontsize=14)
    plt.tight_layout()
    fig.savefig(CAL_DIR / "calibration_comparison.png", dpi=150)
    plt.close(fig)
    print("    Saved calibration_comparison.png")

    # ── Save CSV outputs ──
    calib_df = pd.DataFrame(calib_results)
    calib_df = calib_df.rename(columns={
        "brier_score": "Brier Score",
        "calibration_intercept": "Intercept",
        "calibration_slope": "Slope",
        "hosmer_lemeshow_statistic": "HL Statistic",
        "hosmer_lemeshow_p_value": "HL p-value",
        "ece": "ECE",
        "mce": "MCE",
    })
    reorder = [
        "model", "recalibration", "Brier Score", "Intercept", "Slope",
        "HL Statistic", "HL p-value", "ECE", "MCE",
        "n_bins", "n_samples",
    ]
    calib_df = calib_df[[c for c in reorder if c in calib_df.columns]]

    # brier_scores.csv ── just Brier by model and method
    brier_wide = calib_df.pivot_table(
        index="model", columns="recalibration",
        values="Brier Score", aggfunc="first",
    ).reset_index()
    brier_wide.columns.name = None
    brier_path = CAL_DIR / "brier_scores.csv"
    brier_wide.to_csv(brier_path, index=False)
    print(f"    Saved {brier_path.name}")

    # calibration_metrics.csv ── full metrics for original models only
    orig_df = calib_df[calib_df["recalibration"] == "original"].copy()
    orig_df = orig_df.drop(columns=["recalibration"])
    metrics_path = CAL_DIR / "calibration_metrics.csv"
    orig_df.to_csv(metrics_path, index=False)
    print(f"    Saved {metrics_path.name}")

    # ── Save JSON per model ──
    for key in ["spatial", "clinical", "combined"]:
        info = FEATURE_SETS[key]
        path = CAL_DIR / f"calibration_metrics_{info['prefix']}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(all_original_metrics[key], f, indent=2)

    # ── Generate report ──
    rows = []
    for key in ["spatial", "clinical", "combined"]:
        m = all_original_metrics[key]
        rows.append({
            "Model": FEATURE_SETS[key]["name"],
            "Brier Score": f"{m['brier_score']:.4f}",
            "Intercept": f"{m['calibration_intercept']:.4f}",
            "Slope": f"{m['calibration_slope']:.4f}",
            "ECE": f"{m['ece']:.4f}",
            "MCE": f"{m['mce']:.4f}",
            "HL p-value": f"{m['hosmer_lemeshow_p_value']:.4f}",
        })
    report_df = pd.DataFrame(rows)

    recalib_summary = calib_df[calib_df["recalibration"] != "original"][
        ["model", "recalibration", "Brier Score", "ECE", "MCE"]
    ].copy()
    recalib_summary.columns = [
        "Model", "Recalibration", "Brier Score", "ECE", "MCE",
    ]

    # Per-model detail
    for key in ["spatial", "clinical", "combined"]:
        m = all_original_metrics[key]
        hl_sig = "significant" if m["hosmer_lemeshow_p_value"] < 0.05 else "not significant"

    print(f"\n{'=' * 60}")
    print(f"  All outputs in: {CAL_DIR}")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
