#!/usr/bin/env python3
"""Evaluate stability of the final SVM survival classifier across repeated splits.

Uses RepeatedStratifiedKFold (5 splits x 10 repeats = 50 runs) with the same
leakage-safe Pipeline [StandardScaler -> SVC(RBF)] and fixed hyperparameters
(C=100, gamma=0.001, class_weight="balanced"). No hyperparameter search.

Outputs (under outputs/multimodal_lobewise_svm/):
  - model_stability_results.csv   — per-fold metrics for all 50 runs
  - model_stability_summary.json  — mean / std / min / max across runs
  - auc_distribution.png          — histogram + boxplot of ROC-AUC
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

METADATA_COLS = {"patient_id", "risk_label", "OS_months",
                 "lobe_assignment_reliable"}
DROP_COLS = {"OS_months", "lobe_assignment_reliable"}
TARGET_COL = "risk_label"

DEFAULT_INPUT = "outputs/multimodal_lobewise/merged_features.csv"
DEFAULT_OUTPUT_DIR = "outputs/multimodal_lobewise_svm"

BEST_C = 100
BEST_GAMMA = 0.001

N_SPLITS = 5
N_REPEATS = 10
RANDOM_STATE = 42


def _resolve_path(root: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (root / p)


def _verify_integrity(X: pd.DataFrame, y: pd.Series,
                      patient_ids: pd.Series) -> None:
    assert patient_ids.name == "patient_id", \
        "patient_id series not found or misnamed."
    assert X.columns.is_unique, \
        f"Duplicate columns: {X.columns[X.columns.duplicated()].tolist()}"
    non_numeric = X.select_dtypes(exclude=["number"]).columns.tolist()
    assert len(non_numeric) == 0, \
        f"Non-numeric feature column(s): {non_numeric}"
    leaked = X.columns.intersection(METADATA_COLS)
    assert leaked.empty, \
        f"Metadata leakage detected: {leaked.tolist()}"
    assert X.isnull().sum().sum() == 0, \
        f"NaNs remain in feature matrix: {X.isnull().sum().sum()}"
    assert y.isnull().sum() == 0, "NaNs in target vector."


def load_dataset(in_csv: Path) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    df = pd.read_csv(in_csv)
    assert TARGET_COL in df.columns, \
        f"Target column '{TARGET_COL}' not found."
    patient_ids = df["patient_id"].astype(str)
    y = df[TARGET_COL].astype(int)
    feature_cols = [c for c in df.columns
                    if c not in METADATA_COLS and c not in DROP_COLS]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    _verify_integrity(X, y, patient_ids)
    return patient_ids, X, y


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf", C=BEST_C, gamma=BEST_GAMMA,
            class_weight="balanced", probability=True, random_state=RANDOM_STATE,
        )),
    ])


def evaluate_model(model: Pipeline, X_test: pd.DataFrame,
                   y_test: pd.Series) -> dict[str, float]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
    }


def plot_auc_distribution(aucs: np.ndarray, save_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(aucs, bins=12, color="steelblue", edgecolor="black",
             alpha=0.8)
    ax1.axvline(np.mean(aucs), color="red", linestyle="--", linewidth=2,
                label=f"Mean = {np.mean(aucs):.4f}")
    ax1.axvline(np.median(aucs), color="darkorange", linestyle=":",
                linewidth=2, label=f"Median = {np.median(aucs):.4f}")
    ax1.set_xlabel("ROC-AUC")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of ROC-AUC across 50 Runs")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    bp = ax2.boxplot(aucs, vert=True, patch_artist=True,
                     widths=0.4)
    bp["boxes"][0].set_facecolor("steelblue")
    bp["boxes"][0].set_alpha(0.7)
    ax2.axhline(np.mean(aucs), color="red", linestyle="--", linewidth=2,
                label=f"Mean = {np.mean(aucs):.4f}")
    ax2.set_ylabel("ROC-AUC")
    ax2.set_title("Boxplot of ROC-AUC across 50 Runs")
    ax2.set_xticks([])
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def compute_summary(results: pd.DataFrame) -> dict[str, dict[str, float]]:
    metric_cols = ["roc_auc", "accuracy", "precision", "recall", "f1_score"]
    summary = {}
    for col in metric_cols:
        vals = results[col].values
        summary[col] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return summary


def print_interpretation(summary: dict[str, dict[str, float]],
                         n_total: int) -> None:
    auc = summary["roc_auc"]
    acc = summary["accuracy"]
    auc_range = auc["max"] - auc["min"]

    print()
    print("=" * 72)
    print("  STABILITY ANALYSIS — SVM (RBF) Repeated StratifiedKFold")
    print("=" * 72)
    print(f"  Total evaluations:        {n_total}")
    print(f"  Configuration:            {N_SPLITS}-fold x {N_REPEATS} repeats")
    print()
    print(f"  ROC-AUC:                  {auc['mean']:.4f} ± {auc['std']:.4f}")
    print(f"    Range:                  [{auc['min']:.4f}, {auc['max']:.4f}]")
    print(f"    Spread:                 {auc_range:.4f}")
    print()
    print(f"  Accuracy:                 {acc['mean']:.4f} ± {acc['std']:.4f}")
    print(f"    Range:                  [{acc['min']:.4f}, {acc['max']:.4f}]")
    print()
    for col in ["precision", "recall", "f1_score"]:
        s = summary[col]
        print(f"  {col.replace('_', ' ').title():15s}  "
              f"{s['mean']:.4f} ± {s['std']:.4f}  "
              f"[{s['min']:.4f}, {s['max']:.4f}]")
    print()
    print("=" * 72)
    print("  INTERPRETATION")
    print("=" * 72)
    print()

    cv_pct = auc["std"] / max(auc["mean"], 1e-8) * 100
    if cv_pct < 10:
        print(f"  The model demonstrates STABLE performance across repeated")
        print(f"  splits (CV of ROC-AUC = {cv_pct:.1f}%).")
    elif cv_pct < 20:
        print(f"  The model shows MODERATELY STABLE performance across repeated")
        print(f"  splits (CV of ROC-AUC = {cv_pct:.1f}%).")
    else:
        print(f"  The model exhibits HIGH VARIABILITY across repeated splits")
        print(f"  (CV of ROC-AUC = {cv_pct:.1f}%).")

    if auc_range < 0.15:
        print(f"  Narrow AUC range ({auc_range:.3f}) suggests the model does not")
        print(f"  depend strongly on the particular random split.")
    elif auc_range < 0.30:
        print(f"  Moderate AUC range ({auc_range:.3f}) indicates some dependence")
        print(f"  on the random split; results should be interpreted with")
        print(f"  caution.")
    else:
        print(f"  Wide AUC range ({auc_range:.3f}) indicates strong dependence")
        print(f"  on the random split, which undermines reliability.")

    if auc["mean"] - auc["std"] > 0.5:
        print(f"  Even at one standard deviation below the mean, the ROC-AUC")
        print(f"  exceeds 0.5, indicating consistent above-chance performance.")
    else:
        print(f"  The lower bound (mean - std = {auc['mean'] - auc['std']:.3f})")
        print(f"  falls near or below 0.5, suggesting the model may sometimes")
        print(f"  perform no better than random chance.")

    print()
    print("  Implications for reliability:")
    print()
    print(f"  - A single train/test split (as in the main pipeline) provides a")
    print(f"    point estimate of {auc['mean']:.3f}, but individual splits range")
    print(f"    from {auc['min']:.3f} to {auc['max']:.3f}.")
    print(f"  - Reporting the mean ± std across repeated splits gives a more")
    print(f"    honest assessment of expected generalization performance.")
    print(f"  - If variability is high, bagging or ensemble strategies may")
    print(f"    improve stability at the cost of interpretability.")
    print(f"  - The 50-run distribution provides a basis for comparing future")
    print(f"    model improvements against a stable baseline.")
    print()
    print("=" * 72)
    print()


def evaluate_stability(in_csv: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    patient_ids, X, y = load_dataset(in_csv)
    print(f"Loaded dataset: {len(patient_ids)} patients, "
          f"{X.shape[1]} features\n")

    rskf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE,
    )

    rows = []
    fold_idx = 0

    for repeat_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        _verify_integrity(X_train, y_train, patient_ids.iloc[train_idx])

        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_test, y_test)

        rows.append({
            "repeat": repeat_idx + 1,
            "fold": (fold_idx % N_SPLITS) + 1,
            "fold_index": fold_idx + 1,
            "train_size": len(X_train),
            "test_size": len(X_test),
            **metrics,
        })

        fold_idx += 1
        if (fold_idx) % 10 == 0 or fold_idx == 1:
            print(f"  Completed {fold_idx}/{N_SPLITS * N_REPEATS} evaluations"
                  f"  (repeat {repeat_idx + 1}, "
                  f"AUC = {metrics['roc_auc']:.4f})")

    results_df = pd.DataFrame(rows)
    summary = compute_summary(results_df)

    results_df.to_csv(output_dir / "model_stability_results.csv", index=False)
    print(f"\nWrote {output_dir / 'model_stability_results.csv'}")

    with (output_dir / "model_stability_summary.json").open("w",
                                                             encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {output_dir / 'model_stability_summary.json'}")

    auc_path = output_dir / "auc_distribution.png"
    plot_auc_distribution(results_df["roc_auc"].values, auc_path)
    print(f"Wrote {auc_path}")

    print_interpretation(summary, len(results_df))

    return 0


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Evaluate SVM stability across repeated stratified folds.",
    )
    parser.add_argument("--input", default=None,
                        help=f"Raw merged CSV (default: {DEFAULT_INPUT})")
    parser.add_argument("--output-dir", default=None,
                        help=f"Output dir (default: {DEFAULT_OUTPUT_DIR})")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    in_csv = _resolve_path(root, args.input or DEFAULT_INPUT)
    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")
    output_dir = _resolve_path(root, args.output_dir or DEFAULT_OUTPUT_DIR)
    return evaluate_stability(in_csv, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
