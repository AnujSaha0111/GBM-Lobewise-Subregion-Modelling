#!/usr/bin/env python3

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
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


def _resolve_path(root: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (root / p)


def load_dataset(in_csv: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(in_csv)
    y = df[TARGET_COL].astype(int)
    feature_cols = [c for c in df.columns
                    if c not in METADATA_COLS and c not in DROP_COLS]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    return X, y


def run_roc_analysis(X: pd.DataFrame, y: pd.Series) -> dict:
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=100, gamma=0.001,
                    probability=True, class_weight="balanced",
                    random_state=42)),
    ])

    rskf = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=10, random_state=42)

    mean_fpr = np.linspace(0, 1, 100)
    n_splits = 5
    repeat_tprs = []
    repeat_aucs = []

    splits = list(rskf.split(X, y))
    for rep_idx in range(10):
        fold_tprs = []
        for fold_idx in range(n_splits):
            split_idx = rep_idx * n_splits + fold_idx
            train_idx, test_idx = splits[split_idx]
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            pipeline.fit(X_train, y_train)
            y_proba = pipeline.predict_proba(X_test)[:, 1]

            fpr, tpr, _ = roc_curve(y_test, y_proba)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tpr[-1] = 1.0
            fold_tprs.append(interp_tpr)

        repeat_mean_tpr = np.mean(fold_tprs, axis=0)
        repeat_tprs.append(repeat_mean_tpr)
        repeat_aucs.append(auc(mean_fpr, repeat_mean_tpr))

    overall_mean_tpr = np.mean(repeat_tprs, axis=0)
    overall_mean_auc = auc(mean_fpr, overall_mean_tpr)
    std_auc = np.std(repeat_aucs)

    return {
        "mean_fpr": mean_fpr,
        "repeat_tprs": repeat_tprs,
        "repeat_aucs": repeat_aucs,
        "overall_mean_tpr": overall_mean_tpr,
        "overall_mean_auc": overall_mean_auc,
        "std_auc": std_auc,
    }


def plot_cv_roc(results: dict, save_path: Path) -> None:
    mean_fpr = results["mean_fpr"]
    repeat_tprs = results["repeat_tprs"]
    overall_mean_tpr = results["overall_mean_tpr"]
    overall_mean_auc = results["overall_mean_auc"]
    std_auc = results["std_auc"]

    fig, ax = plt.subplots(figsize=(8, 6))

    for tpr in repeat_tprs:
        ax.plot(mean_fpr, tpr, color="grey", alpha=0.3, lw=0.8,
                linestyle=":")

    ax.plot(mean_fpr, overall_mean_tpr, color="blue", lw=2.5,
            label=f"Mean ROC-AUC = {overall_mean_auc:.3f} ± {std_auc:.3f}")

    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--",
            label="Chance")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("10x5-Fold Cross-Validated ROC - SVM (RBF)", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    in_csv = _resolve_path(root, DEFAULT_INPUT)
    out_dir = _resolve_path(root, DEFAULT_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {in_csv}")
    X, y = load_dataset(in_csv)
    print(f"Data shape: {X.shape}, positive class: {y.sum()} / {len(y)}")

    print("Running 10×5-fold CV ROC analysis...")
    results = run_roc_analysis(X, y)

    fig_path = out_dir / "cv_mean_roc_curve.png"
    plot_cv_roc(results, fig_path)
    print(f"Saved figure: {fig_path}")

    repeat_aucs = results["repeat_aucs"]
    summary = {
        "mean_auc": round(results["overall_mean_auc"], 4),
        "std_auc": round(results["std_auc"], 4),
        "min_auc": round(float(np.min(repeat_aucs)), 4),
        "max_auc": round(float(np.max(repeat_aucs)), 4),
        "repeat_aucs": [round(float(v), 4) for v in repeat_aucs],
    }
    json_path = out_dir / "cv_auc_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {json_path}")

    print()
    print(f"Mean ROC-AUC  = {summary['mean_auc']:.4f} ± {summary['std_auc']:.4f}")
    print(f"Best  repeat  = {summary['max_auc']:.4f}")
    print(f"Worst repeat  = {summary['min_auc']:.4f}")
    print()

    spread = summary["max_auc"] - summary["min_auc"]
    cv = summary["std_auc"] / summary["mean_auc"] if summary["mean_auc"] > 0 else 0
    if cv < 0.05:
        print(f"Tight clustering (CV={cv:.3f}, spread={spread:.4f}) -> "
              "stable generalization across splits.")
    elif spread > 0.15:
        print(f"Wide spread (CV={cv:.3f}, spread={spread:.4f}) -> "
              "split-sensitive; consider increasing repeats or sample size.")
    else:
        print(f"Moderate spread (CV={cv:.3f}, spread={spread:.4f}) -> "
              "reasonable stability; review per-fold diagnostics.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
