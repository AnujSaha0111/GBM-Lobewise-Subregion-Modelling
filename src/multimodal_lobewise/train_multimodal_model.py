#!/usr/bin/env python3
"""Train and evaluate a multimodal lobewise-subregion survival risk classifier.

Loads processed_features.csv, trains an XGBoost classifier with
scale_pos_weight imbalance handling and early stopping, evaluates on a
held-out test set, and saves model artifacts, metrics, and plots.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

xgb.set_config(verbosity=0)

# ── Constants ──

METADATA_COLS = {"patient_id", "risk_label"}
TARGET_COL = "risk_label"
LABEL_NAMES = ["low-risk", "high-risk"]

DEFAULT_INPUT = "outputs/multimodal_lobewise/processed_features.csv"
DEFAULT_OUTPUT_DIR = "outputs/multimodal_lobewise"

XGB_PARAMS: dict[str, object] = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": 42,
    "objective": "binary:logistic",
    "tree_method": "hist",
    "verbosity": 0,
}


# ── Helpers ──

def _resolve_path(root: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (root / p)


def _verify_integrity(X: pd.DataFrame, y: pd.Series, patient_ids: pd.Series) -> None:
    assert patient_ids.name == "patient_id", \
        "patient_id series not found or misnamed."

    assert X.columns.is_unique, \
        f"Duplicate columns in feature matrix: {X.columns[X.columns.duplicated()].tolist()}"

    non_numeric = X.select_dtypes(exclude=["number"]).columns.tolist()
    assert len(non_numeric) == 0, \
        f"Non-numeric feature column(s): {non_numeric}"

    assert "patient_id" not in X.columns, \
        "patient_id leaked into feature matrix."
    assert "risk_label" not in X.columns, \
        "risk_label leaked into feature matrix."
    assert X.columns.intersection(METADATA_COLS).empty, \
        f"Metadata leakage detected: {X.columns.intersection(METADATA_COLS).tolist()}"

    assert X.isnull().sum().sum() == 0, \
        f"NaNs remain in feature matrix: {X.isnull().sum().sum()}"
    assert y.isnull().sum() == 0, \
        "NaNs in target vector."


def _compute_scale_pos_weight(y: pd.Series) -> float:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        return 1.0
    return float(n_neg) / float(n_pos)


# ── Public Functions ──

def load_dataset(in_csv: Path) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Load and split the processed multimodal dataset."""
    df = pd.read_csv(in_csv)

    assert TARGET_COL in df.columns, \
        f"Target column '{TARGET_COL}' not found in input."

    patient_ids = df["patient_id"].astype(str)
    y = df[TARGET_COL].astype(int)

    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    _verify_integrity(X, y, patient_ids)

    return patient_ids, X, y


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> xgb.XGBClassifier:
    """Train an XGBoost classifier with imbalance handling and early stopping."""
    params = dict(XGB_PARAMS)
    params["scale_pos_weight"] = _compute_scale_pos_weight(y_train)

    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    return model


def evaluate_model(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, object]:
    """Compute all evaluation metrics on the test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred,
            target_names=LABEL_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }
    return metrics


def plot_roc_curve(y_test: pd.Series, y_proba: np.ndarray, save_path: Path) -> None:
    """Plot and save the ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2,
            label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Chance")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic — Multimodal XGBoost")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix_heatmap(y_test: pd.Series, y_pred: np.ndarray, save_path: Path) -> None:
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
                cbar=False, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix — Multimodal XGBoost")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def save_artifacts(
    model: xgb.XGBClassifier,
    metrics: dict[str, object],
    predictions: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save model, metrics JSON, and predictions CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_dir / "model.pkl")

    report = metrics.get("classification_report")
    if report is not None:
        metrics["classification_report"] = json.loads(json.dumps(report))

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    predictions.to_csv(output_dir / "predictions.csv", index=False)


def train_multimodal_model(in_csv: Path, output_dir: Path) -> int:
    """Full training pipeline: load, split, train, evaluate, save."""
    patient_ids, X, y = load_dataset(in_csv)

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, patient_ids,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Verify no metadata in training features
    _verify_integrity(X_train, y_train, ids_train)

    model = train_model(X_train, y_train, X_test, y_test)
    metrics = evaluate_model(model, X_test, y_test)

    # Predictions for output
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    predictions_df = pd.DataFrame({
        "patient_id": ids_test.values,
        "true_label": y_test.values,
        "predicted_label": y_pred.astype(int),
        "prediction_probability": y_proba,
    })

    save_artifacts(model, metrics, predictions_df, output_dir)

    plot_roc_curve(y_test, y_proba, output_dir / "roc_curve.png")
    plot_confusion_matrix_heatmap(y_test, y_pred, output_dir / "confusion_matrix.png")

    # Summary prints
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape:  {X_test.shape}")
    print(f"Train class distribution:\n{y_train.value_counts().to_string()}")
    print(f"Test class distribution:\n{y_test.value_counts().to_string()}")
    print()
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"Precision:          {metrics['precision']:.4f}")
    print(f"Recall:             {metrics['recall']:.4f}")
    print(f"F1-score:           {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:            {metrics['roc_auc']:.4f}")
    print()
    cm = metrics["confusion_matrix"]
    print(f"Confusion Matrix:")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")
    print()
    print(f"Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=LABEL_NAMES,
        zero_division=0,
    ))
    print(f"Wrote {output_dir / 'model.pkl'}")
    print(f"Wrote {output_dir / 'metrics.json'}")
    print(f"Wrote {output_dir / 'predictions.csv'}")
    print(f"Wrote {output_dir / 'roc_curve.png'}")
    print(f"Wrote {output_dir / 'confusion_matrix.png'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate multimodal lobewise-subregion survival risk classifier.",
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

    return train_multimodal_model(in_csv, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
