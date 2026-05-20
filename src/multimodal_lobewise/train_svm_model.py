#!/usr/bin/env python3
"""Train and evaluate an SVM multimodal lobewise-subregion survival classifier.

Loads merged_features.csv (raw, un-scaled features), performs a stratified
train/test split, and uses a Pipeline [StandardScaler → SVC(RBF)] with
GridSearchCV so that scaling is fit ONLY on training data (no leakage).

All artifacts are saved under outputs/multimodal_lobewise_svm/.
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

METADATA_COLS = {"patient_id", "risk_label", "OS_months",
                 "lobe_assignment_reliable"}
DROP_COLS = {"OS_months", "lobe_assignment_reliable"}
TARGET_COL = "risk_label"
LABEL_NAMES = ["low-risk", "high-risk"]

DEFAULT_INPUT = "outputs/multimodal_lobewise/merged_features.csv"
DEFAULT_OUTPUT_DIR = "outputs/multimodal_lobewise_svm"

PARAM_GRID: dict[str, list[object]] = {
    "svm__C": [0.01, 0.1, 1, 10, 100],
    "svm__gamma": ["scale", 0.001, 0.01, 0.1, 1],
}


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


def train_svm(
    X_train: pd.DataFrame, y_train: pd.Series,
) -> tuple[Pipeline, dict[str, object], float, pd.DataFrame]:
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True,
                    class_weight="balanced", random_state=42)),
    ])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=PARAM_GRID,
        scoring="roc_auc",
        cv=skf,
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_, gs.best_score_, \
        pd.DataFrame(gs.cv_results_)


def evaluate_model(model: Pipeline, X_test: pd.DataFrame,
                   y_test: pd.Series) -> dict[str, object]:
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
            y_test, y_pred, target_names=LABEL_NAMES,
            output_dict=True, zero_division=0,
        ),
    }
    return metrics


def plot_roc_curve(y_test: pd.Series, y_proba: np.ndarray,
                   save_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2,
            label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--",
            label="Chance")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic — SVM (RBF, corrected)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix_heatmap(y_test: pd.Series, y_pred: np.ndarray,
                                  save_path: Path) -> None:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
                cbar=False, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix — SVM (RBF, corrected)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def save_artifacts(model: Pipeline, metrics: dict[str, object],
                   predictions: pd.DataFrame, best_params: dict[str, object],
                   cv_results: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "svm_model.pkl")
    report = metrics.get("classification_report")
    if report is not None:
        metrics["classification_report"] = json.loads(json.dumps(report))
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    with (output_dir / "best_params.json").open("w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    cv_results.to_csv(output_dir / "cv_results.csv", index=False)


def train_svm_pipeline(in_csv: Path, output_dir: Path) -> int:
    patient_ids, X, y = load_dataset(in_csv)

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, patient_ids, test_size=0.2, random_state=42, stratify=y,
    )
    _verify_integrity(X_train, y_train, ids_train)

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape:  {X_test.shape}")
    print(f"Train class distribution:\n{y_train.value_counts().to_string()}")
    print(f"Test class distribution:\n{y_test.value_counts().to_string()}")
    print()

    model, best_params, best_cv_auc, cv_results_df = train_svm(X_train,
                                                                y_train)

    print(f"\nBest CV ROC-AUC: {best_cv_auc:.4f}")
    print(f"Best parameters: {best_params}")
    print()

    metrics = evaluate_model(model, X_test, y_test)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    predictions_df = pd.DataFrame({
        "patient_id": ids_test.values,
        "true_label": y_test.values,
        "predicted_label": y_pred.astype(int),
        "prediction_probability": y_proba,
    })

    save_artifacts(model, metrics, predictions_df, best_params,
                   cv_results_df, output_dir)

    plot_roc_curve(y_test, y_proba, output_dir / "roc_curve.png")
    plot_confusion_matrix_heatmap(y_test, y_pred,
                                  output_dir / "confusion_matrix.png")

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
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES,
                                zero_division=0))
    print()

    fp = cm[0][1]
    fn = cm[1][0]
    print("Interpretation:")
    print(f"  FP={fp}: low-risk misclassified as high-risk"
          f" (potential over-treatment).")
    print(f"  FN={fn}: high-risk misclassified as low-risk"
          f" (missed intervention — more dangerous).")
    print(f"  Recall={metrics['recall']:.3f} — critical in survival"
          f" prediction.")
    print()

    print(f"Wrote {output_dir / 'svm_model.pkl'}")
    print(f"Wrote {output_dir / 'metrics.json'}")
    print(f"Wrote {output_dir / 'predictions.csv'}")
    print(f"Wrote {output_dir / 'best_params.json'}")
    print(f"Wrote {output_dir / 'cv_results.csv'}")
    print(f"Wrote {output_dir / 'roc_curve.png'}")
    print(f"Wrote {output_dir / 'confusion_matrix.png'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train SVM survival classifier (corrected — no leakage).",
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
    return train_svm_pipeline(in_csv, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
