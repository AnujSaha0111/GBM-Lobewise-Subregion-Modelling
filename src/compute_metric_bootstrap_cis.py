#!/usr/bin/env python3

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "outputs" / "multimodal_lobewise" / "merged_features_with_metadata.csv"
OUTPUT_DIR = ROOT / "outputs" / "confidence_intervals"
METRICS_DIR = ROOT / "outputs" / "multimodal_lobewise_svm_comparison"
PREDICTIONS_CSV = ROOT / "outputs" / "multimodal_lobewise_svm" / "predictions.csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_BOOTSTRAP = 5000

SPATIAL_FEATURE_PREFIXES = ("T1_", "T2_", "T1GD_", "FLAIR_")
METADATA_FEATURE_COLS = ["age", "sex", "idh", "mgmt", "who_grade", "eor"]

PARAM_GRID = {
    "svm__C": [0.01, 0.1, 1, 10, 100],
    "svm__gamma": ["scale", 0.0001, 0.001, 0.01, 0.1],
}

BEST_PARAMS = {
    "spatial": {"svm__C": 100, "svm__gamma": 0.001},
    "metadata": {"svm__C": 1, "svm__gamma": 0.001},
    "combined": {"svm__C": 1, "svm__gamma": "scale"},
}

METRIC_NAMES = {
    "accuracy": ("Accuracy", "accuracy"),
    "precision": ("Precision", "precision"),
    "recall": ("Recall", "recall"),
    "f1_score": ("F1-score", "f1_score"),
}


def load_known_metrics():
    """Load the original point estimates from saved metrics JSONs."""
    known = {}
    for key, prefix in [("spatial", "spatial"), ("metadata", "metadata"),
                         ("combined", "combined")]:
        path = METRICS_DIR / f"metrics_{prefix}.json"
        with path.open("r") as f:
            data = json.load(f)
        known[key] = {
            "accuracy": data["accuracy"],
            "precision": data["precision"],
            "recall": data["recall"],
            "f1_score": data["f1_score"],
        }
    return known


def load_and_split():
    df = pd.read_csv(INPUT_CSV)
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    y = df["risk_label"].astype(int)

    spatial_cols = [c for c in df.columns if c.startswith(SPATIAL_FEATURE_PREFIXES)]
    meta_cols = [c for c in METADATA_FEATURE_COLS if c in df.columns]
    combined_cols = spatial_cols + meta_cols

    feature_map = {
        "spatial": spatial_cols,
        "metadata": meta_cols,
        "combined": combined_cols,
    }

    splits = {}
    for key, cols in feature_map.items():
        X = df[cols].apply(pd.to_numeric, errors="coerce")
        X_tr, X_te, y_tr, y_te, ids_tr, ids_te = train_test_split(
            X, y, df["patient_id"],
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        splits[key] = (X_tr, X_te, y_tr, y_te, ids_tr, ids_te)
    return splits


def train_with_gridsearch(X_train, y_train):
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
    return gs.best_estimator_


def train_and_predict(X_train, y_train, X_test, params):
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf", probability=True, class_weight="balanced",
            random_state=RANDOM_STATE,
            C=params["svm__C"],
            gamma=params["svm__gamma"],
        )),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline.predict(X_test)


def bootstrap_metrics_ci(y_true, y_pred, n_bootstraps, random_state):
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    metrics = {"accuracy": [], "precision": [], "recall": [], "f1_score": []}
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        pb = y_pred[idx]
        if len(np.unique(yb)) < 2:
            continue
        metrics["accuracy"].append(accuracy_score(yb, pb))
        metrics["precision"].append(precision_score(yb, pb, zero_division=0))
        metrics["recall"].append(recall_score(yb, pb, zero_division=0))
        metrics["f1_score"].append(f1_score(yb, pb, zero_division=0))
    results = {}
    for m in ["accuracy", "precision", "recall", "f1_score"]:
        arr = np.array(metrics[m])
        results[m] = {
            "ci_lower": float(np.percentile(arr, 2.5)),
            "ci_upper": float(np.percentile(arr, 97.5)),
            "n_valid_bootstrap": len(arr),
        }
    return results


def round_ci(val):
    """Round to 4 decimal places for display."""
    return round(val, 4)


def fmt_pct(val):
    """Format as percentage string with 2 decimal places."""
    return f"{val * 100:.2f}%"


def main():
    print("Loading known metrics...")
    known_metrics = load_known_metrics()
    print("Loading data and splitting...")
    splits = load_and_split()
    all_results = {}

    # Spatial: use saved predictions CSV for exact match
    print("\nSpatial model: loading saved predictions...")
    pdf = pd.read_csv(PREDICTIONS_CSV)
    y_true_spatial = pdf["true_label"].values
    y_pred_spatial = pdf["predicted_label"].values
    ci_spatial = bootstrap_metrics_ci(y_true_spatial, y_pred_spatial, N_BOOTSTRAP, RANDOM_STATE)

    # Verify match with known metrics
    for m in ["accuracy", "precision", "recall", "f1_score"]:
        obs = accuracy_score if m == "accuracy" else (precision_score if m == "precision" else (recall_score if m == "recall" else f1_score))
        computed = obs(y_true_spatial, y_pred_spatial, zero_division=0) if m != "accuracy" else obs(y_true_spatial, y_pred_spatial)
        known = known_metrics["spatial"][m]
        print(f"  {m}: computed={computed:.6f}, known={known:.6f}, match={abs(computed - known) < 1e-10}")

    all_results["spatial"] = {"cdf": {}, "metrics": known_metrics["spatial"]}
    for m in ["accuracy", "precision", "recall", "f1_score"]:
        all_results["spatial"]["cdf"][m] = ci_spatial[m]

    # Metadata model: train with grid search (as original)
    for key in ["metadata", "combined"]:
        print(f"\n{key} model: training with GridSearchCV...")
        X_tr, X_te, y_tr, y_te, ids_tr, ids_te = splits[key]
        # Use grid search for exact match with original pipeline
        model = train_with_gridsearch(X_tr, y_tr)
        y_pred = model.predict(X_te)
        ci = bootstrap_metrics_ci(y_te.values, y_pred, N_BOOTSTRAP, RANDOM_STATE)
        # Verify match
        for m in ["accuracy", "precision", "recall", "f1_score"]:
            obs_func = accuracy_score if m == "accuracy" else (precision_score if m == "precision" else (recall_score if m == "recall" else f1_score))
            args = (y_te, y_pred) if m == "accuracy" else (y_te, y_pred, {"zero_division": 0})
            computed = obs_func(y_te.values, y_pred)
            known = known_metrics[key][m]
            print(f"  {m}: computed={computed:.6f}, known={known:.6f}, match={abs(computed - known) < 1e-10}")
        all_results[key] = {"cdf": {}, "metrics": known_metrics[key]}
        for m in ["accuracy", "precision", "recall", "f1_score"]:
            all_results[key]["cdf"][m] = ci[m]

    # Combine into final structure
    final = {}
    for key in ["spatial", "metadata", "combined"]:
        final[key] = {}
        for m in ["accuracy", "precision", "recall", "f1_score"]:
            pe = all_results[key]["metrics"][m]
            ci = all_results[key]["cdf"][m]
            final[key][m] = {
                "point_estimate": pe,
                "ci_lower": ci["ci_lower"],
                "ci_upper": ci["ci_upper"],
                "n_valid_bootstrap": ci["n_valid_bootstrap"],
            }

    # Print final results
    model_labels = {"spatial": "Spatial", "metadata": "Clinical-molecular", "combined": "Combined"}
    metric_labels = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1_score": "F1-score"}
    for key in ["spatial", "metadata", "combined"]:
        print(f"\n{model_labels[key]}:")
        for m in ["accuracy", "precision", "recall", "f1_score"]:
            r = final[key][m]
            print(f"  {metric_labels[m]}: {fmt_pct(r['point_estimate'])} [{fmt_pct(r['ci_lower'])}, {fmt_pct(r['ci_upper'])}]")

    # ── Save outputs ──
    json_path = OUTPUT_DIR / "metric_cis.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    print(f"\nWrote {json_path}")

    rows = []
    for model_key in ["spatial", "metadata", "combined"]:
        for m in ["accuracy", "precision", "recall", "f1_score"]:
            r = final[model_key][m]
            rows.append({
                "Model": model_key,
                "Metric": m,
                "PointEstimate": round_ci(r["point_estimate"]),
                "CI_Lower": round_ci(r["ci_lower"]),
                "CI_Upper": round_ci(r["ci_upper"]),
                "CI_Display": f"{fmt_pct(r['point_estimate'])} [{fmt_pct(r['ci_lower'])}, {fmt_pct(r['ci_upper'])}]",
            })
    csv_path = OUTPUT_DIR / "metric_cis.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    lines = [
        "# Confidence Interval Report\n",
        "Bootstrap resampling (5000 iterations, percentile method) on the held-out test set (n=99).\n",
        "| Model | Metric | Point Estimate | 95% CI |",
        "|---|---|---|---|",
    ]
    for model_key in ["spatial", "metadata", "combined"]:
        for m in ["accuracy", "precision", "recall", "f1_score"]:
            r = final[model_key][m]
            lines.append(
                f"| {model_labels[model_key]} ({model_key}) | {metric_labels[m]} "
                f"| {fmt_pct(r['point_estimate'])} | [{fmt_pct(r['ci_lower'])}, {fmt_pct(r['ci_upper'])}] |"
            )
    lines.append("")
    md_path = OUTPUT_DIR / "confidence_interval_report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {md_path}")

    print(f"\n{'='*50}")
    print("  DONE")
    print(f"{'='*50}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
