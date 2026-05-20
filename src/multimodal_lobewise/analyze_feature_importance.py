#!/usr/bin/env python3
"""Analyze multimodal lobe-subregion feature importance for survival risk prediction.

Loads a trained XGBoost model and processed dataset, computes XGBoost gain-based
importance and SHAP values, and produces grouped summaries by modality, lobe, and
subregion. All plots and tables are saved under outputs/multimodal_lobewise/.
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
import shap
import xgboost as xgb

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

xgb.set_config(verbosity=0)

# ── Constants ──

METADATA_COLS = {"patient_id", "risk_label"}
TARGET_COL = "risk_label"
LABEL_NAMES = ["low-risk", "high-risk"]

LOBES = ["frontal", "temporal", "parietal", "occipital"]

DEFAULT_MODEL = "outputs/multimodal_lobewise/model.pkl"
DEFAULT_DATA = "outputs/multimodal_lobewise/processed_features.csv"
DEFAULT_OUTPUT_DIR = "outputs/multimodal_lobewise"


# ── Helpers ──

def _resolve_path(root: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (root / p)


def _verify_integrity(X: pd.DataFrame, model_features: list[str]) -> None:
    assert X.columns.is_unique, \
        f"Duplicate columns: {X.columns[X.columns.duplicated()].tolist()}"

    non_numeric = X.select_dtypes(exclude=["number"]).columns.tolist()
    assert len(non_numeric) == 0, \
        f"Non-numeric feature column(s): {non_numeric}"

    assert X.columns.intersection(METADATA_COLS).empty, \
        f"Metadata leakage: {X.columns.intersection(METADATA_COLS).tolist()}"

    assert X.isnull().sum().sum() == 0, \
        f"NaNs remain: {X.isnull().sum().sum()}"

    assert list(X.columns) == model_features, \
        "Feature columns do not match model training features."


def parse_feature_name(name: str) -> dict[str, str]:
    """Parse a feature name into modality, lobe, and subregion components.

    Feature names follow the pattern ``{MODALITY}_{base_feature}`` where
    *base_feature* is one of the 16 original spatial features.

    Examples:
        T1GD_frontal_en_ratio   -> modality=T1GD, lobe=frontal,  subregion=en
        T2_global_nc_en_ratio   -> modality=T2,   lobe=global,   subregion=nc_en
        FLAIR_tumor_burden_index -> modality=FLAIR, lobe=global, subregion=tumor_burden
    """
    modality, rest = name.split("_", 1)

    for lobe in LOBES:
        prefix = f"{lobe}_"
        if rest.startswith(prefix):
            subregion = rest[len(prefix):].replace("_ratio", "")
            return {"modality": modality, "lobe": lobe, "subregion": subregion}

    if rest.startswith("global_"):
        subregion = rest.replace("global_", "").replace("_ratio", "")
        return {"modality": modality, "lobe": "global", "subregion": subregion}

    if rest == "tumor_burden_index":
        return {"modality": modality, "lobe": "global", "subregion": "tumor_burden"}

    return {"modality": modality, "lobe": "unknown", "subregion": rest}


def _to_native(obj):
    """Recursively convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return _to_native(obj.tolist())
    return obj


# ── Loading ──

def load_artifacts(
    model_path: Path, data_path: Path,
) -> tuple[xgb.XGBClassifier, pd.Series, pd.Series, pd.DataFrame]:
    """Load the trained model and processed dataset; separate metadata and features."""
    model: xgb.XGBClassifier = joblib.load(model_path)
    df = pd.read_csv(data_path)

    patient_ids = df["patient_id"].astype(str)
    y = df[TARGET_COL].astype(int)

    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    model_features = model.get_booster().feature_names
    assert model_features is not None, "Model has no stored feature names."
    _verify_integrity(X, model_features)

    return model, patient_ids, y, X


# ── SHAP Computation ──

def compute_shap(
    model: xgb.XGBClassifier, X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, shap.TreeExplainer]:
    """Compute SHAP values using a TreeExplainer.

    Positive SHAP values push the prediction toward high-risk (class 1);
    negative SHAP values push the prediction toward low-risk (class 0).
    The magnitude |SHAP| reflects the strength of that feature's contribution
    for a given patient.
    """
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)

    # For binary classification, shap_values may be a list of two arrays.
    # Select the array corresponding to the high-risk class (index 1).
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    # Base (expected) value for the high-risk class
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[1]

    return shap_vals, expected_value, explainer


# ── CSV Export: Feature Importance ──

def build_importance_df(
    feature_names: list[str],
    xgb_importance: np.ndarray,
    mean_abs_shap: np.ndarray,
) -> pd.DataFrame:
    """Build a structured importance table with parsed feature components."""
    rows = []
    for name, xgb_imp, mas in zip(feature_names, xgb_importance, mean_abs_shap):
        parsed = parse_feature_name(name)
        rows.append({
            "feature_name": name,
            "xgb_importance": float(xgb_imp),
            "mean_abs_shap": float(mas),
            "modality": parsed["modality"],
            "lobe": parsed["lobe"],
            "subregion": parsed["subregion"],
        })
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)


# ── Grouped Summaries ──

def grouped_importance(
    imp_df: pd.DataFrame, group_col: str,
) -> pd.DataFrame:
    """Aggregate mean absolute SHAP by *group_col* (modality, lobe, or subregion).

    Higher values indicate that, on average, features belonging to that group
    have a larger impact on the model's survival risk predictions.
    """
    grouped = (
        imp_df.groupby(group_col)["mean_abs_shap"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    grouped.columns = [group_col, "mean_abs_shap"]
    return grouped


# ── Plotting ──

def plot_shap_summary(
    shap_vals: np.ndarray, X: pd.DataFrame, save_path: Path,
) -> None:
    """SHAP beeswarm summary plot.

    Each point represents a single patient's SHAP value for a feature.
    Color encodes the feature value (red = high, blue = low).
    Features are sorted by mean absolute SHAP.
    """
    shap.summary_plot(shap_vals, X, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_shap_bar(
    shap_vals: np.ndarray, X: pd.DataFrame, save_path: Path,
) -> None:
    """Mean absolute SHAP bar chart.

    Bars show the average magnitude of each feature's contribution across
    all patients, regardless of direction (positive or negative).
    """
    shap.summary_plot(shap_vals, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_top20_xgb_importance(
    imp_df: pd.DataFrame, save_path: Path,
) -> None:
    """Horizontal bar plot of the top-20 features by XGBoost gain importance."""
    top20 = imp_df.head(20).sort_values("xgb_importance")

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(top20)), top20["xgb_importance"], color="steelblue")
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["feature_name"], fontsize=8)
    ax.set_xlabel("XGBoost Gain Importance")
    ax.set_title("Top-20 Features by XGBoost Gain Importance")
    ax.invert_yaxis()

    for bar, val in zip(bars, top20["xgb_importance"]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ── Top-K Reporting ──

def print_top_k_by_direction(
    imp_df: pd.DataFrame, shap_vals: np.ndarray, feature_names: list[str], k: int = 10,
) -> None:
    """Print features with the largest positive (high-risk) and negative (low-risk) mean SHAP."""
    mean_shap = pd.Series(np.mean(shap_vals, axis=0), index=feature_names)

    high_risk = mean_shap.nlargest(k)
    low_risk = mean_shap.nsmallest(k)

    print(f"Top {k} high-risk contributing features (positive SHAP -> high risk):")
    for name, val in high_risk.items():
        print(f"  {name:45s}  {val:+.6f}")
    print()

    print(f"Top {k} low-risk contributing features (negative SHAP -> low risk):")
    for name, val in low_risk.items():
        print(f"  {name:45s}  {val:+.6f}")
    print()


# ── Saving Utilities ──

def save_grouped_summaries(imp_df: pd.DataFrame, output_dir: Path) -> None:
    """Compute and save grouped importance summaries by modality, lobe, and subregion."""
    for group_col, filename in [
        ("modality", "modality_importance.csv"),
        ("lobe", "lobe_importance.csv"),
        ("subregion", "subregion_importance.csv"),
    ]:
        grouped = grouped_importance(imp_df, group_col)
        grouped.to_csv(output_dir / filename, index=False)
        print(f"Wrote {output_dir / filename}")


# ── Pipeline ──

def analyze_feature_importance(
    model_path: Path, data_path: Path, output_dir: Path,
) -> int:
    """Full importance analysis pipeline: load, compute, plot, export."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    model, patient_ids, y, X = load_artifacts(model_path, data_path)
    feature_names = list(X.columns)
    print(f"Loaded model and dataset: {len(patient_ids)} patients, {len(feature_names)} features")

    # ── SHAP values ──
    #
    # SHAP (SHapley Additive exPlanations) decomposes each prediction into
    # additive feature contributions that sum to the model output.
    #
    # Interpretation:
    #   Positive SHAP -> pushes prediction toward high-risk (class 1 / shorter survival)
    #   Negative SHAP -> pushes prediction toward low-risk (class 0 / longer survival)
    #
    # By examining which modalities, lobes, and subregions consistently show
    # large |SHAP| values, we can identify the spatial patterns that the model
    # relies on for survival risk stratification.
    shap_vals, expected_value, _explainer = compute_shap(model, X)
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

    # ── XGBoost gain-based importance ──
    xgb_importance = model.feature_importances_

    # ── Build structured table ──
    imp_df = build_importance_df(feature_names, xgb_importance, mean_abs_shap)
    imp_df.to_csv(output_dir / "feature_importance.csv", index=False)
    print(f"Wrote {output_dir / 'feature_importance.csv'}")

    # ── Plots ──
    plot_shap_summary(shap_vals, X, output_dir / "shap_summary.png")
    print(f"Wrote {output_dir / 'shap_summary.png'}")

    plot_shap_bar(shap_vals, X, output_dir / "shap_bar.png")
    print(f"Wrote {output_dir / 'shap_bar.png'}")

    plot_top20_xgb_importance(imp_df, output_dir / "top20_features.png")
    print(f"Wrote {output_dir / 'top20_features.png'}")

    # ── Grouped summaries ──
    save_grouped_summaries(imp_df, output_dir)

    # ── Console output ──
    print()
    print_top_k_by_direction(imp_df, shap_vals, feature_names, k=10)

    y_pred = model.predict(X)
    print(f"Model accuracy on full dataset: {(y_pred == y).mean():.4f}")
    print()

    # Print grouped summaries
    for group_col, title in [
        ("modality", "Mean |SHAP| by modality"),
        ("lobe", "Mean |SHAP| by lobe"),
        ("subregion", "Mean |SHAP| by subregion"),
    ]:
        grouped = grouped_importance(imp_df, group_col)
        print(f"{title}:")
        for _, row in grouped.iterrows():
            print(f"  {row[group_col]:15s}  {row['mean_abs_shap']:.6f}")
        print()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze multimodal lobe-subregion feature importance for survival risk.",
    )
    parser.add_argument(
        "--model", default=None,
        help=f"Trained model pickle (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--data", default=None,
        help=f"Processed dataset CSV (default: {DEFAULT_DATA})",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]

    model_path = _resolve_path(root, args.model or DEFAULT_MODEL)
    data_path = _resolve_path(root, args.data or DEFAULT_DATA)

    for p, label in [(model_path, "Model"), (data_path, "Data")]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    output_dir = _resolve_path(root, args.output_dir or DEFAULT_OUTPUT_DIR)

    return analyze_feature_importance(model_path, data_path, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
