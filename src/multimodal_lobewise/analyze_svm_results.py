#!/usr/bin/env python3
"""Permutation-importance analysis for the SVM multimodal survival classifier.

Because RBF-kernel SVM is not directly interpretable, this script uses
permutation feature importance (model-agnostic) to identify which multimodal
lobewise-subregion features contribute most strongly. Results are grouped by
modality, lobe, and subregion, and saved under outputs/multimodal_lobewise_svm/.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

METADATA_COLS = {"patient_id", "risk_label", "OS_months",
                 "lobe_assignment_reliable"}
DROP_COLS = {"OS_months", "lobe_assignment_reliable"}
TARGET_COL = "risk_label"
LOBES = ["frontal", "temporal", "parietal", "occipital"]

DEFAULT_MODEL = "outputs/multimodal_lobewise_svm/svm_model.pkl"
DEFAULT_DATA = "outputs/multimodal_lobewise/merged_features.csv"
DEFAULT_OUTPUT_DIR = "outputs/multimodal_lobewise_svm"


def _resolve_path(root: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (root / p)


def _verify_integrity(X: pd.DataFrame, model_features: list[str]) -> None:
    assert X.columns.is_unique, \
        f"Duplicate columns: {X.columns[X.columns.duplicated()].tolist()}"
    non_numeric = X.select_dtypes(exclude=["number"]).columns.tolist()
    assert len(non_numeric) == 0, \
        f"Non-numeric feature column(s): {non_numeric}"
    leaked = X.columns.intersection(METADATA_COLS)
    assert leaked.empty, f"Metadata leakage: {leaked.tolist()}"
    assert X.isnull().sum().sum() == 0, f"NaNs remain: {X.isnull().sum().sum()}"
    assert list(X.columns) == model_features, \
        "Feature columns do not match model training features."


def parse_feature_name(name: str) -> dict[str, str]:
    modality, rest = name.split("_", 1)
    for lobe in LOBES:
        prefix = f"{lobe}_"
        if rest.startswith(prefix):
            subregion = rest[len(prefix):].replace("_ratio", "")
            return {"modality": modality, "lobe": lobe,
                    "subregion": subregion}
    if rest.startswith("global_"):
        subregion = rest.replace("global_", "").replace("_ratio", "")
        return {"modality": modality, "lobe": "global",
                "subregion": subregion}
    if rest == "tumor_burden_index":
        return {"modality": modality, "lobe": "global",
                "subregion": "tumor_burden"}
    return {"modality": modality, "lobe": "unknown", "subregion": rest}


def load_artifacts(
    model_path: Path, data_path: Path,
) -> tuple[Pipeline, pd.Series, pd.Series, pd.DataFrame]:
    model: Pipeline = joblib.load(model_path)
    df = pd.read_csv(data_path)
    patient_ids = df["patient_id"].astype(str)
    y = df[TARGET_COL].astype(int)
    feature_cols = [c for c in df.columns
                    if c not in METADATA_COLS and c not in DROP_COLS]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    model_features: list[str] = model.feature_names_in_.tolist()
    _verify_integrity(X, model_features)
    return model, patient_ids, y, X


def compute_permutation_importance(
    model: Pipeline, X: pd.DataFrame, y: pd.Series,
) -> pd.DataFrame:
    result = permutation_importance(
        model, X, y, scoring="roc_auc", n_repeats=30,
        random_state=42, n_jobs=-1,
    )
    rows = []
    for i, name in enumerate(X.columns):
        parsed = parse_feature_name(name)
        rows.append({
            "feature_name": name,
            "importance_mean": float(result.importances_mean[i]),
            "importance_std": float(result.importances_std[i]),
            "modality": parsed["modality"],
            "lobe": parsed["lobe"],
            "subregion": parsed["subregion"],
        })
    return pd.DataFrame(rows).sort_values("importance_mean",
                                           ascending=False)


def grouped_importance(imp_df: pd.DataFrame,
                       group_col: str) -> pd.DataFrame:
    g = (imp_df.groupby(group_col)["importance_mean"]
         .mean().sort_values(ascending=False).reset_index())
    g.columns = [group_col, "importance_mean"]
    return g


def plot_top20_importance(imp_df: pd.DataFrame, save_path: Path) -> None:
    top20 = imp_df.head(20).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top20)))
    bars = ax.barh(range(len(top20)), top20["importance_mean"],
                   color=colors)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["feature_name"], fontsize=8)
    ax.set_xlabel("Permutation Importance (mean AUC drop)")
    ax.set_title("Top-20 Features by Permutation Importance — SVM (RBF)")
    ax.invert_yaxis()
    for bar, val in zip(bars, top20["importance_mean"]):
        ax.text(bar.get_width() + 0.0002,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def save_grouped_summaries(imp_df: pd.DataFrame, output_dir: Path) -> None:
    for group_col, filename in [
        ("modality", "modality_importance.csv"),
        ("lobe", "lobe_importance.csv"),
        ("subregion", "subregion_importance.csv"),
    ]:
        gdf = grouped_importance(imp_df, group_col)
        gdf.to_csv(output_dir / filename, index=False)
        print(f"Wrote {output_dir / filename}")


def analyze_svm_results(model_path: Path, data_path: Path,
                        output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    model, patient_ids, y, X = load_artifacts(model_path, data_path)
    feature_names = list(X.columns)
    print(f"Loaded model and dataset: {len(patient_ids)} patients, "
          f"{len(feature_names)} features")

    imp_df = compute_permutation_importance(model, X, y)
    imp_df.to_csv(output_dir / "permutation_importance.csv", index=False)
    print(f"Wrote {output_dir / 'permutation_importance.csv'}")

    plot_top20_importance(imp_df,
                          output_dir / "top20_permutation_features.png")
    print(f"Wrote {output_dir / 'top20_permutation_features.png'}")

    save_grouped_summaries(imp_df, output_dir)

    print("\nTop 10 high-contributing features:")
    for _, row in imp_df.head(10).iterrows():
        print(f"  {row['feature_name']:45s}  "
              f"{row['importance_mean']:.6f} ± {row['importance_std']:.6f}")
    print()

    for group_col, title in [
        ("modality", "\nMean permutation importance by modality:"),
        ("lobe", "\nMean permutation importance by lobe:"),
        ("subregion", "\nMean permutation importance by subregion:"),
    ]:
        print(title)
        gdf = grouped_importance(imp_df, group_col)
        for _, r in gdf.iterrows():
            print(f"  {r[group_col]:15s}  {r['importance_mean']:.6f}")
    print()

    dominant_mod = grouped_importance(imp_df, "modality").iloc[0]
    dominant_lobe = grouped_importance(imp_df, "lobe").iloc[0]
    print(f"Dominant modality: {dominant_mod['modality']} "
          f"(importance = {dominant_mod['importance_mean']:.6f})")
    print(f"Dominant lobe:     {dominant_lobe['lobe']} "
          f"(importance = {dominant_lobe['importance_mean']:.6f})")
    print()

    print("Comparison with XGBoost:")
    print("  XGBoost uses tree-based splits and captures non-linear")
    print("  interactions among modalities, lobes, and subregions.")
    print("  SVM with RBF kernel captures non-linear decision boundaries")
    print("  through a different inductive bias (max-margin separation")
    print("  in a transformed feature space via the kernel trick).")
    print("  If both models rely on similar top features, it suggests")
    print("  those spatial patterns are robustly informative. If top")
    print("  features diverge, it may reflect each algorithm exploiting")
    print("  different statistical regularities in the data.")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Permutation importance analysis for SVM survival model.",
    )
    parser.add_argument("--model", default=None,
                        help=f"Model pickle (default: {DEFAULT_MODEL})")
    parser.add_argument("--data", default=None,
                        help=f"Dataset CSV (default: {DEFAULT_DATA})")
    parser.add_argument("--output-dir", default=None,
                        help=f"Output dir (default: {DEFAULT_OUTPUT_DIR})")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    model_path = _resolve_path(root, args.model or DEFAULT_MODEL)
    data_path = _resolve_path(root, args.data or DEFAULT_DATA)
    for p, label in [(model_path, "Model"), (data_path, "Data")]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")
    output_dir = _resolve_path(root, args.output_dir or DEFAULT_OUTPUT_DIR)
    return analyze_svm_results(model_path, data_path, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
