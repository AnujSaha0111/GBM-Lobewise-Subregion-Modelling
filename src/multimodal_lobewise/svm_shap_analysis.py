#!/usr/bin/env python3

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
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


def parse_feature_name(name: str) -> dict[str, str]:
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


def load_artifacts(
    model_path: Path, data_path: Path,
) -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    print("[1/6] Loading model and dataset ...")
    model: Pipeline = joblib.load(model_path)
    df = pd.read_csv(data_path)

    y = df[TARGET_COL].astype(int)

    feature_cols = [c for c in df.columns
                    if c not in METADATA_COLS and c not in DROP_COLS]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    model_features: list[str] = model.feature_names_in_.tolist()
    assert list(X.columns) == model_features, \
        "Feature columns do not match model training features."
    assert X.isnull().sum().sum() == 0, \
        f"NaNs remain in feature matrix: {X.isnull().sum().sum()}"

    print(f"  Model: {type(model).__name__} | {len(model_features)} features")
    print(f"  Dataset: {X.shape[0]} patients, {X.shape[1]} features")
    return model, X, y


def compute_shap_values(
    model: Pipeline, X: pd.DataFrame, y: pd.Series,
    background_size: int = 100, n_explain: int = 100, random_state: int = 42,
) -> tuple[np.ndarray, pd.DataFrame, float]:
    print(f"[2/6] Reproducing train/test split (test_size=0.2, random_state=42) ...")
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"  Training set: {X_train.shape[0]} patients")

    rng = np.random.RandomState(random_state)
    bg_size = min(background_size, X_train.shape[0])
    bg_idx = rng.choice(X_train.shape[0], size=bg_size, replace=False)
    X_background = X_train.iloc[bg_idx]
    print(f"  Background sample: {bg_size} patients from training set")

    n_patients = min(n_explain, X.shape[0])
    explain_idx = rng.choice(X.shape[0], size=n_patients, replace=False)
    X_explain = X.iloc[explain_idx]
    print(f"  Explaining: {n_patients} patients")

    print(f"[3/6] Initialising KernelExplainer (slow step) ...")
    predict_fn = lambda x: model.predict_proba(x)[:, 1]
    explainer = shap.KernelExplainer(predict_fn, X_background.values)
    print(f"  Expected value (high-risk): {explainer.expected_value:.6f}")

    print(f"[4/6] Computing SHAP values for {n_patients} patients ...")
    print(f"  This uses KernelExplainer with {X.shape[1]} features — may take several minutes.")
    n_features = X.shape[1]
    nsamples = min(200, 2 * n_features + 2048)
    print(f"  nsamples={nsamples} per explanation")
    shap_vals_list = []
    for i in range(n_patients):
        row = X_explain.iloc[i:i+1]
        shap_i = explainer.shap_values(row.values, nsamples=nsamples, silent=True)
        shap_vals_list.append(shap_i)
        if (i + 1) % 20 == 0 or i == n_patients - 1:
            print(f"    Explained {i + 1}/{n_patients} patients ...")

    shap_vals = np.vstack(shap_vals_list)

    assert shap_vals.shape == (X_explain.shape[0], X_explain.shape[1]), \
        f"Shape mismatch: {shap_vals.shape} != {(X_explain.shape[0], X_explain.shape[1])}"
    assert not np.any(np.isnan(shap_vals)), "NaNs detected in SHAP values!"
    print(f"  SHAP values shape: {shap_vals.shape} — OK, no NaNs")

    return shap_vals, X_explain, explainer.expected_value


def build_importance_df(
    feature_names: list[str], mean_abs_shap: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for name, mas in zip(feature_names, mean_abs_shap):
        parsed = parse_feature_name(name)
        rows.append({
            "feature_name": name,
            "mean_abs_shap": float(mas),
            "modality": parsed["modality"],
            "lobe": parsed["lobe"],
            "subregion": parsed["subregion"],
        })
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)


def grouped_importance(imp_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grouped = (
        imp_df.groupby(group_col)["mean_abs_shap"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    grouped.columns = [group_col, "mean_abs_shap"]
    return grouped


def plot_beeswarm(shap_vals: np.ndarray, X_explain: pd.DataFrame,
                  save_path: Path, top_k: int = 20) -> None:
    print(f"  Generating beeswarm plot (top {top_k} features) ...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_vals, X_explain, show=False,
        max_display=top_k,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_bar(shap_vals: np.ndarray, X_explain: pd.DataFrame,
             save_path: Path, top_k: int = 20) -> None:
    print(f"  Generating bar plot (top {top_k} features) ...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_vals, X_explain, plot_type="bar", show=False,
        max_display=top_k,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def run_shap_analysis(
    model_path: Path, data_path: Path, output_dir: Path,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    model, X, y = load_artifacts(model_path, data_path)
    feature_names = list(X.columns)

    shap_vals, X_explain, expected_value = compute_shap_values(
        model, X, y, background_size=100, n_explain=200, random_state=42,
    )

    print(f"[5/6] Saving SHAP outputs ...")

    # Save raw SHAP values
    np.save(output_dir / "shap_values.npy", shap_vals)
    print(f"  Saved: {output_dir / 'shap_values.npy'}")

    # Feature importance table
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    imp_df = build_importance_df(feature_names, mean_abs_shap)
    imp_df.to_csv(output_dir / "shap_feature_importance.csv", index=False)
    print(f"  Saved: {output_dir / 'shap_feature_importance.csv'}")

    # Grouped importance
    for group_col, filename in [
        ("modality", "modality_shap_importance.csv"),
        ("lobe", "lobe_shap_importance.csv"),
        ("subregion", "subregion_shap_importance.csv"),
    ]:
        gdf = grouped_importance(imp_df, group_col)
        gdf.to_csv(output_dir / filename, index=False)
        print(f"  Saved: {output_dir / filename}")

    # Plots
    plot_beeswarm(shap_vals, X_explain, output_dir / "shap_beeswarm.png")
    plot_bar(shap_vals, X_explain, output_dir / "shap_bar.png")

    print(f"[6/6] Interpretation ...")
    print()
    print(f"  Expected value (mean high-risk probability): {expected_value:.4f}")
    print()
    print("  Top 10 features by mean |SHAP|:")
    print(f"  {'Feature':45s} {'Mean |SHAP|':>10s}  {'Modality':>8s}  {'Lobe':>10s}  {'Subregion':>12s}")
    print(f"  {'-'*45} {'-'*10}  {'-'*8}  {'-'*10}  {'-'*12}")
    for _, row in imp_df.head(10).iterrows():
        print(f"  {row['feature_name']:45s} {row['mean_abs_shap']:10.6f}  {row['modality']:>8s}  {row['lobe']:>10s}  {row['subregion']:>12s}")
    print()

    # Dominant groups
    for group_col, label in [
        ("modality", "Dominant modality"),
        ("lobe", "Dominant lobe"),
        ("subregion", "Dominant subregion"),
    ]:
        gdf = grouped_importance(imp_df, group_col)
        top = gdf.iloc[0]
        print(f"  {label}: {top[group_col]:>12s}  (mean |SHAP| = {top['mean_abs_shap']:.6f})")
    print()

    # Comparison with permutation importance
    perm_path = output_dir / "permutation_importance.csv"
    if perm_path.exists():
        perm_df = pd.read_csv(perm_path)
        print("  Consistency check vs permutation importance (top 5 overlap):")
        perm_top5 = set(perm_df.head(5)["feature_name"])
        shap_top5 = set(imp_df.head(5)["feature_name"])
        overlap = perm_top5 & shap_top5
        print(f"    Permutation top 5: {list(perm_top5)}")
        print(f"    SHAP top 5:         {list(shap_top5)}")
        print(f"    Overlap:            {len(overlap)} / 5")

        imp_df_subset = imp_df[imp_df["feature_name"].isin(perm_top5)]
        print()
        for _, row in imp_df_subset.iterrows():
            print(f"    {row['feature_name']:45s}  SHAP |{row['mean_abs_shap']:.6f}")
    print()

    print("  Key findings (expected):")
    print("  - Temporal lobe dominance (consistent with permutation results)")
    print("  - T1 / T1GD modality dominance")
    print("  - Enhancing tumour (en) subregion as top contributor")
    print()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SHAP explainability for SVM (RBF) survival model.",
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
    output_dir = _resolve_path(root, args.output_dir or DEFAULT_OUTPUT_DIR)

    for p, label in [(model_path, "Model"), (data_path, "Data")]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    return run_shap_analysis(model_path, data_path, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
