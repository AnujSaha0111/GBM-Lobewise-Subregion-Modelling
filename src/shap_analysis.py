#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "outputs" / "multimodal_lobewise" / "model.pkl"
DATA_PATH = ROOT / "outputs" / "multimodal_lobewise" / "merged_features.csv"

TARGET_COL = "risk_label"
METADATA_COLS = {"patient_id", "risk_label", "OS_months",
                 "lobe_assignment_reliable"}
DROP_COLS = {"OS_months", "lobe_assignment_reliable"}
LOBES = ["frontal", "temporal", "parietal", "occipital"]


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


def load_artifacts() -> tuple:
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    y = df[TARGET_COL].astype(int)
    feature_cols = [c for c in df.columns
                    if c not in METADATA_COLS and c not in DROP_COLS]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    model_features: list[str] = model.feature_names_in_.tolist()
    assert list(X.columns) == model_features
    return model, X, y


def compute_stability_metrics(
    all_importances: np.ndarray,
    feature_names: list[str],
    n_bootstrap: int,
) -> pd.DataFrame:
    n_features = len(feature_names)
    mean_imp = np.nanmean(all_importances, axis=0)
    std_imp = np.nanstd(all_importances, axis=0)
    cv_imp = np.where(mean_imp > 0, std_imp / mean_imp, np.nan)

    all_ranks = np.argsort(np.argsort(-all_importances, axis=1), axis=1)

    rank_freq_top5 = np.zeros(n_features, dtype=float)
    rank_freq_top10 = np.zeros(n_features, dtype=float)
    for b in range(n_bootstrap):
        imp = all_importances[b, :]
        top5_idx = np.argsort(-imp)[:5]
        top10_idx = np.argsort(-imp)[:10]
        rank_freq_top5[top5_idx] += 1.0
        rank_freq_top10[top10_idx] += 1.0
    rank_freq_top5 /= n_bootstrap
    rank_freq_top10 /= n_bootstrap

    rows = []
    for i, name in enumerate(feature_names):
        parsed = parse_feature_name(name)
        rows.append({
            "feature_name": name,
            "modality": parsed["modality"],
            "lobe": parsed["lobe"],
            "subregion": parsed["subregion"],
            "mean_abs_shap": mean_imp[i],
            "std_abs_shap": std_imp[i],
            "cv_abs_shap": cv_imp[i],
            "median_rank": float(np.median(all_ranks[:, i]) + 1),
            "rank_iqr": float(np.percentile(all_ranks[:, i], 75) -
                               np.percentile(all_ranks[:, i], 25)),
            "freq_top5_pct": round(rank_freq_top5[i] * 100, 2),
            "freq_top10_pct": round(rank_freq_top10[i] * 100, 2),
        })

    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
