#!/usr/bin/env python3

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "outputs" / "multimodal_lobewise_svm" / "svm_model.pkl"
DATA_PATH = ROOT / "outputs" / "multimodal_lobewise" / "merged_features.csv"
OUTPUT_DIR = ROOT / "outputs" / "shap_stability"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 100
BACKGROUND_SIZE = 50
N_EXPLAIN = 50
RANDOM_STATE = 42
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


def load_artifacts() -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    model: Pipeline = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    y = df[TARGET_COL].astype(int)
    feature_cols = [c for c in df.columns
                    if c not in METADATA_COLS and c not in DROP_COLS]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    model_features: list[str] = model.feature_names_in_.tolist()
    assert list(X.columns) == model_features
    return model, X, y


def single_bootstrap_shap(
    model: Pipeline, X: pd.DataFrame, y: pd.Series,
    seed: int,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    bg_size = min(BACKGROUND_SIZE, X_train.shape[0])
    bg_idx = rng.choice(X_train.shape[0], size=bg_size, replace=False)
    X_background = X_train.iloc[bg_idx]

    n_explain = min(N_EXPLAIN, X.shape[0])
    explain_idx = rng.choice(X.shape[0], size=n_explain, replace=False)
    X_explain = X.iloc[explain_idx]

    n_features = X.shape[1]
    nsamples = min(200, 2 * n_features + 2048)

    predict_fn = lambda x: model.predict_proba(x)[:, 1]
    explainer = shap.KernelExplainer(predict_fn, X_background.values)
    shap_vals = explainer.shap_values(X_explain.values, nsamples=nsamples, silent=True)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
    assert not np.any(np.isnan(shap_vals))
    return np.mean(np.abs(shap_vals), axis=0)


def main():
    print("=" * 60)
    print("  SHAP RANKING STABILITY — BOOTSTRAP EVALUATION")
    print("=" * 60)

    print("\n[1/4] Loading model and data ...")
    model, X, y = load_artifacts()
    feature_names: list[str] = model.feature_names_in_.tolist()
    n_features = len(feature_names)
    print(f"  Model: {type(model).__name__} | {n_features} features")
    print(f"  Dataset: {X.shape[0]} patients")

    print(f"\n[2/4] Running {N_BOOTSTRAP} bootstrap repetitions ...")
    print(f"  Background size: {BACKGROUND_SIZE}")
    print(f"  Explained instances: {N_EXPLAIN}")
    print(f"  nsamples: {min(200, 2 * n_features + 2048)}")

    all_importances = np.full((N_BOOTSTRAP, n_features), np.nan)
    all_ranks = np.full((N_BOOTSTRAP, n_features), np.nan)

    t0 = time.time()
    for b in range(N_BOOTSTRAP):
        seed = RANDOM_STATE + b * 1000
        mean_abs = single_bootstrap_shap(model, X, y, seed=seed)
        all_importances[b, :] = mean_abs
        all_ranks[b, :] = np.argsort(np.argsort(-mean_abs))
        elapsed = time.time() - t0
        avg_per_rep = elapsed / (b + 1)
        remaining = avg_per_rep * (N_BOOTSTRAP - b - 1)
        print(f"  [{b + 1:3d}/{N_BOOTSTRAP}] "
              f"elapsed={elapsed:.0f}s "
              f"eta={remaining:.0f}s")

    total_time = time.time() - t0
    print(f"\n  Total time: {total_time:.0f}s ({total_time / 60:.1f} min)")

    print("\n[3/4] Computing stability metrics ...")
    mean_imp = np.nanmean(all_importances, axis=0)
    std_imp = np.nanstd(all_importances, axis=0)
    cv_imp = np.where(mean_imp > 0, std_imp / mean_imp, np.nan)

    rank_freq_top5 = np.zeros(n_features, dtype=float)
    rank_freq_top10 = np.zeros(n_features, dtype=float)
    for b in range(N_BOOTSTRAP):
        imp = all_importances[b, :]
        top5_idx = np.argsort(-imp)[:5]
        top10_idx = np.argsort(-imp)[:10]
        rank_freq_top5[top5_idx] += 1.0
        rank_freq_top10[top10_idx] += 1.0
    rank_freq_top5 /= N_BOOTSTRAP
    rank_freq_top10 /= N_BOOTSTRAP

    # ── CSV ──
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

    stability_df = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
    csv_path = OUTPUT_DIR / "shap_rank_stability.csv"
    stability_df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path.name}")

    # ── JSON ──
    json_data = {
        "metadata": {
            "n_bootstrap": N_BOOTSTRAP,
            "background_size": BACKGROUND_SIZE,
            "n_explain": N_EXPLAIN,
            "n_features": n_features,
            "total_time_seconds": total_time,
        },
        "features": [],
    }
    for _, r in stability_df.iterrows():
        json_data["features"].append(r.to_dict())
    json_path = OUTPUT_DIR / "shap_rank_stability.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"  Saved {json_path.name}")

    # ── Figures ──
    print("\n[4/4] Generating figures ...")

    # Sort by mean importance for plotting
    sort_idx = np.argsort(mean_imp)
    sorted_names = [feature_names[i] for i in sort_idx]
    sorted_imps = all_importances[:, sort_idx]

    # 1. Rank distribution (boxplot)
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.boxplot(
        [all_importances[:, i] for i in sort_idx],
        vert=False, labels=sorted_names, patch_artist=True,
        boxprops=dict(facecolor="#2166AC", alpha=0.6),
        medianprops=dict(color="black", lw=1.5),
        whis=(2.5, 97.5),
    )
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title("SHAP Importance Distribution Across Bootstrap Repetitions")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "shap_rank_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved shap_rank_distribution.png")

    # 2. Top-10 stability heatmap
    top10_names = [r["feature_name"] for _, r in stability_df.head(10).iterrows()]
    top10_idx = [feature_names.index(n) for n in top10_names]
    rank_matrix = all_ranks[:, top10_idx] + 1

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(rank_matrix.T, aspect="auto", cmap="viridis_r",
                   vmin=1, vmax=n_features)
    ax.set_yticks(range(len(top10_names)))
    ax.set_yticklabels(top10_names, fontsize=9)
    ax.set_xlabel("Bootstrap Repetition")
    ax.set_title("SHAP Rank of Top-10 Features Across Bootstrap Repetitions")
    cbar = fig.colorbar(im, ax=ax, label="Rank (1 = most important)")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "shap_top10_stability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved shap_top10_stability.png")

    # ── Report ──
    top10 = stability_df.head(10)
    t1gd_temporal_en = stability_df[stability_df["feature_name"] == "T1GD_temporal_en_ratio"]
    t1gd_frontal_en = stability_df[stability_df["feature_name"] == "T1GD_frontal_en_ratio"]

    t1gd_temporal_en_top5 = t1gd_temporal_en["freq_top5_pct"].values[0] if len(t1gd_temporal_en) > 0 else 0.0
    t1gd_frontal_en_top5 = t1gd_frontal_en["freq_top5_pct"].values[0] if len(t1gd_frontal_en) > 0 else 0.0

    # Determine if conclusions are stable: top-10 composition consistent
    stable_features = top10[top10["freq_top10_pct"] >= 50]
    unstable_features = top10[top10["freq_top10_pct"] < 50]
    conclusions_stable = len(unstable_features) <= 2

    # Average CV of top 10
    top10_cv_mean = top10["cv_abs_shap"].mean()

    print(f"\n{'=' * 60}")
    print(f"  SHAP stability complete. Outputs in: {OUTPUT_DIR}")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
