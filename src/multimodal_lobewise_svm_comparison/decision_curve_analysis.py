#!/usr/bin/env python3
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ──
ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = ROOT / "outputs" / "multimodal_lobewise" / "merged_features_with_metadata.csv"
OUTPUT_DIR = ROOT / "outputs" / "multimodal_lobewise_svm_comparison"
DCA_DIR = OUTPUT_DIR / "dca"
DCA_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──
RANDOM_STATE = 42
TEST_SIZE = 0.20

THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)

SPATIAL_FEATURE_PREFIXES = ("T1_", "T2_", "T1GD_", "FLAIR_")
METADATA_FEATURE_COLS = ["age", "sex", "idh", "mgmt", "who_grade", "eor"]

FEATURE_SETS = {
    "spatial": {
        "name": "Spatial Only",
        "prefix": "spatial",
        "best_params": {"svm__C": 100, "svm__gamma": 0.001},
        "color": "#2166AC",
    },
    "clinical": {
        "name": "Clinical Only",
        "prefix": "clinical",
        "best_params": {"svm__C": 1, "svm__gamma": 0.001},
        "color": "#4DAF4A",
    },
    "combined": {
        "name": "Combined",
        "prefix": "combined",
        "best_params": {"svm__C": 1, "svm__gamma": "scale"},
        "color": "#E41A1C",
    },
}

TREAT_ALL_COLOR = "gray"
TREAT_NONE_COLOR = "black"

IEEE_WIDTH_INCHES = 6.5   # single column IEEE
IEEE_HEIGHT_INCHES = 5.0

# ══════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════

def load_and_split():
    df = pd.read_csv(INPUT_CSV)
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    y = df["risk_label"].astype(int)

    spatial_cols = [c for c in df.columns
                    if c.startswith(SPATIAL_FEATURE_PREFIXES)]
    meta_cols = [c for c in METADATA_FEATURE_COLS if c in df.columns]
    combined_cols = spatial_cols + meta_cols

    feature_map = {
        "spatial": spatial_cols,
        "clinical": meta_cols,
        "combined": combined_cols,
    }

    X_train_dict = {}
    X_test_dict = {}
    y_train_dict = {}
    y_test_dict = {}
    y_test_vals = None

    for key in feature_map:
        cols = feature_map[key]
        X = df[cols].apply(pd.to_numeric, errors="coerce")

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        X_train_dict[key] = X_tr
        X_test_dict[key] = X_te
        y_train_dict[key] = y_tr
        y_test_dict[key] = y_te
        if y_test_vals is None:
            y_test_vals = y_te.values

    return feature_map, X_train_dict, X_test_dict, y_train_dict, y_test_dict, y_test_vals


def fit_and_predict():
    """Fit three pipelines with known best params, return test-set probabilities."""
    feature_map, X_tr, X_te, y_tr, y_te, y_test_vals = load_and_split()

    probs = {}
    for key in ["spatial", "clinical", "combined"]:
        info = FEATURE_SETS[key]
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True,
                        class_weight="balanced", random_state=RANDOM_STATE)),
        ])
        pipeline.set_params(**info["best_params"])
        pipeline.fit(X_tr[key], y_tr[key])
        prob = pipeline.predict_proba(X_te[key])[:, 1]
        probs[key] = prob

    return probs, y_test_vals, X_te, y_te


# ══════════════════════════════════════════════════════
#  NET BENEFIT
# ══════════════════════════════════════════════════════

def net_benefit(y_true, y_prob, thresholds):
    n = len(y_true)
    nb = np.full(len(thresholds), np.nan)
    for i, pt in enumerate(thresholds):
        pred = (y_prob >= pt).astype(int)
        tp = np.sum((y_true == 1) & (pred == 1))
        fp = np.sum((y_true == 0) & (pred == 1))
        nb[i] = (tp / n) - (fp / n) * (pt / (1.0 - pt))
    return nb


def treat_all_nb(y_true, thresholds):
    n = len(y_true)
    tp = np.sum(y_true == 1)
    fp = np.sum(y_true == 0)
    nb = np.full(len(thresholds), np.nan)
    for i, pt in enumerate(thresholds):
        nb[i] = (tp / n) - (fp / n) * (pt / (1.0 - pt))
    return nb


# ══════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════

def set_ieee_style(ax):
    ax.tick_params(direction="in", which="both", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.set_xlabel("Threshold Probability", fontsize=10)
    ax.set_ylabel("Net Benefit", fontsize=10)


def save_figure(fig, stem, dpi=300):
    for ext, kwargs in [
        ("png", {"dpi": dpi, "bbox_inches": "tight"}),
        ("png", {"dpi": 600, "bbox_inches": "tight"}),
        ("svg", {"bbox_inches": "tight"}),
        ("pdf", {"bbox_inches": "tight"}),
    ]:
        path = DCA_DIR / f"{stem}.{ext}"
        if ext == "png" and kwargs.get("dpi") == 600:
            path = DCA_DIR / f"{stem}_600.png"
        fig.savefig(path, format=ext, **kwargs)


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  DECISION CURVE ANALYSIS")
    print("=" * 60)

    probs, y_test_vals, X_te, y_te = fit_and_predict()
    n_test = len(y_test_vals)
    print(f"\nTest set size: {n_test}")
    print(f"  High-risk: {y_test_vals.sum()} / {n_test}")

    # ── Compute net benefits ──
    all_nb = {}
    for key in ["spatial", "clinical", "combined"]:
        all_nb[key] = net_benefit(y_test_vals, probs[key], THRESHOLDS)
    all_nb["treat_all"] = treat_all_nb(y_test_vals, THRESHOLDS)

    # ── Decision curve data CSV ──
    rows = []
    for i, pt in enumerate(THRESHOLDS):
        row = {
            "threshold": round(pt, 4),
            "treat_none": 0.0,
            "treat_all": round(all_nb["treat_all"][i], 6),
        }
        for key in ["spatial", "clinical", "combined"]:
            row[FEATURE_SETS[key]["name"]] = round(all_nb[key][i], 6)
        rows.append(row)

    data_df = pd.DataFrame(rows)
    cols_order = ["threshold", "treat_none", "treat_all"] + [FEATURE_SETS[k]["name"] for k in ["spatial", "clinical", "combined"]]
    data_df = data_df[cols_order]
    data_path = DCA_DIR / "decision_curve_data.csv"
    data_df.to_csv(data_path, index=False)
    print(f"\nSaved {data_path.name}  ({len(rows)} thresholds)")

    # ── Summary CSV ──
    summary_rows = []
    for key in ["spatial", "clinical", "combined"]:
        nb = all_nb[key]
        pos_mask = nb > 0
        pos_ranges = []
        if pos_mask.any():
            idxs = np.where(pos_mask)[0]
            start = THRESHOLDS[idxs[0]]
            end = THRESHOLDS[idxs[-1]]
            pos_ranges.append(f"{start:.2f} - {end:.2f}")
        max_idx = np.argmax(nb)
        summary_rows.append({
            "Model": FEATURE_SETS[key]["name"],
            "Threshold range with positive NB": "; ".join(pos_ranges) if pos_ranges else "None",
            "Maximum NB": f"{nb[max_idx]:.6f}",
            "Threshold at max NB": f"{THRESHOLDS[max_idx]:.2f}",
        })

    # Also for treat_all
    nb_ta = all_nb["treat_all"]
    pos_mask_ta = nb_ta > 0
    pos_ranges_ta = []
    if pos_mask_ta.any():
        idxs = np.where(pos_mask_ta)[0]
        pos_ranges_ta.append(f"{THRESHOLDS[idxs[0]]:.2f} - {THRESHOLDS[idxs[-1]]:.2f}")
    max_idx_ta = np.argmax(nb_ta)
    summary_rows.append({
        "Model": "Treat All",
        "Threshold range with positive NB": "; ".join(pos_ranges_ta) if pos_ranges_ta else "None",
        "Maximum NB": f"{nb_ta[max_idx_ta]:.6f}",
        "Threshold at max NB": f"{THRESHOLDS[max_idx_ta]:.2f}",
    })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = DCA_DIR / "decision_curve_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved {summary_path.name}")
    print(summary_df.to_string(index=False))

    # ── Metrics JSON ──
    metrics = {}
    for key in ["spatial", "clinical", "combined"]:
        nb = all_nb[key]
        pos_mask = nb > 0
        pos_thresholds = THRESHOLDS[pos_mask]
        metrics[key] = {
            "model": FEATURE_SETS[key]["name"],
            "n_test": int(n_test),
            "n_high_risk": int(y_test_vals.sum()),
            "max_net_benefit": float(nb.max()),
            "threshold_at_max_nb": float(THRESHOLDS[np.argmax(nb)]),
            "positive_nb_min_threshold": float(pos_thresholds.min()) if len(pos_thresholds) else None,
            "positive_nb_max_threshold": float(pos_thresholds.max()) if len(pos_thresholds) else None,
            "net_benefit_at_01": float(nb[0]),
            "net_benefit_at_05": float(nb[THRESHOLDS == 0.05][0]) if 0.05 in THRESHOLDS else None,
            "net_benefit_at_10": float(nb[THRESHOLDS == 0.10][0]) if 0.10 in THRESHOLDS else None,
            "net_benefit_at_20": float(nb[THRESHOLDS == 0.20][0]) if 0.20 in THRESHOLDS else None,
            "net_benefit_at_30": float(nb[THRESHOLDS == 0.30][0]) if 0.30 in THRESHOLDS else None,
            "threshold_range_positive_nb": (
                f"[{pos_thresholds.min():.2f}, {pos_thresholds.max():.2f}]"
                if len(pos_thresholds) else "none"
            ),
        }
    metrics_path = DCA_DIR / "decision_curve_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved {metrics_path.name}")

    # ── Figures ──
    print("\n  Generating figures ...")

    # Full range
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH_INCHES, IEEE_HEIGHT_INCHES))

    ax.plot(THRESHOLDS, all_nb["treat_all"],
            color=TREAT_ALL_COLOR, lw=1.5, linestyle="--", label="Treat All")
    ax.plot(THRESHOLDS, np.zeros_like(THRESHOLDS),
            color=TREAT_NONE_COLOR, lw=1.5, linestyle=":", label="Treat None")

    for key in ["spatial", "clinical", "combined"]:
        info = FEATURE_SETS[key]
        ax.plot(THRESHOLDS, all_nb[key],
                color=info["color"], lw=2, label=info["name"])

    set_ieee_style(ax)
    ax.set_xlim(-0.01, 1.01)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.set_title("Decision Curve Analysis", fontsize=11, fontweight="bold")

    plt.tight_layout()
    save_figure(fig, "decision_curve_analysis")
    plt.close(fig)
    print("  Saved decision_curve_analysis (PNG 300/600, SVG, PDF)")

    # Zoomed 0.10 - 0.50
    zoom_mask = (THRESHOLDS >= 0.10) & (THRESHOLDS <= 0.50)
    zoom_pt = THRESHOLDS[zoom_mask]

    fig, ax = plt.subplots(figsize=(IEEE_WIDTH_INCHES, IEEE_HEIGHT_INCHES))

    ax.plot(zoom_pt, all_nb["treat_all"][zoom_mask],
            color=TREAT_ALL_COLOR, lw=1.5, linestyle="--", label="Treat All")
    ax.plot(zoom_pt, np.zeros_like(zoom_pt),
            color=TREAT_NONE_COLOR, lw=1.5, linestyle=":", label="Treat None")

    for key in ["spatial", "clinical", "combined"]:
        info = FEATURE_SETS[key]
        ax.plot(zoom_pt, all_nb[key][zoom_mask],
                color=info["color"], lw=2, label=info["name"])

    set_ieee_style(ax)
    ax.set_xlim(0.10, 0.50)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.set_title("Decision Curve Analysis  (Thresholds 0.10-0.50)", fontsize=11, fontweight="bold")

    plt.tight_layout()
    save_figure(fig, "decision_curve_zoomed")
    plt.close(fig)
    print("  Saved decision_curve_zoomed (PNG 300/600, SVG, PDF)")

    # ── Report ──
    # Determine which model has highest net benefit across the most thresholds
    # Compare clinical vs combined for question 4

    # Compute average NB over full range
    avg_nb = {}
    for key in ["spatial", "clinical", "combined"]:
        avg_nb[key] = np.mean(all_nb[key])

    # Compute which model has highest NB at each threshold
    best_counts = {"spatial": 0, "clinical": 0, "combined": 0}
    for i in range(len(THRESHOLDS)):
        vals = {k: all_nb[k][i] for k in ["spatial", "clinical", "combined"]}
        best_key = max(vals, key=vals.get)
        best_counts[best_key] += 1

    best_model_key = max(best_counts, key=best_counts.get)
    best_model_name = FEATURE_SETS[best_model_key]["name"]

    # Compare clinical vs combined directly (question 4)
    clinical_higher = np.mean(all_nb["clinical"] > all_nb["combined"])
    combined_higher = np.mean(all_nb["combined"] > all_nb["clinical"])

    # Clinical utility: does spatial beat treat-all/treat-none in any region?
    spatial_best_range = []
    nb_treat_none = np.zeros_like(THRESHOLDS)
    for i, pt in enumerate(THRESHOLDS):
        spatial_val = all_nb["spatial"][i]
        ta_val = all_nb["treat_all"][i]
        if spatial_val > 0 and spatial_val > ta_val:
            spatial_best_range.append(pt)
    spatial_has_utility = len(spatial_best_range) > 0

    clinical_pos_mask = all_nb["clinical"] > 0
    clinical_pos_thresholds = THRESHOLDS[clinical_pos_mask]
    clinical_pos_str = (
        f"[{clinical_pos_thresholds.min():.2f}, {clinical_pos_thresholds.max():.2f}]"
        if len(clinical_pos_thresholds) else "none"
    )

    combined_pos_mask = all_nb["combined"] > 0
    combined_pos_thresholds = THRESHOLDS[combined_pos_mask]
    combined_pos_str = (
        f"[{combined_pos_thresholds.min():.2f}, {combined_pos_thresholds.max():.2f}]"
        if len(combined_pos_thresholds) else "none"
    )

    spatial_pos_mask = all_nb["spatial"] > 0
    spatial_pos_thresholds = THRESHOLDS[spatial_pos_mask]
    spatial_pos_str = (
        f"[{spatial_pos_thresholds.min():.2f}, {spatial_pos_thresholds.max():.2f}]"
        if len(spatial_pos_thresholds) else "none"
    )

    max_nb_clinical = float(all_nb["clinical"].max())
    max_nb_combined = float(all_nb["combined"].max())
    max_nb_spatial = float(all_nb["spatial"].max())
    thresh_clinical = float(THRESHOLDS[np.argmax(all_nb["clinical"])])
    thresh_combined = float(THRESHOLDS[np.argmax(all_nb["combined"])])
    thresh_spatial = float(THRESHOLDS[np.argmax(all_nb["spatial"])])

    # Pre‑compute comparison flags for safe f‑string interpolation
    if len(combined_pos_thresholds) and len(clinical_pos_thresholds):
        combined_range = combined_pos_thresholds.max() - combined_pos_thresholds.min()
        clinical_range = clinical_pos_thresholds.max() - clinical_pos_thresholds.min()
        range_comparison = "wider" if combined_range > clinical_range else "narrower or equal"
    else:
        range_comparison = "comparable (limited positive range)"

    if max_nb_combined > max_nb_clinical:
        peak_comparison = "higher"
    elif max_nb_combined < max_nb_clinical:
        peak_comparison = "lower"
    else:
        peak_comparison = "equal to"

    if avg_nb["combined"] > avg_nb["clinical"]:
        avg_improvement = "improves"
    elif avg_nb["combined"] < avg_nb["clinical"]:
        avg_improvement = "does not improve"
    else:
        avg_improvement = "matches"

    if best_model_key == "clinical":
        rec_text = ("report the Clinical Only model as the primary decision-making tool, "
                    "as it achieves the highest net benefit with only 6 clinical features.")
    elif best_model_key == "combined":
        rec_text = ("report the Combined model, as the addition of spatial features "
                    "improves net benefit over clinical features alone.")
    else:
        rec_text = ("report the Spatial Only model as a non-invasive alternative.")

    print(f"\n{'=' * 60}")
    print(f"  DCA complete. All outputs in: {DCA_DIR}")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
