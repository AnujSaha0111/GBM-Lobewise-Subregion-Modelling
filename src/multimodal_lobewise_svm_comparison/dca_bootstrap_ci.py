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
N_BOOTSTRAP = 5000
CI_ALPHA = 0.05

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

IEEE_WIDTH_INCHES = 6.5
IEEE_HEIGHT_INCHES = 5.0


def load_and_split():
    df = pd.read_csv(INPUT_CSV)
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    y = df["risk_label"].astype(int)

    spatial_cols = [c for c in df.columns if c.startswith(SPATIAL_FEATURE_PREFIXES)]
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


def bootstrap_dca(y_true, probs, thresholds, n_bootstrap, random_state):
    """Bootstrap net benefit curves and return mean + percentile CIs.

    Returns a dict:
        key -> {
            "mean": np.ndarray,
            "ci_lower": np.ndarray,
            "ci_upper": np.ndarray,
            "point_nb": np.ndarray,   (point estimate on original test set)
            "all_bootstrap": np.ndarray (n_bootstrap x n_thresholds)
        }
    Also includes "treat_all" key.
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    n_thresh = len(thresholds)
    keys = list(probs.keys())

    # Point estimate (original full test set)
    point = {}
    for key in keys:
        point[key] = net_benefit(y_true, probs[key], thresholds)
    point["treat_all"] = treat_all_nb(y_true, thresholds)

    # Bootstrap storage
    boot = {key: np.full((n_bootstrap, n_thresh), np.nan) for key in keys}
    boot["treat_all"] = np.full((n_bootstrap, n_thresh), np.nan)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        for key in keys:
            pb = probs[key][idx]
            boot[key][b, :] = net_benefit(yb, pb, thresholds)
        boot["treat_all"][b, :] = treat_all_nb(yb, thresholds)

    results = {}
    for key in list(keys) + ["treat_all"]:
        mean_nb = np.nanmean(boot[key], axis=0)
        ci_lower = np.nanpercentile(boot[key], 100 * CI_ALPHA / 2, axis=0)
        ci_upper = np.nanpercentile(boot[key], 100 * (1 - CI_ALPHA / 2), axis=0)
        results[key] = {
            "mean": mean_nb,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "point_nb": point[key],
            "all_bootstrap": boot[key],
        }

    return results


def set_ieee_style(ax):
    ax.tick_params(direction="in", which="both", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.set_xlabel("Threshold Probability", fontsize=10)
    ax.set_ylabel("Net Benefit", fontsize=10)


def save_figure(fig, stem):
    for ext, kwargs in [
        ("png", {"dpi": 300, "bbox_inches": "tight"}),
        ("svg", {"bbox_inches": "tight"}),
        ("pdf", {"bbox_inches": "tight"}),
    ]:
        path = DCA_DIR / f"{stem}.{ext}"
        fig.savefig(path, format=ext, **kwargs)


def compute_ci_overlap(results, thresholds):
    """Return threshold regions where CIs of different model pairs overlap."""
    keys = ["spatial", "clinical", "combined"]
    overlap_regions = {}
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            label = f"{FEATURE_SETS[k1]['name']} vs {FEATURE_SETS[k2]['name']}"
            overlap = (
                (results[k1]["ci_lower"] <= results[k2]["ci_upper"]) &
                (results[k2]["ci_lower"] <= results[k1]["ci_upper"])
            )
            # Find contiguous regions
            regions = []
            if overlap.any():
                # Cast to int BEFORE diff so that -1 transitions are preserved
                overlap_int = overlap.astype(np.int8)
                padded = np.concatenate(([0], overlap_int, [0]))
                diffs = np.diff(padded)
                starts = np.where(diffs == 1)[0]
                ends = np.where(diffs == -1)[0]
                for s, e in zip(starts, ends):
                    regions.append(f"[{thresholds[s]:.2f}, {thresholds[e - 1]:.2f}]")
            overlap_regions[label] = regions if regions else ["None"]
    return overlap_regions


def compute_significant_regions(results, thresholds):
    """Threshold regions where Combined model CI is entirely above Treat All / Treat None."""
    combined = results["combined"]
    treat_all = results["treat_all"]
    treat_none = np.zeros_like(thresholds)

    sig_vs_treat_all = (
        (combined["ci_lower"] > treat_all["ci_upper"]) &
        (combined["mean"] > treat_all["mean"])
    )
    sig_vs_treat_none = (
        (combined["ci_lower"] > treat_none) &
        (combined["mean"] > 0)
    )

    def regions(mask):
        regions_list = []
        if mask.any():
            mask_int = mask.astype(np.int8)
            padded = np.concatenate(([0], mask_int, [0]))
            diffs = np.diff(padded)
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]
            for s, e in zip(starts, ends):
                regions_list.append(f"[{thresholds[s]:.2f}, {thresholds[e - 1]:.2f}]")
        return regions_list if regions_list else ["None"]

    return {
        "combined_exceeds_treat_all": regions(sig_vs_treat_all),
        "combined_exceeds_treat_none": regions(sig_vs_treat_none),
    }


def main():
    print("=" * 60)
    print("  DECISION CURVE ANALYSIS — BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 60)

    # ── Fit models and get test-set probability predictions ──
    probs, y_test_vals, X_te, y_te = fit_and_predict()
    n_test = len(y_test_vals)
    print(f"\nTest set size: {n_test}")
    print(f"  High-risk: {y_test_vals.sum()} / {n_test}")

    # ── Bootstrap ──
    print(f"\nRunning {N_BOOTSTRAP} bootstrap iterations...")
    results = bootstrap_dca(y_test_vals, probs, THRESHOLDS, N_BOOTSTRAP, RANDOM_STATE)
    print("  Done.")

    # ── Decision curve with CI: CSV ──
    rows = []
    for i, pt in enumerate(THRESHOLDS):
        row = {
            "threshold": round(pt, 4),
            "treat_none": 0.0,
            "treat_all_mean": round(results["treat_all"]["mean"][i], 6),
            "treat_all_ci_lower": round(results["treat_all"]["ci_lower"][i], 6),
            "treat_all_ci_upper": round(results["treat_all"]["ci_upper"][i], 6),
        }
        for key in ["spatial", "clinical", "combined"]:
            name = FEATURE_SETS[key]["name"]
            row[f"{name}_mean"] = round(results[key]["mean"][i], 6)
            row[f"{name}_ci_lower"] = round(results[key]["ci_lower"][i], 6)
            row[f"{name}_ci_upper"] = round(results[key]["ci_upper"][i], 6)
        rows.append(row)

    data_df = pd.DataFrame(rows)
    csv_path = DCA_DIR / "decision_curve_with_ci.csv"
    data_df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path.name}")

    # ── JSON ──
    json_data = {
        "metadata": {
            "n_test": n_test,
            "n_high_risk": int(y_test_vals.sum()),
            "n_bootstrap": N_BOOTSTRAP,
            "ci_method": "percentile",
            "ci_level": 1 - CI_ALPHA,
            "thresholds": THRESHOLDS.tolist(),
        },
        "models": {},
    }
    for key in ["spatial", "clinical", "combined", "treat_all"]:
        name = FEATURE_SETS[key]["name"] if key in FEATURE_SETS else "Treat All"
        model_data = {}
        for i, pt in enumerate(THRESHOLDS):
            model_data[str(round(pt, 4))] = {
                "mean": round(float(results[key]["mean"][i]), 6),
                "ci_lower": round(float(results[key]["ci_lower"][i]), 6),
                "ci_upper": round(float(results[key]["ci_upper"][i]), 6),
                "point_nb": round(float(results[key]["point_nb"][i]), 6),
            }
        json_data["models"][key] = {
            "name": name,
            "net_benefit": model_data,
        }

    # Add overlap analysis
    json_data["ci_overlap"] = compute_ci_overlap(results, THRESHOLDS)
    json_data["significant_regions"] = compute_significant_regions(results, THRESHOLDS)

    json_path = DCA_DIR / "decision_curve_with_ci.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved {json_path.name}")

    # ── Figures ──
    print("\n  Generating figures ...")

    fig, ax = plt.subplots(figsize=(IEEE_WIDTH_INCHES, IEEE_HEIGHT_INCHES))

    # Treat All with CI
    ta = results["treat_all"]
    ax.fill_between(THRESHOLDS, ta["ci_lower"], ta["ci_upper"],
                    color=TREAT_ALL_COLOR, alpha=0.15)
    ax.plot(THRESHOLDS, ta["mean"], color=TREAT_ALL_COLOR, lw=1.5,
            linestyle="--", label="Treat All")

    # Treat None (zero line)
    ax.plot(THRESHOLDS, np.zeros_like(THRESHOLDS),
            color=TREAT_NONE_COLOR, lw=1.5, linestyle=":", label="Treat None")

    # Model curves with CI bands
    for key in ["spatial", "clinical", "combined"]:
        info = FEATURE_SETS[key]
        r = results[key]
        ax.fill_between(THRESHOLDS, r["ci_lower"], r["ci_upper"],
                        color=info["color"], alpha=0.15)
        ax.plot(THRESHOLDS, r["mean"], color=info["color"], lw=2,
                label=info["name"])

    set_ieee_style(ax)
    ax.set_xlim(-0.01, 1.01)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.set_title("Decision Curve Analysis with 95% Bootstrap CI", fontsize=11, fontweight="bold")

    plt.tight_layout()
    save_figure(fig, "decision_curve_with_ci")
    plt.close(fig)
    print("  Saved decision_curve_with_ci (PNG, SVG, PDF)")

    # ── Report ──
    overlap_info = compute_ci_overlap(results, THRESHOLDS)
    sig_info = compute_significant_regions(results, THRESHOLDS)

    # Mean NB summary
    mean_nb = {}
    for key in ["spatial", "clinical", "combined"]:
        mean_nb[key] = np.mean(results[key]["mean"])

    # Positive NB ranges based on mean curve
    pos_ranges = {}
    for key in ["spatial", "clinical", "combined"]:
        mask = results[key]["mean"] > 0
        if mask.any():
            idxs = np.where(mask)[0]
            pos_ranges[key] = f"[{THRESHOLDS[idxs[0]]:.2f}, {THRESHOLDS[idxs[-1]]:.2f}]"
        else:
            pos_ranges[key] = "none"

    # CI overlap regions text
    overlap_lines = []
    for label, regions in overlap_info.items():
        overlap_lines.append(f"- **{label}**: {'overlap at ' + ', '.join(regions) if regions != ['None'] else 'no overlap detected'}")

    # Significant regions text
    sig_lines = []
    sig_all = []
    for region_label, region_list in sig_info.items():
        if region_list != ["None"]:
            sig_lines.append(f"- **{region_label}**: {', '.join(region_list)}")
            sig_all.extend(region_list)
        else:
            sig_lines.append(f"- **{region_label}**: none")

    combined_sig = sig_info["combined_exceeds_treat_all"] != ["None"] or sig_info["combined_exceeds_treat_none"] != ["None"]

    print(f"\n{'=' * 60}")
    print(f"  DCA bootstrap CI complete. Outputs in: {DCA_DIR}")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
