#!/usr/bin/env python3
"""FORMAL statistical significance testing for SVM ROC-AUC.

Computes:
  - ROC-AUC via sklearn.metrics.roc_auc_score
  - Bootstrap 95 % confidence interval (5 000 resamples)
  - Permutation-test p-value (5 000 shuffles)
  - Histogram of the permutation AUC distribution

Outputs:
  outputs/multimodal_lobewise_svm/auc_significance.json
  outputs/multimodal_lobewise_svm/permutation_auc_distribution.png
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PREDICTIONS_PATH = Path("outputs/multimodal_lobewise_svm/predictions.csv")
OUTPUT_DIR = Path("outputs/multimodal_lobewise_svm")
N_BOOTSTRAPS = 5000
N_PERMUTATIONS = 5000
RANDOM_STATE = 42
ALPHA = 0.05


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    required = {"patient_id", "true_label", "predicted_label",
                "prediction_probability"}
    missing = required - set(df.columns)
    assert not missing, f"Missing columns in predictions CSV: {missing}"
    y_true = df["true_label"].to_numpy(dtype=int)
    y_prob = df["prediction_probability"].to_numpy(dtype=float)
    assert y_true.ndim == 1 and y_prob.ndim == 1
    assert len(y_true) == len(y_prob) > 0
    classes = set(y_true)
    assert classes == {0, 1}, f"Both classes must be present, got {classes}"
    return y_true, y_prob


def compute_observed_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_prob))


def bootstrap_auc_ci(
    y_true: np.ndarray, y_prob: np.ndarray,
    n_bootstraps: int, random_state: int,
) -> tuple[float, float, float, np.ndarray]:
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    boot_aucs = np.empty(n_bootstraps, dtype=np.float64)
    n_valid = 0

    for i in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        pb = y_prob[idx]
        if len(np.unique(yb)) < 2:
            continue
        boot_aucs[n_valid] = roc_auc_score(yb, pb)
        n_valid += 1

    assert n_valid > 0, "All bootstrap samples contained only one class"
    boot_aucs = boot_aucs[:n_valid]

    mean_auc = float(np.mean(boot_aucs))
    ci_lower = float(np.percentile(boot_aucs, 2.5))
    ci_upper = float(np.percentile(boot_aucs, 97.5))
    return mean_auc, ci_lower, ci_upper, boot_aucs


def permutation_test(
    y_true: np.ndarray, y_prob: np.ndarray,
    observed_auc: float, n_permutations: int,
    random_state: int,
) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(random_state)
    perm_aucs = np.empty(n_permutations, dtype=np.float64)

    for i in range(n_permutations):
        y_shuffled = y_true.copy()
        rng.shuffle(y_shuffled)
        perm_aucs[i] = roc_auc_score(y_shuffled, y_prob)

    n_exceed = int(np.sum(perm_aucs >= observed_auc))
    p_value = (n_exceed + 1.0) / (n_permutations + 1.0)
    return p_value, perm_aucs


def plot_permutation_distribution(
    perm_aucs: np.ndarray, observed_auc: float,
    p_value: float, save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(perm_aucs, bins=50, color="skyblue", edgecolor="black",
            alpha=0.8, density=True)
    ax.axvline(observed_auc, color="red", linestyle="--", linewidth=2,
               label=f"Observed AUC = {observed_auc:.4f}")

    n_exceed = int(np.sum(perm_aucs >= observed_auc))
    ax.legend(fontsize=11)
    ax.set_xlabel("ROC-AUC under Permutation (Random Labels)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"Permutation Distribution of ROC-AUC\n"
        f"Observed AUC = {observed_auc:.4f}  |  "
        f"Permutations \u2265 observed: {n_exceed}/{len(perm_aucs)}  |  "
        f"p = {p_value:.4f}",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def print_interpretation(
    observed_auc: float, ci_lower: float, ci_upper: float,
    p_value: float, n_samples: int,
) -> None:
    sig_str = ("statistically significant" if p_value < ALPHA
               else "not statistically significant")
    ci_width = ci_upper - ci_lower

    print()
    print("=" * 72)
    print("  STATISTICAL SIGNIFICANCE — SVM ROC-AUC")
    print("=" * 72)
    print(f"  Observed ROC-AUC:              {observed_auc:.4f}")
    print(f"  95 % Bootstrap CI:             [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  Permutation p-value:           {p_value:.4f}  ({p_value:.6g})")
    print(f"  Bootstrap mean AUC:            {observed_auc:.4f}  "
          f"(n = {N_BOOTSTRAPS})")
    print(f"  Sample size (test set):        {n_samples}")
    print(f"  Permutation iterations:        {N_PERMUTATIONS}")
    print("=" * 72)
    print()
    print("  SCIENTIFIC INTERPRETATION")
    print()
    print(f"  The SVM achieves a ROC-AUC of {observed_auc:.4f}, which is "
          f"{sig_str}")
    print(f"  at the alpha = {ALPHA} level (p = {p_value:.4f}).")

    if observed_auc >= 0.55:
        print(f"  The model outperforms random classification; however, the")
        print(f"  effect size remains modest, reflecting limited discriminative")
        print(f"  capacity.")
    else:
        print(f"  The model does not meaningfully outperform random guessing.")

    if ci_width > 0.2:
        print(f"  The 95 % CI [{ci_lower:.3f}, {ci_upper:.3f}] spans "
              f"{ci_width:.3f}, indicating")
        print(f"  substantial uncertainty around the point estimate. This is")
        print(f"  attributable to the modest test-set size (n = {n_samples}).")
    else:
        print(f"  The 95 % CI [{ci_lower:.3f}, {ci_upper:.3f}] is relatively")
        print(f"  narrow, suggesting reasonably stable AUC estimation.")

    print()
    print("  Limitations:")
    print(f"    - Test-set size (n = {n_samples}) limits precision of the")
    print(f"      AUC estimate and its confidence interval.")
    print(f"    - Coarse lobewise-subregion features may not capture")
    print(f"      fine-grained spatial heterogeneity relevant to outcome.")
    print(f"    - Permutation tests assume exchangeability under the null;")
    print(f"      violations can affect p-value validity.")
    print(f"    - No correction for multiple comparisons is applied.")
    print(f"    - Bootstrap CIs are percentile-based and may exhibit")
    print(f"      coverage distortion with small samples.")
    print("=" * 72)
    print()


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    y_true, y_prob = load_predictions(PREDICTIONS_PATH)
    n_samples = len(y_true)
    print(f"\nLoaded {n_samples} test-set predictions from {PREDICTIONS_PATH}")

    observed_auc = compute_observed_auc(y_true, y_prob)
    print(f"Observed ROC-AUC: {observed_auc:.4f}")

    print(f"Bootstrap resampling ({N_BOOTSTRAPS}) ...")
    boot_mean, ci_lower, ci_upper, _ = bootstrap_auc_ci(
        y_true, y_prob, N_BOOTSTRAPS, RANDOM_STATE,
    )
    print(f"  Bootstrap mean AUC: {boot_mean:.4f}")
    print(f"  95 % CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    print(f"Permutation test ({N_PERMUTATIONS}) ...")
    p_value, perm_aucs = permutation_test(
        y_true, y_prob, observed_auc, N_PERMUTATIONS, RANDOM_STATE + 1,
    )
    print(f"  Permutation p-value: {p_value:.4f}")

    results = {
        "observed_auc": observed_auc,
        "bootstrap_mean_auc": boot_mean,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "permutation_p_value": p_value,
        "n_bootstraps": N_BOOTSTRAPS,
        "n_permutations": N_PERMUTATIONS,
        "n_test_samples": n_samples,
        "alpha": ALPHA,
    }
    json_path = OUTPUT_DIR / "auc_significance.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {json_path}")

    plot_path = OUTPUT_DIR / "permutation_auc_distribution.png"
    plot_permutation_distribution(perm_aucs, observed_auc, p_value, plot_path)
    print(f"Saved {plot_path}")

    print_interpretation(observed_auc, ci_lower, ci_upper, p_value, n_samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
