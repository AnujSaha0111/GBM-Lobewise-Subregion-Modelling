#!/usr/bin/env python3
"""Re-run only the IDH-wildtype subgroup analysis with the fix."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count
from lifelines import CoxPHFitter
from scipy.stats import chi2
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.survival_analysis import (
    CLINICAL_COLS,
    FEATURE_COLS,
    ALL_FEATURE_COLS,
    RANDOM_SEED,
    N_BOOTSTRAP,
    load_and_prepare_data,
    _impute_clinical,
    _lifelines_cindex,
    _bootstrap_summary,
    _save_json,
)

OUTPUT_DIR = ROOT / "outputs" / "survival_incremental_value"
RIDGE_PENALTY = 0.5
CLINICAL_WT = ["age", "sex", "mgmt", "eor"]
ALL_FEATURE_WT = [c for c in ALL_FEATURE_COLS if c != "idh"]


def _extract_model_fit(cph, label, n):
    k = len(cph.params_)
    ll = cph.log_likelihood_
    aic = -2 * ll + 2 * k
    bic = -2 * ll + k * np.log(n)
    n_events = cph.event_observed
    if hasattr(n_events, '__iter__'):
        n_events = int(n_events.iloc[0]) if len(n_events) > 0 else int(n_events.sum())
    else:
        n_events = int(n_events)
    return {
        "model": label, "n": n, "n_events": n_events,
        "n_params": k, "log_likelihood": float(ll),
        "aic": float(aic), "bic": float(bic),
    }


def _lrt(full_ll, restricted_ll, df):
    stat = -2.0 * (restricted_ll - full_ll)
    p = chi2.sf(stat, df)
    return {"lrt_statistic": float(stat), "degrees_of_freedom": df, "p_value": float(p)}


def _lifelines_cindex_from_model(cph, X, y_struct):
    risk = cph.predict_partial_hazard(X).values
    return float(concordance_index_censored(y_struct["event"], y_struct["time"], risk)[0])


def _make_delta_cindex_worker(cols_m1, cols_m2, duration_col, event_col,
                               cox_cols_idx_m1, cox_cols_idx_m2, all_cols):
    def _worker(seed_i, Xy_values, y_events, y_times):
        n = len(y_events)
        rng = np.random.default_rng(seed_i)
        idx = rng.integers(0, n, size=n)
        Xy_bs = pd.DataFrame(Xy_values[idx], columns=all_cols)
        try:
            cph1 = CoxPHFitter(penalizer=RIDGE_PENALTY)
            cph1.fit(Xy_bs[cols_m1 + [duration_col, event_col]],
                     duration_col=duration_col, event_col=event_col)
            risk1 = cph1.predict_partial_hazard(Xy_bs.iloc[:, cox_cols_idx_m1]).values
            c1 = float(concordance_index_censored(y_events[idx], y_times[idx], risk1)[0])

            cph2 = CoxPHFitter(penalizer=RIDGE_PENALTY)
            cph2.fit(Xy_bs[cols_m2 + [duration_col, event_col]],
                     duration_col=duration_col, event_col=event_col)
            risk2 = cph2.predict_partial_hazard(Xy_bs.iloc[:, cox_cols_idx_m2]).values
            c2 = float(concordance_index_censored(y_events[idx], y_times[idx], risk2)[0])
            return c2 - c1
        except Exception:
            return float("nan")
    return _worker


def _bootstrap_delta_cindex(df_m1, df_m2, y_struct, n_iter=N_BOOTSTRAP, seed=RANDOM_SEED + 100):
    duration_col = "OS_months"
    event_col = "event"
    cols_m1 = [c for c in df_m1.columns if c not in (duration_col, event_col)]
    cols_m2 = [c for c in df_m2.columns if c not in (duration_col, event_col)]
    all_cols = list(dict.fromkeys(cols_m1 + cols_m2 + [duration_col, event_col]))
    cox_cols_idx_m1 = [all_cols.index(c) for c in cols_m1]
    cox_cols_idx_m2 = [all_cols.index(c) for c in cols_m2]
    Xy_values = df_m2[all_cols].values
    y_events = y_struct["event"]
    y_times = y_struct["time"]
    worker = _make_delta_cindex_worker(cols_m1, cols_m2, duration_col, event_col,
                                       cox_cols_idx_m1, cox_cols_idx_m2, all_cols)
    seeds = [seed + i for i in range(n_iter)]
    n_jobs = max(1, cpu_count() // 2)
    results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=5)(
        delayed(worker)(s, Xy_values, y_events, y_times) for s in seeds
    )
    deltas = np.array([r for r in results if not np.isnan(r)])
    print(f"  [Bootstrap delta C-index] {len(deltas)}/{n_iter} valid iterations")
    return deltas


def _delta_bootstrap_summary(deltas, observed_delta):
    mean_delta = float(np.mean(deltas))
    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))
    centered = deltas - observed_delta
    p_value = float(np.mean(np.abs(centered) >= np.abs(observed_delta)))
    return {
        "n_bootstrap": int(len(deltas)),
        "observed_delta_cindex": observed_delta,
        "mean_delta_cindex": mean_delta,
        "median_delta_cindex": float(np.median(deltas)),
        "std_delta_cindex": float(np.std(deltas, ddof=1)),
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "p_value_bootstrap": p_value,
    }


def main():
    print("=" * 60)
    print("  IDH-Wildtype Fix (Re-run)")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare_data()
    _impute_clinical(df)

    df_wt = df[df["idh"] == 0].copy()
    n_wt = len(df_wt)
    print(f"IDH-wildtype N: {n_wt}")

    for col in CLINICAL_WT:
        if col in df_wt.columns and df_wt[col].isnull().any():
            df_wt[col] = df_wt[col].fillna(df_wt[col].median())

    y_surv_wt = Surv.from_arrays(
        event=df_wt["event"].astype(bool), time=df_wt["OS_months"]
    )

    df_wt_m1 = df_wt[CLINICAL_WT + ["OS_months", "event"]].copy()
    df_wt_m2 = df_wt[ALL_FEATURE_WT + ["OS_months", "event"]].copy()

    X_wt_m1 = df_wt_m1[[c for c in df_wt_m1.columns if c not in ("OS_months", "event")]]
    X_wt_m2 = df_wt_m2[[c for c in df_wt_m2.columns if c not in ("OS_months", "event")]]

    print("\n[IDHwt Model 1] Clinical + Molecular...")
    cph_wt1 = CoxPHFitter(penalizer=RIDGE_PENALTY)
    cph_wt1.fit(df_wt_m1, duration_col="OS_months", event_col="event")
    cph_wt1.print_summary()

    print("\n[IDHwt Model 2] + Spatial...")
    cph_wt2 = CoxPHFitter(penalizer=RIDGE_PENALTY)
    cph_wt2.fit(df_wt_m2, duration_col="OS_months", event_col="event")
    cph_wt2.print_summary()

    fit_wt1 = _extract_model_fit(cph_wt1, "Clinical + Molecular (IDHwt)", n_wt)
    fit_wt2 = _extract_model_fit(cph_wt2, "Clinical + Molecular + Spatial (IDHwt)", n_wt)

    df_lrt = fit_wt2["n_params"] - fit_wt1["n_params"]
    lrt_wt = _lrt(fit_wt2["log_likelihood"], fit_wt1["log_likelihood"], df_lrt)
    print(f"\n  IDHwt LRT chi2({df_lrt}) = {lrt_wt['lrt_statistic']:.4f}, p = {lrt_wt['p_value']:.6g}")

    # Write nested comparison
    rows = [
        {"model": fit_wt1["model"], "n": fit_wt1["n"], "n_events": fit_wt1["n_events"],
         "n_params": fit_wt1["n_params"], "log_likelihood": fit_wt1["log_likelihood"],
         "aic": fit_wt1["aic"], "bic": fit_wt1["bic"]},
        {"model": fit_wt2["model"], "n": fit_wt2["n"], "n_events": fit_wt2["n_events"],
         "n_params": fit_wt2["n_params"], "log_likelihood": fit_wt2["log_likelihood"],
         "aic": fit_wt2["aic"], "bic": fit_wt2["bic"]},
    ]
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "idh_wildtype_nested_cox.csv", index=False)
    _save_json({"model_1": fit_wt1, "model_2": fit_wt2, "likelihood_ratio_test": lrt_wt},
               OUTPUT_DIR / "idh_wildtype_nested_cox.json")

    # C-index
    cindex_wt1 = _lifelines_cindex_from_model(cph_wt1, X_wt_m1, y_surv_wt)
    cindex_wt2 = _lifelines_cindex_from_model(cph_wt2, X_wt_m2, y_surv_wt)
    delta_wt = cindex_wt2 - cindex_wt1
    print(f"  IDHwt C-index M1: {cindex_wt1:.4f}, M2: {cindex_wt2:.4f}, Delta: {delta_wt:.4f}")

    # Bootstrap
    print("\n  Bootstrapping Delta C-index (IDHwt, B=5000)...")
    deltas_wt = _bootstrap_delta_cindex(df_wt_m1, df_wt_m2, y_surv_wt)
    delta_wt_bs = _delta_bootstrap_summary(deltas_wt, delta_wt)
    print(f"  Mean Delta: {delta_wt_bs['mean_delta_cindex']:.4f}")
    print(f"  95% CI: [{delta_wt_bs['ci_95_lower']:.4f}, {delta_wt_bs['ci_95_upper']:.4f}]")

    pd.DataFrame([delta_wt_bs]).to_csv(OUTPUT_DIR / "idh_wildtype_delta_cindex_bootstrap.csv", index=False)
    _save_json(delta_wt_bs, OUTPUT_DIR / "idh_wildtype_delta_cindex_bootstrap.json")

    # Report
    lrt_wt_sig = lrt_wt["p_value"] < 0.05
    report = f"""# IDH-Wildtype Nested Cox Analysis Report

## Cohort
- **N**: {n_wt}
- **Events**: {fit_wt1['n_events']} / {n_wt} ({fit_wt1['n_events']/n_wt*100:.1f}%)

## Nested Cox Model Comparison

| Metric | Model 1 (Clinical + Molecular) | Model 2 (+ Spatial) |
|--------|------|------|
| N parameters | {fit_wt1['n_params']} | {fit_wt2['n_params']} |
| Log-likelihood | {fit_wt1['log_likelihood']:.4f} | {fit_wt2['log_likelihood']:.4f} |
| AIC | {fit_wt1['aic']:.4f} | {fit_wt2['aic']:.4f} |
| BIC | {fit_wt1['bic']:.4f} | {fit_wt2['bic']:.4f} |
| C-index | {cindex_wt1:.4f} | {cindex_wt2:.4f} |

## Likelihood Ratio Test
- **chi2({df_lrt})**: {lrt_wt['lrt_statistic']:.4f}
- **p-value**: {lrt_wt['p_value']:.6g}

**Interpretation:** {'Significant' if lrt_wt_sig else 'Not significant'} - spatial features {'do' if lrt_wt_sig else 'do not'} significantly improve model fit in IDH-wildtype patients.

## Bootstrap Delta C-index
- **Bootstrap iterations**: {delta_wt_bs['n_bootstrap']}
- **Observed Delta**: {delta_wt_bs['observed_delta_cindex']:.4f}
- **Mean Delta**: {delta_wt_bs['mean_delta_cindex']:.4f}
- **95% CI**: [{delta_wt_bs['ci_95_lower']:.4f}, {delta_wt_bs['ci_95_upper']:.4f}]
- **Bootstrap p-value**: {delta_wt_bs['p_value_bootstrap']:.6g}

**Note:** Ridge regression (L2 penalty = {RIDGE_PENALTY}) was used due to multicollinearity in spatial features.
"""
    (OUTPUT_DIR / "idh_wildtype_report.md").write_text(report, encoding="utf-8")
    print("  Saved idh_wildtype_report.md")
    print("\nDone!")


if __name__ == "__main__":
    raise SystemExit(main())
