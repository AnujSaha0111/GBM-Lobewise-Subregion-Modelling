#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import warnings
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from scipy.stats import chi2
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.util import Surv

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT
OUTPUT_DIR = ROOT / "outputs" / "survival_analysis"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

# ── Constants ────────────────────────────────────────────────────────
RANDOM_SEED = 42
N_BOOTSTRAP = 5000
N_BOOTSTRAP_RSF = 5000
N_JOBS = -1

FEATURE_COLS = [
    "global_nc_en_ratio",
    "global_ed_en_ratio",
    "global_ed_total_ratio",
    "tumor_burden_index",
    *(f"{lb}_{sub}_ratio"
      for lb in ("frontal", "temporal", "parietal", "occipital")
      for sub in ("ed", "en", "nc")),
]

CLINICAL_COLS = ["age", "sex", "mgmt", "idh", "eor"]
CLINICAL_LABELS = {
    "age": "Age",
    "sex": "Sex (Male)",
    "mgmt": "MGMT promoter (methylated)",
    "idh": "IDH (mutant)",
    "eor": "EOR (GTR vs STR vs biopsy)",
}
ALL_FEATURE_COLS = CLINICAL_COLS + FEATURE_COLS

# ── Helpers ──────────────────────────────────────────────────────────


def _normalize_patient_id(pid: str) -> str:
    pid = str(pid).strip()
    m = re.match(r"^(UCSF-PDGM-)(\d+)(.*)", pid)
    if m:
        return f"{m.group(1)}{int(m.group(2)):04d}{m.group(3)}"
    return pid


def _save_json(payload: dict | list, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _checkpoint_path(name: str) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR / name


def _save_checkpoint_json(name: str, payload: dict | list) -> None:
    _save_json(payload, _checkpoint_path(name))


def _load_checkpoint_json(name: str) -> dict | list | None:
    path = _checkpoint_path(name)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_checkpoint_array(name: str, values: np.ndarray) -> None:
    path = _checkpoint_path(name)
    np.save(path, values)


def _load_checkpoint_array(name: str) -> np.ndarray | None:
    path = _checkpoint_path(name)
    if not path.exists():
        return None
    return np.load(path)


def _append_rows(existing: list[dict[str, object]], new_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seen = {(row.get("model"), row.get("variable")) for row in existing}
    merged = list(existing)
    for row in new_rows:
        key = (row.get("model"), row.get("variable"))
        if key not in seen:
            merged.append(row)
            seen.add(key)
    return merged


def _upsert_bootstrap_row(existing: list[dict[str, object]], new_row: dict[str, object]) -> list[dict[str, object]]:
    merged = [row for row in existing if row.get("model") != new_row.get("model")]
    merged.append(new_row)
    return merged


def _write_intermediate_outputs(
    results: dict[str, object],
    cox_rows: list[dict[str, object]],
    bootstrap_cindex_rows: list[dict[str, object]],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cox_rows).to_csv(OUTPUT_DIR / "cox_results.csv", index=False)
    _save_json(results, OUTPUT_DIR / "cox_results.json")
    pd.DataFrame(bootstrap_cindex_rows).to_csv(
        OUTPUT_DIR / "cindex_bootstrap.csv", index=False
    )
    _save_json(
        {row["model"]: row for row in bootstrap_cindex_rows},
        OUTPUT_DIR / "cindex_bootstrap.json",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run survival analysis.")
    parser.add_argument(
        "--skip-missing-rsf-bootstrap",
        action="store_true",
        help="Do not run the RSF bootstrap if its checkpoint is missing; write NA summary instead.",
    )
    return parser.parse_args()


def _evaluate_survival_functions_at_times(surv_functions: list, times: np.ndarray) -> np.ndarray:
    return np.asarray([[fn(t) for t in times] for fn in surv_functions], dtype=float)


# ── Data Loading ────────────────────────────────────────────────────


def load_and_prepare_data() -> pd.DataFrame:
    """Load data, apply same cohort filters as classification, return n=493.

    Returns DataFrame with columns:
        patient_id, OS_months, event,
        age, sex, mgmt, idh, eor,
        + 16 spatial feature columns.
    """
    # Load raw features
    raw = pd.read_csv(DATA_DIR / "outputs" / "features_raw.csv")

    # Filter reliable lobe assignments (same as preprocessing.py)
    reliable_series = raw["lobe_assignment_reliable"]
    if pd.api.types.is_bool_dtype(reliable_series):
        mask = reliable_series.fillna(False)
    elif pd.api.types.is_numeric_dtype(reliable_series):
        mask = reliable_series.fillna(0).astype(int) != 0
    else:
        mask = reliable_series.astype(str).str.strip().str.lower().isin(
            ["true", "1", "yes", "y"]
        )
    raw = raw[mask].copy()

    # Filter missing / invalid OS_months
    os_num = pd.to_numeric(raw["OS_months"], errors="coerce")
    raw = raw[os_num.notna()].copy()
    raw["OS_months"] = os_num[os_num.notna()]

    n = len(raw)
    print(f"[Data] Cohort size after filtering: {n}")

    # Median-impute spatial features (same as preprocessing.py default path)
    for col in FEATURE_COLS:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    medians = raw[FEATURE_COLS].median()
    raw[FEATURE_COLS] = raw[FEATURE_COLS].fillna(medians)

    # Load metadata
    meta_cols = [
        "ID", "Sex", "Age at MRI", "MGMT status",
        "IDH", "EOR", "1-dead 0-alive",
    ]
    meta = pd.read_csv(
        DATA_DIR / "UCSF-PDGM-metadata_v5.csv",
        usecols=meta_cols,
    )
    meta["patient_id"] = meta.pop("ID").apply(_normalize_patient_id)

    # Merge
    df = raw[["patient_id", "OS_months"] + FEATURE_COLS].merge(
        meta, on="patient_id", how="left"
    )

    # ── Encode clinical variables ──
    # Sex: M→1, F→0
    df["sex"] = df.pop("Sex").map({"M": 1, "F": 0}).astype(int)

    # Age
    df["age"] = pd.to_numeric(df.pop("Age at MRI"), errors="coerce")

    # IDH: wildtype→0, anything else→1
    df["idh"] = (
        df.pop("IDH")
        .apply(lambda x: 0 if str(x).strip().lower() == "wildtype" else 1)
        .astype(int)
    )

    # MGMT: positive→1, negative→0, unknown→NaN
    mgmt_raw = df.pop("MGMT status")
    df["mgmt"] = mgmt_raw.map({"positive": 1.0, "negative": 0.0})

    # EOR: biopsy→0, STR→1, GTR→2
    eor_map = {"biopsy": 0.0, "str": 1.0, "gtr": 2.0}
    df["eor"] = (
        df.pop("EOR").astype(str).str.strip().str.lower().map(eor_map)
    )

    # Event indicator: 1-dead 0-alive
    df["event"] = pd.to_numeric(
        df.pop("1-dead 0-alive"), errors="coerce"
    ).astype(int)

    print(f"[Data] Final shape: {df.shape}")
    print(f"[Data] Events: {df['event'].sum()} / {len(df)}  "
          f"({df['event'].mean() * 100:.1f}%)")
    missing = df[CLINICAL_COLS].isnull().sum()
    if missing.sum() > 0:
        print(f"[Data] Missing clinical values:\n{missing[missing > 0]}")

    return df


# ── Imputation helper ───────────────────────────────────────────────


def _impute_clinical(df: pd.DataFrame) -> pd.DataFrame:
    """Median-impute missing clinical values in-place."""
    for col in CLINICAL_COLS:
        if col in df.columns and df[col].isnull().any():
            med = df[col].median()
            n_miss = df[col].isnull().sum()
            df[col] = df[col].fillna(med)
            print(f"[Impute] {col}: {n_miss} missing -> filled with median ({med})")
    return df


# ── Model definitions ────────────────────────────────────────────────


def _prepare_clinical_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with clinical features + OS_months + event."""
    _impute_clinical(df)
    return df[CLINICAL_COLS + ["OS_months", "event"]].copy()


def _prepare_spatial_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with spatial features + OS_months + event."""
    return df[FEATURE_COLS + ["OS_months", "event"]].copy()


def _prepare_combined_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with all features + OS_months + event."""
    _impute_clinical(df)
    return df[ALL_FEATURE_COLS + ["OS_months", "event"]].copy()


# ── C-index bootstrap ────────────────────────────────────────────────


def _cindex_sksurv(model, X: pd.DataFrame, y_struct: np.ndarray) -> float:
    """Compute C-index from a fitted sksurv model."""
    try:
        risk = model.predict(X)
    except Exception:
        risk = model.predict(X.values if hasattr(X, "values") else X)
    return float(
        concordance_index_censored(
            y_struct["event"], y_struct["time"], risk
        )[0]
    )


def _lifelines_cindex(
    cph: CoxPHFitter, X: pd.DataFrame, y_struct: np.ndarray
) -> float:
    """C-index from lifelines Cox model partial hazard."""
    risk = cph.predict_partial_hazard(X).values
    return float(
        concordance_index_censored(
            y_struct["event"], y_struct["time"], risk
        )[0]
    )


def _make_cox_boot_worker(
    Xy_cols: list[str],
    duration_col: str,
    event_col: str,
    cox_cols_idx: list[int],
):
    """Factory returning a worker function for Cox bootstrap (avoids globals)."""
    def _worker(seed_i: int, Xy_values: np.ndarray,
                y_events: np.ndarray, y_times: np.ndarray) -> float:
        n = len(y_events)
        rng = np.random.default_rng(seed_i)
        idx = rng.integers(0, n, size=n)
        Xy_bs = pd.DataFrame(Xy_values[idx], columns=Xy_cols)
        try:
            cph = CoxPHFitter()
            cph.fit(Xy_bs, duration_col=duration_col, event_col=event_col)
            risk = cph.predict_partial_hazard(
                Xy_bs.iloc[:, cox_cols_idx]
            ).values
            return float(
                concordance_index_censored(
                    y_events[idx], y_times[idx], risk
                )[0]
            )
        except Exception:
            return float("nan")
    return _worker


def _bootstrap_cindex_cox(
    Xy: pd.DataFrame,
    y_struct: np.ndarray,
    duration_col: str = "OS_months",
    event_col: str = "event",
    n_iter: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Bootstrap C-index for lifelines Cox model using joblib parallel."""
    cox_cols = [c for c in Xy.columns if c not in (duration_col, event_col)]
    cox_cols_idx = [list(Xy.columns).index(c) for c in cox_cols]
    Xy_cols = list(Xy.columns)
    Xy_values = Xy.values
    y_events = y_struct["event"]
    y_times = y_struct["time"]
    worker = _make_cox_boot_worker(
        Xy_cols, duration_col, event_col, cox_cols_idx
    )

    seeds = [seed + i for i in range(n_iter)]
    n_jobs = max(1, cpu_count() // 2)

    results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=5)(
        delayed(worker)(s, Xy_values, y_events, y_times)
        for s in seeds
    )
    scores = np.array([r for r in results if not np.isnan(r)])
    print(f"  [Bootstrap Cox] {len(scores)}/{n_iter} valid iterations")
    return scores


def _make_rsf_boot_worker(**rsf_kwargs):
    """Factory returning a worker function for RSF bootstrap."""
    def _worker(seed_i: int, X_values: np.ndarray,
                y_events: np.ndarray, y_times: np.ndarray) -> float:
        n = len(y_events)
        rng = np.random.default_rng(seed_i)
        idx = rng.integers(0, n, size=n)
        X_bs = pd.DataFrame(X_values[idx])
        y_bs = np.empty(
            dtype=[("event", bool), ("time", float)], shape=n
        )
        y_bs["event"] = y_events[idx]
        y_bs["time"] = y_times[idx]
        try:
            m = RandomSurvivalForest(**rsf_kwargs)
            m.fit(X_bs, y_bs)
            risk = m.predict(X_bs)
            return float(
                concordance_index_censored(
                    y_bs["event"], y_bs["time"], risk
                )[0]
            )
        except Exception:
            return float("nan")
    return _worker


def _bootstrap_cindex_rsf(
    X: pd.DataFrame,
    y_struct: np.ndarray,
    n_iter: int = N_BOOTSTRAP_RSF,
    seed: int = RANDOM_SEED,
    **rsf_kwargs: object,
) -> np.ndarray:
    """Bootstrap C-index for RSF model using joblib parallel."""
    X_values = X.values
    y_events = y_struct["event"]
    y_times = y_struct["time"]
    worker = _make_rsf_boot_worker(**rsf_kwargs)

    seeds = [seed + i for i in range(n_iter)]
    n_jobs = max(1, cpu_count() // 2)

    results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=5)(
        delayed(worker)(s, X_values, y_events, y_times)
        for s in seeds
    )
    scores = np.array([r for r in results if not np.isnan(r)])
    print(f"  [Bootstrap RSF] {len(scores)}/{n_iter} valid iterations")
    return scores


def _bootstrap_summary(scores: np.ndarray, label: str) -> dict:
    """Compute summary statistics from bootstrap scores."""
    return {
        "model": label,
        "n_bootstrap": int(len(scores)),
        "cindex_mean": float(np.mean(scores)),
        "cindex_median": float(np.median(scores)),
        "cindex_std": float(np.std(scores, ddof=1)),
        "ci_95_lower": float(np.percentile(scores, 2.5)),
        "ci_95_upper": float(np.percentile(scores, 97.5)),
    }


# ── Main ─────────────────────────────────────────────────────────────


def main() -> int:
    args = _parse_args()
    print("=" * 60)
    print("  GBM Survival Analysis")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    df = load_and_prepare_data()

    # Survival structured array (for sksurv)
    y_surv = Surv.from_arrays(
        event=df["event"].astype(bool), time=df["OS_months"]
    )

    # ── Data for each model ──
    df_clin = _prepare_clinical_df(df)
    df_spat = _prepare_spatial_df(df)
    df_comb = _prepare_combined_df(df)

    clin_feat = [c for c in df_clin.columns if c not in ("OS_months", "event")]
    spat_feat = [c for c in df_spat.columns if c not in ("OS_months", "event")]
    comb_feat = [c for c in df_comb.columns if c not in ("OS_months", "event")]

    X_clin = df_clin[clin_feat]
    X_spat = df_spat[spat_feat]
    X_comb = df_comb[comb_feat]

    results = _load_checkpoint_json("results.json") or {}
    cox_rows = _load_checkpoint_json("cox_rows.json") or []
    bootstrap_cindex_rows = _load_checkpoint_json("bootstrap_rows.json") or []

    # ================================================================
    #  A) Clinical-molecular Cox PH
    # ================================================================
    print("\n" + "-" * 40)
    print("Model A: Clinical-molecular Cox PH")
    print("-" * 40)

    cph_clin = CoxPHFitter()
    cph_clin.fit(df_clin, duration_col="OS_months", event_col="event")
    cph_clin.print_summary()

    cindex_clin = _lifelines_cindex(cph_clin, X_clin, y_surv)
    print(f"  C-index (in-sample): {cindex_clin:.4f}")

    if "clinical_molecular_cox" not in results:
        summary_clin = cph_clin.summary
        new_rows = []
        for idx in summary_clin.index:
            row = summary_clin.loc[idx]
            new_rows.append({
                "model": "Clinical-molecular",
                "variable": idx,
                "coef": float(row["coef"]),
                "exp_coef": float(row["exp(coef)"]),
                "se": float(row["se(coef)"]),
                "z": float(row["z"]),
                "p": float(row["p"]),
                "ci_lower": float(row["coef lower 95%"]),
                "ci_upper": float(row["coef upper 95%"]),
            })
        cox_rows = _append_rows(cox_rows, new_rows)
        results["clinical_molecular_cox"] = {
            "cindex": cindex_clin,
            "n_features": len(clin_feat),
            "features": clin_feat,
            "coefficients": {
                idx: {
                    "coef": float(summary_clin.loc[idx, "coef"]),
                    "exp_coef": float(summary_clin.loc[idx, "exp(coef)"]),
                    "p": float(summary_clin.loc[idx, "p"]),
                }
                for idx in summary_clin.index
            },
        }
        _save_checkpoint_json("results.json", results)
        _save_checkpoint_json("cox_rows.json", cox_rows)
        _write_intermediate_outputs(results, cox_rows, bootstrap_cindex_rows)

    bs_clin = _load_checkpoint_array("bs_clin.npy")
    if bs_clin is None:
        print("  Bootstrapping C-index (B=5000)...")
        bs_clin = _bootstrap_cindex_cox(df_clin, y_surv)
        _save_checkpoint_array("bs_clin.npy", bs_clin)
    bs_summary_clin = _bootstrap_summary(bs_clin, "Clinical-molecular Cox")
    bootstrap_cindex_rows = _upsert_bootstrap_row(bootstrap_cindex_rows, bs_summary_clin)
    _save_checkpoint_json("bootstrap_rows.json", bootstrap_cindex_rows)
    _write_intermediate_outputs(results, cox_rows, bootstrap_cindex_rows)

    # ================================================================
    #  B) Spatial-feature Cox PH
    # ================================================================
    print("\n" + "-" * 40)
    print("Model B: Spatial-feature Cox PH")
    print("-" * 40)

    cph_spat = CoxPHFitter()
    cph_spat.fit(df_spat, duration_col="OS_months", event_col="event")
    cph_spat.print_summary()

    cindex_spat = _lifelines_cindex(cph_spat, X_spat, y_surv)
    print(f"  C-index (in-sample): {cindex_spat:.4f}")

    if "spatial_cox" not in results:
        summary_spat = cph_spat.summary
        new_rows = []
        for idx in summary_spat.index:
            row = summary_spat.loc[idx]
            new_rows.append({
                "model": "Spatial",
                "variable": idx,
                "coef": float(row["coef"]),
                "exp_coef": float(row["exp(coef)"]),
                "se": float(row["se(coef)"]),
                "z": float(row["z"]),
                "p": float(row["p"]),
                "ci_lower": float(row["coef lower 95%"]),
                "ci_upper": float(row["coef upper 95%"]),
            })
        cox_rows = _append_rows(cox_rows, new_rows)
        results["spatial_cox"] = {
            "cindex": cindex_spat,
            "n_features": len(spat_feat),
            "features": spat_feat,
            "coefficients": {
                idx: {
                    "coef": float(summary_spat.loc[idx, "coef"]),
                    "exp_coef": float(summary_spat.loc[idx, "exp(coef)"]),
                    "p": float(summary_spat.loc[idx, "p"]),
                }
                for idx in summary_spat.index
            },
        }
        _save_checkpoint_json("results.json", results)
        _save_checkpoint_json("cox_rows.json", cox_rows)
        _write_intermediate_outputs(results, cox_rows, bootstrap_cindex_rows)

    bs_spat = _load_checkpoint_array("bs_spat.npy")
    if bs_spat is None:
        print("  Bootstrapping C-index (B=5000)...")
        bs_spat = _bootstrap_cindex_cox(df_spat, y_surv)
        _save_checkpoint_array("bs_spat.npy", bs_spat)
    bs_summary_spat = _bootstrap_summary(bs_spat, "Spatial Cox")
    bootstrap_cindex_rows = _upsert_bootstrap_row(bootstrap_cindex_rows, bs_summary_spat)
    _save_checkpoint_json("bootstrap_rows.json", bootstrap_cindex_rows)
    _write_intermediate_outputs(results, cox_rows, bootstrap_cindex_rows)

    # ================================================================
    #  C) Combined Cox PH
    # ================================================================
    print("\n" + "-" * 40)
    print("Model C: Combined Cox PH")
    print("-" * 40)

    cph_comb = CoxPHFitter()
    cph_comb.fit(df_comb, duration_col="OS_months", event_col="event")
    cph_comb.print_summary()

    cindex_comb = _lifelines_cindex(cph_comb, X_comb, y_surv)
    print(f"  C-index (in-sample): {cindex_comb:.4f}")

    if "combined_cox" not in results:
        summary_comb = cph_comb.summary
        new_rows = []
        for idx in summary_comb.index:
            row = summary_comb.loc[idx]
            new_rows.append({
                "model": "Combined",
                "variable": idx,
                "coef": float(row["coef"]),
                "exp_coef": float(row["exp(coef)"]),
                "se": float(row["se(coef)"]),
                "z": float(row["z"]),
                "p": float(row["p"]),
                "ci_lower": float(row["coef lower 95%"]),
                "ci_upper": float(row["coef upper 95%"]),
            })
        cox_rows = _append_rows(cox_rows, new_rows)
        results["combined_cox"] = {
            "cindex": cindex_comb,
            "n_features": len(comb_feat),
            "features": comb_feat,
            "coefficients": {
                idx: {
                    "coef": float(summary_comb.loc[idx, "coef"]),
                    "exp_coef": float(summary_comb.loc[idx, "exp(coef)"]),
                    "p": float(summary_comb.loc[idx, "p"]),
                }
                for idx in summary_comb.index
            },
        }
        _save_checkpoint_json("results.json", results)
        _save_checkpoint_json("cox_rows.json", cox_rows)
        _write_intermediate_outputs(results, cox_rows, bootstrap_cindex_rows)

    bs_comb = _load_checkpoint_array("bs_comb.npy")
    if bs_comb is None:
        print("  Bootstrapping C-index (B=5000)...")
        bs_comb = _bootstrap_cindex_cox(df_comb, y_surv, seed=RANDOM_SEED + 1)
        _save_checkpoint_array("bs_comb.npy", bs_comb)
    bs_summary_comb = _bootstrap_summary(bs_comb, "Combined Cox")
    bootstrap_cindex_rows = _upsert_bootstrap_row(bootstrap_cindex_rows, bs_summary_comb)
    _save_checkpoint_json("bootstrap_rows.json", bootstrap_cindex_rows)
    _write_intermediate_outputs(results, cox_rows, bootstrap_cindex_rows)

    # ================================================================
    #  D) Random Survival Forest
    # ================================================================
    print("\n" + "-" * 40)
    print("Model D: Random Survival Forest (combined features)")
    print("-" * 40)

    rsf = RandomSurvivalForest(
        n_estimators=1000,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=RANDOM_SEED,
        n_jobs=N_JOBS,
    )
    rsf.fit(X_comb, y_surv)
    cindex_rsf = _cindex_sksurv(rsf, X_comb, y_surv)
    print(f"  C-index (in-sample): {cindex_rsf:.4f}")

    results["rsf"] = {
        "cindex": cindex_rsf,
        "n_features": X_comb.shape[1],
        "n_estimators": 1000,
    }
    _save_checkpoint_json("results.json", results)
    _write_intermediate_outputs(results, cox_rows, bootstrap_cindex_rows)

    bs_rsf = _load_checkpoint_array("bs_rsf.npy")
    if bs_rsf is None and not args.skip_missing_rsf_bootstrap:
        print(f"  Bootstrapping C-index (B={N_BOOTSTRAP_RSF})...")
        bs_rsf = _bootstrap_cindex_rsf(
            X_comb, y_surv, n_iter=N_BOOTSTRAP_RSF, seed=RANDOM_SEED,
            n_estimators=1000, min_samples_split=10,
            min_samples_leaf=int(np.ceil(493 * 0.01)),
            max_features="sqrt", random_state=RANDOM_SEED, n_jobs=1,
        )
        _save_checkpoint_array("bs_rsf.npy", bs_rsf)

    if bs_rsf is None:
        bs_summary_rsf = {
            "model": "RSF",
            "n_bootstrap": 0,
            "cindex_mean": float("nan"),
            "cindex_median": float("nan"),
            "cindex_std": float("nan"),
            "ci_95_lower": float("nan"),
            "ci_95_upper": float("nan"),
            "note": "RSF bootstrap not completed; summary omitted to avoid rerunning the multi-hour bootstrap.",
        }
    else:
        bs_summary_rsf = _bootstrap_summary(bs_rsf, "RSF")
    bootstrap_cindex_rows = _upsert_bootstrap_row(bootstrap_cindex_rows, bs_summary_rsf)
    _save_checkpoint_json("bootstrap_rows.json", bootstrap_cindex_rows)
    _write_intermediate_outputs(results, cox_rows, bootstrap_cindex_rows)

    # -- C-index comparison summary --
    cindex_summary = {
        "Clinical-molecular Cox": cindex_clin,
        "Spatial Cox": cindex_spat,
        "Combined Cox": cindex_comb,
        "RSF (combined)": cindex_rsf,
    }
    results["cindex_summary"] = cindex_summary

    # ================================================================
    #  Integrated Brier Score
    # ================================================================
    print("\n" + "-" * 40)
    print("Integrated Brier Score")
    print("-" * 40)

    times = np.linspace(
        df["OS_months"].min() + 0.1, df["OS_months"].max() - 0.1, 100
    )

    ibs_results: dict[str, float] = {}

    # Helper to compute IBS for a model given survival function predictions
    def compute_ibs(surv_fn, X, label: str) -> float:
        try:
            pred_surv = surv_fn(X, times)
            ibs = integrated_brier_score(y_surv, y_surv, pred_surv, times)
            ibs_results[label] = float(ibs)
            print(f"  {label}: IBS = {ibs:.4f}")
            return float(ibs)
        except Exception as e:
            print(f"  {label}: IBS failed ({e})")
            ibs_results[label] = float("nan")
            return float("nan")

    # For Cox models we need to predict survival function via sksurv
    from sksurv.linear_model import CoxPHSurvivalAnalysis

    # Clinical-molecular Cox via sksurv
    cox_clin_sksurv = CoxPHSurvivalAnalysis()
    cox_clin_sksurv.fit(X_clin, y_surv)
    compute_ibs(
        lambda x, t: _evaluate_survival_functions_at_times(
            cox_clin_sksurv.predict_survival_function(x), t
        ),
        X_clin,
        "Clinical-molecular Cox",
    )

    # Spatial Cox via sksurv
    cox_spat_sksurv = CoxPHSurvivalAnalysis()
    cox_spat_sksurv.fit(X_spat, y_surv)
    compute_ibs(
        lambda x, t: _evaluate_survival_functions_at_times(
            cox_spat_sksurv.predict_survival_function(x), t
        ),
        X_spat,
        "Spatial Cox",
    )

    # Combined Cox via sksurv
    cox_comb_sksurv = CoxPHSurvivalAnalysis()
    cox_comb_sksurv.fit(X_comb, y_surv)
    compute_ibs(
        lambda x, t: _evaluate_survival_functions_at_times(
            cox_comb_sksurv.predict_survival_function(x), t
        ),
        X_comb,
        "Combined Cox",
    )

    # RSF
    compute_ibs(
        lambda x, t: _evaluate_survival_functions_at_times(
            rsf.predict_survival_function(x), t
        ),
        X_comb,
        "RSF",
    )

    results["integrated_brier_score"] = ibs_results

    # ================================================================
    #  Time-dependent AUC at 12, 24, 36 months
    # ================================================================
    print("\n" + "-" * 40)
    print("Time-dependent AUC")
    print("-" * 40)

    auc_times = [12.0, 24.0, 36.0]
    auc_results: dict[str, dict[str, float]] = {}

    for label, risk_fn, X_eval in [
        ("Clinical-molecular Cox",
         lambda x: cox_clin_sksurv.predict(x), X_clin),
        ("Spatial Cox",
         lambda x: cox_spat_sksurv.predict(x), X_spat),
        ("Combined Cox",
         lambda x: cox_comb_sksurv.predict(x), X_comb),
        ("RSF",
         lambda x: rsf.predict(x), X_comb),
    ]:
        try:
            risk = risk_fn(X_eval)
            auc_out = cumulative_dynamic_auc(y_surv, y_surv, risk, auc_times)
            auc_vals = dict(zip([f"{int(t)}m" for t in auc_times],
                                [float(v) for v in auc_out[0]]))
            auc_results[label] = auc_vals
            print(f"  {label}: {auc_vals}")
        except Exception as e:
            print(f"  {label}: AUC failed ({e})")
            auc_results[label] = {f"{int(t)}m": float("nan") for t in auc_times}

    results["time_dependent_auc"] = auc_results

    # ================================================================
    #  Proportional Hazards Assumption (Schoenfeld residuals)
    # ================================================================
    print("\n" + "-" * 40)
    print("Proportional Hazards Assumption (Schoenfeld residuals)")
    print("-" * 40)

    ph_results: dict[str, object] = {}

    for label, cph, X_df in [
        ("Clinical-molecular Cox", cph_clin, X_clin),
        ("Spatial Cox", cph_spat, X_spat),
        ("Combined Cox", cph_comb, X_comb),
    ]:
        df_for_test = X_df.copy()
        df_for_test["OS_months"] = df["OS_months"].values
        df_for_test["event"] = df["event"].values

        try:
            test_result = proportional_hazard_test(cph, df_for_test)
            global_test_stat = float(test_result.summary["test_statistic"].sum())
            global_df = int(len(test_result.summary.index))
            global_p = float(chi2.sf(global_test_stat, global_df))
            ph_results[label] = {
                "global_test_p": global_p,
                "global_test_statistic": global_test_stat,
                "global_df": global_df,
                "per_variable": {},
            }
            print(f"  {label}: global PH test p = {global_p:.6g}")
            for var_name in test_result.summary.index:
                if var_name != "T":
                    var_p = float(test_result.summary.loc[var_name, "p"])
                    ph_results[label]["per_variable"][var_name] = var_p
                    print(f"    {var_name}: p = {var_p:.6g}")
        except Exception as e:
            print(f"  {label}: PH test failed ({e})")
            ph_results[label] = {
                "global_test_p": float("nan"),
                "per_variable": {},
            }

    results["proportional_hazards"] = ph_results

    # ================================================================
    #  Kaplan-Meier curves - low vs high risk (median risk split)
    # ================================================================
    print("\n" + "-" * 40)
    print("Kaplan-Meier curves (median risk split)")
    print("-" * 40)

    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    # Use the Combined Cox model for risk stratification
    risk_comb = cph_comb.predict_partial_hazard(X_comb).values
    median_risk = float(np.median(risk_comb))
    groups = (risk_comb > median_risk).astype(int)
    labels_groups = ["Low risk", "High risk"]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#2ecc71", "#e74c3c"]
    for g in [0, 1]:
        mask = groups == g
        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=df.loc[mask, "OS_months"],
            event_observed=df.loc[mask, "event"],
            label=labels_groups[g],
        )
        kmf.plot_survival_function(ax=ax, color=colors[g], linewidth=2)

    low_mask = groups == 0
    high_mask = groups == 1
    lr = logrank_test(
        df.loc[low_mask, "OS_months"],
        df.loc[high_mask, "OS_months"],
        event_observed_A=df.loc[low_mask, "event"],
        event_observed_B=df.loc[high_mask, "event"],
    )
    ax.set_title(
        f"Kaplan-Meier by Predicted Risk (Combined Cox Model)\n"
        f"Log-rank p = {lr.p_value:.6g}",
        fontsize=14,
    )
    ax.set_xlabel("Time (months)", fontsize=12)
    ax.set_ylabel("Survival probability", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "km_risk_groups.png", dpi=150)
    fig.savefig(OUTPUT_DIR / "km_risk_groups.pdf")
    plt.close(fig)
    print("  Saved km_risk_groups.png/pdf")

    # ================================================================
    #  Time-dependent AUC figure
    # ================================================================
    print("\n" + "-" * 40)
    print("Time-dependent AUC figure")
    print("-" * 40)

    fig, ax = plt.subplots(figsize=(10, 7))
    markers = ["o", "s", "D", "^"]
    colors_auc = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6"]
    for i, (label, auc_vals) in enumerate(auc_results.items()):
        t_vals = [int(k.replace("m", "")) for k in auc_vals.keys()]
        a_vals = list(auc_vals.values())
        ax.plot(t_vals, a_vals, marker=markers[i], color=colors_auc[i],
                label=label, linewidth=2, markersize=8)
    ax.set_xlabel("Time (months)", fontsize=12)
    ax.set_ylabel("Time-dependent AUC", fontsize=12)
    ax.set_title("Time-dependent AUC at 12, 24, 36 months", fontsize=14)
    ax.set_ylim(0.3, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "time_dependent_auc.png", dpi=150)
    plt.close(fig)
    print("  Saved time_dependent_auc.png")

    # ================================================================
    #  Write outputs
    # ================================================================
    print("\n" + "-" * 40)
    print("Writing outputs")
    print("-" * 40)

    # cox_results.csv
    pd.DataFrame(cox_rows).to_csv(
        OUTPUT_DIR / "cox_results.csv", index=False
    )

    # cox_results.json
    _save_json(results, OUTPUT_DIR / "cox_results.json")

    # cindex_bootstrap.csv
    pd.DataFrame(bootstrap_cindex_rows).to_csv(
        OUTPUT_DIR / "cindex_bootstrap.csv", index=False
    )

    # cindex_bootstrap.json
    _save_json(
        {r["model"]: r for r in bootstrap_cindex_rows},
        OUTPUT_DIR / "cindex_bootstrap.json",
    )

    # survival_report.md
    report = _build_report(df, results, bootstrap_cindex_rows, cox_rows)
    (OUTPUT_DIR / "survival_report.md").write_text(report, encoding="utf-8")
    print("  Saved survival_report.md")

    print("\n" + "=" * 60)
    print("  Survival analysis complete!")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
