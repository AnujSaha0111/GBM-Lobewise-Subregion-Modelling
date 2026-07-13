#!/usr/bin/env python3
"""Reviewer Experiment: Regularized Feature Selection for Combined Model

Addresses Reviewer Comment #2. Tests whether feature selection / regularization
resolves the combined model's performance degradation.

Models:
  A: L1 Logistic Regression feature selection -> RBF SVM
  B: Elastic Net Logistic Regression feature selection -> RBF SVM
  C: Recursive Feature Elimination (RFECV) -> RBF SVM
  D: Mutual Information (SelectKBest) -> RBF SVM (K in {10,20,30,40,50})
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    RFECV,
    SelectFromModel,
    SelectKBest,
    mutual_info_classif,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    ParameterGrid,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = ROOT / "outputs" / "multimodal_lobewise" / "merged_features_with_metadata.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_BOOTSTRAP = 5000
N_SPLITS = 5
N_REPEATS = 10

SVM_GRID = {
    "svm__C": [0.01, 0.1, 1, 10, 100],
    "svm__gamma": ["scale", 0.0001, 0.001, 0.01, 0.1],
}

SPATIAL_PREFIXES = ("T1_", "T2_", "T1GD_", "FLAIR_")
META_COLS = ["age", "sex", "idh", "mgmt", "who_grade", "eor"]
MAX_ITER = 10000
T0 = time.time()


def log(msg):
    elapsed = time.time() - T0
    print(f"[{elapsed:6.0f}s] {msg}", flush=True)


# ── Data ───────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(INPUT_CSV)
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    y = df["risk_label"].astype(int)

    spatial_cols = [c for c in df.columns if c.startswith(SPATIAL_PREFIXES)]
    meta_cols = [c for c in META_COLS if c in df.columns]
    combined_cols = spatial_cols + meta_cols

    X = df[combined_cols].apply(pd.to_numeric, errors="coerce")

    X_tr, X_te, y_tr, y_te, ids_tr, ids_te = train_test_split(
        X, y, df["patient_id"],
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    log(f"Split: {len(X_tr)} train / {len(X_te)} test, {X.shape[1]} features")
    return X_tr, X_te, y_tr, y_te, combined_cols, spatial_cols, meta_cols


# ── Helpers ────────────────────────────────────────────────────────────

def make_svm_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True,
                     class_weight="balanced", random_state=RANDOM_STATE)),
    ])


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred).tolist()
    return {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": cm,
    }, y_proba


def bootstrap_auc_ci(y_true, y_prob, n_boot=N_BOOTSTRAP, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true.values[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true.values[idx], y_prob[idx]))
    aucs = np.array(aucs)
    return {
        "mean_auc": float(np.mean(aucs)),
        "ci_lower": float(np.percentile(aucs, 2.5)),
        "ci_upper": float(np.percentile(aucs, 97.5)),
    }


def repeated_stability(build_fn, X, y):
    rskf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE,
    )
    rows = []
    for ri, (tri, tei) in enumerate(rskf.split(X, y)):
        pipe = build_fn()
        pipe.fit(X.iloc[tri], y.iloc[tri])
        yp = pipe.predict(X.iloc[tei])
        ypb = pipe.predict_proba(X.iloc[tei])[:, 1]
        rows.append({
            "roc_auc": float(roc_auc_score(y.iloc[tei], ypb)),
            "accuracy": float(accuracy_score(y.iloc[tei], yp)),
            "precision": float(precision_score(y.iloc[tei], yp, zero_division=0)),
            "recall": float(recall_score(y.iloc[tei], yp, zero_division=0)),
            "f1_score": float(f1_score(y.iloc[tei], yp, zero_division=0)),
        })
    rdf = pd.DataFrame(rows)
    summary = {}
    for c in ["roc_auc", "accuracy", "precision", "recall", "f1_score"]:
        v = rdf[c].values
        summary[c] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))}
    return summary


def get_selected_features(model, feature_names):
    if "fs" not in model.named_steps:
        return list(feature_names)
    mask = model.named_steps["fs"].get_support()
    return list(np.array(feature_names)[mask])


# ── Model pipelines ────────────────────────────────────────────────────

def pipe_a():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("fs", SelectFromModel(LogisticRegression(
            penalty="l1", solver="liblinear", class_weight="balanced",
            max_iter=MAX_ITER))),
        ("svm", SVC(kernel="rbf", probability=True,
                     class_weight="balanced", random_state=RANDOM_STATE)),
    ])


def pipe_b():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("fs", SelectFromModel(LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5,
            class_weight="balanced", max_iter=MAX_ITER))),
        ("svm", SVC(kernel="rbf", probability=True,
                     class_weight="balanced", random_state=RANDOM_STATE)),
    ])


def pipe_c():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("fs", RFECV(
            estimator=LogisticRegression(
                penalty="l1", solver="liblinear", class_weight="balanced",
                max_iter=MAX_ITER),
            step=3,
            cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
            scoring="roc_auc",
            min_features_to_select=2,
            n_jobs=1,
        )),
        ("svm", SVC(kernel="rbf", probability=True,
                     class_weight="balanced", random_state=RANDOM_STATE)),
    ])


def pipe_d(k=20):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("fs", SelectKBest(score_func=mutual_info_classif, k=k)),
        ("svm", SVC(kernel="rbf", probability=True,
                     class_weight="balanced", random_state=RANDOM_STATE)),
    ])


# ── Parameter grids ────────────────────────────────────────────────────

GRID_A = {
    "fs__estimator__C": [0.01, 0.1, 1, 10],
    **SVM_GRID,
}

GRID_B = {
    "fs__estimator__C": [0.01, 0.1, 1, 10],
    "fs__estimator__l1_ratio": [0.2, 0.5, 0.8],
    **SVM_GRID,
}

GRID_C = SVM_GRID

GRID_D = SVM_GRID


# ── Feature selection frequency (LR params only) ──────────────────────

def feature_freq(build_fn, lr_grid, X_tr, y_tr, feature_names):
    freq = {f: 0 for f in feature_names}
    total = 0
    for params in ParameterGrid(lr_grid):
        pipe = build_fn()
        pipe.set_params(**params)
        try:
            pipe.fit(X_tr, y_tr)
            for f in get_selected_features(pipe, feature_names):
                freq[f] += 1
            total += 1
        except Exception:
            continue
    return freq, total


# ── Core run function ──────────────────────────────────────────────────

def run_model(name, build_fn, param_grid, X_tr, y_tr, X_te, y_te,
              feature_names, X_full, y_full):
    log(f"--- {name}: GridSearchCV ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        estimator=build_fn(), param_grid=param_grid,
        scoring="roc_auc", cv=skf, n_jobs=-1, verbose=0,
    )
    gs.fit(X_tr, y_tr)
    best = gs.best_estimator_
    log(f"  Best params: {gs.best_params_}, CV AUC: {gs.best_score_:.4f}")

    selected = get_selected_features(best, feature_names)
    log(f"  Selected features: {len(selected)}/{len(feature_names)}")

    metrics, y_proba = evaluate(best, X_te, y_te)
    log(f"  Test ROC-AUC: {metrics['roc_auc']:.4f}")

    log(f"  Bootstrap CI ...")
    ci = bootstrap_auc_ci(y_te, y_proba)
    log(f"  AUC: {ci['mean_auc']:.4f} [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")

    fpr, tpr, _ = roc_curve(y_te.values, y_proba)

    log(f"  Repeated stability ...")
    cv = repeated_stability(build_fn, X_full, y_full)
    log(f"  CV ROC-AUC: {cv['roc_auc']['mean']:.4f} +/- {cv['roc_auc']['std']:.4f}")

    return {
        "model_name": name,
        "best_params": gs.best_params_,
        "best_cv_auc": gs.best_score_,
        "n_selected": len(selected),
        "selected_features": selected,
        "metrics": metrics,
        "y_proba": y_proba,
        "bootstrap": ci,
        "roc": {"fpr": fpr, "tpr": tpr, "auc": metrics["roc_auc"]},
        "cv": cv,
    }


# ── Main ───────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("  REVIEWER EXPERIMENT: REGULARIZED COMBINED MODEL")
    log("=" * 60)

    X_tr, X_te, y_tr, y_te, combined_cols, spatial_cols, meta_cols = load_data()
    X_full = pd.concat([X_tr, X_te])
    y_full = pd.concat([y_tr, y_te])

    results = {}

    # Clinical-only baseline
    log("\n=== Clinical Only (6 features) ===")
    X_tr_c = X_tr[meta_cols]; X_te_c = X_te[meta_cols]; X_full_c = X_full[meta_cols]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(make_svm_pipeline(), SVM_GRID, scoring="roc_auc",
                      cv=skf, n_jobs=-1, verbose=0)
    gs.fit(X_tr_c, y_tr)
    best_c = gs.best_estimator_
    m, yp = evaluate(best_c, X_te_c, y_te)
    ci = bootstrap_auc_ci(y_te, yp)
    fpr, tpr, _ = roc_curve(y_te.values, yp)
    cv = repeated_stability(make_svm_pipeline, X_full_c, y_full)
    results["clinical"] = {
        "model_name": "Clinical Only", "best_params": gs.best_params_,
        "best_cv_auc": gs.best_score_, "n_selected": len(meta_cols),
        "selected_features": meta_cols, "metrics": m, "y_proba": yp,
        "bootstrap": ci, "roc": {"fpr": fpr, "tpr": tpr, "auc": m["roc_auc"]},
        "cv": cv,
    }
    log(f"  Test ROC-AUC: {m['roc_auc']:.4f}, CV: {cv['roc_auc']['mean']:.4f}")

    # Combined baseline (no FS)
    log("\n=== Combined Baseline (70 features, no FS) ===")
    results["baseline"] = run_model(
        "Combined (no FS)", make_svm_pipeline, SVM_GRID,
        X_tr, y_tr, X_te, y_te, combined_cols, X_full, y_full,
    )

    # Model A: L1 LR
    log("\n=== Model A: L1 LR Feature Selection ===")
    results["A"] = run_model(
        "A_L1_LR", pipe_a, GRID_A,
        X_tr, y_tr, X_te, y_te, combined_cols, X_full, y_full,
    )

    # Model B: Elastic Net
    log("\n=== Model B: Elastic Net Feature Selection ===")
    results["B"] = run_model(
        "B_ElasticNet", pipe_b, GRID_B,
        X_tr, y_tr, X_te, y_te, combined_cols, X_full, y_full,
    )

    # Model C: RFECV
    log("\n=== Model C: RFECV ===")
    results["C"] = run_model(
        "C_RFECV", pipe_c, GRID_C,
        X_tr, y_tr, X_te, y_te, combined_cols, X_full, y_full,
    )

    # Model D: MI K-sweep
    log("\n=== Model D: Mutual Information K-sweep ===")
    results["D"] = []
    for k in [10, 20, 30, 40, 50]:
        log(f"\n--- K={k} ---")
        fn = lambda kk=k: pipe_d(kk)
        g = {**SVM_GRID, "fs__k": [k]}
        r = run_model(f"D_K{k}", fn, g,
                       X_tr, y_tr, X_te, y_te, combined_cols, X_full, y_full)
        results["D"].append(r)

    # Feature selection frequency
    log("\n=== Feature Selection Frequency ===")
    lr_a_grid = {"fs__estimator__C": [0.01, 0.1, 1, 10]}
    freq_a, tot_a = feature_freq(pipe_a, lr_a_grid, X_tr, y_tr, combined_cols)
    pd.DataFrame([
        {"feature": f, "count": c, "rate": c / tot_a if tot_a else 0}
        for f, c in sorted(freq_a.items(), key=lambda x: -x[1])
    ]).to_csv(OUTPUT_DIR / "feature_freq_A.csv", index=False)
    log(f"  Model A frequency: {tot_a} configs")

    lr_b_grid = {"fs__estimator__C": [0.01, 0.1, 1, 10], "fs__estimator__l1_ratio": [0.2, 0.5, 0.8]}
    freq_b, tot_b = feature_freq(pipe_b, lr_b_grid, X_tr, y_tr, combined_cols)
    pd.DataFrame([
        {"feature": f, "count": c, "rate": c / tot_b if tot_b else 0}
        for f, c in sorted(freq_b.items(), key=lambda x: -x[1])
    ]).to_csv(OUTPUT_DIR / "feature_freq_B.csv", index=False)
    log(f"  Model B frequency: {tot_b} configs")

    # ── Save results ──
    log("\n=== Saving Results ===")

    # Comparison table
    all_list = [results["clinical"], results["baseline"],
                results["A"], results["B"], results["C"]] + results["D"]
    rows = []
    for r in all_list:
        rows.append({
            "Model": r["model_name"],
            "Features": r["n_selected"],
            "ROC-AUC": f"{r['metrics']['roc_auc']:.4f}",
            "95% CI": f"[{r['bootstrap']['ci_lower']:.4f}, {r['bootstrap']['ci_upper']:.4f}]",
            "CV ROC-AUC": f"{r['cv']['roc_auc']['mean']:.4f} +/- {r['cv']['roc_auc']['std']:.4f}",
            "Accuracy": f"{r['metrics']['accuracy']:.4f}",
            "Precision": f"{r['metrics']['precision']:.4f}",
            "Recall": f"{r['metrics']['recall']:.4f}",
            "F1": f"{r['metrics']['f1_score']:.4f}",
        })
    cdf = pd.DataFrame(rows)
    cdf.to_csv(OUTPUT_DIR / "comparison_table.csv", index=False)
    log("  Wrote comparison_table.csv")

    # Per-model JSON
    for key in ["clinical", "baseline", "A", "B", "C"]:
        r = results[key]
        out = {k: v for k, v in r.items() if k != "y_proba"}
        out["test_proba"] = r["y_proba"].tolist()
        (OUTPUT_DIR / f"result_{r['model_name'].replace(' ', '_')}.json").write_text(
            json.dumps(out, indent=2, default=str), encoding="utf-8")
    d_out = []
    for r in results["D"]:
        o = {k: v for k, v in r.items() if k != "y_proba"}
        o["test_proba"] = r["y_proba"].tolist()
        d_out.append(o)
    (OUTPUT_DIR / "result_D_all.json").write_text(
        json.dumps(d_out, indent=2, default=str), encoding="utf-8")

    # Selected features CSV
    feat_rows = []
    for key in ["A", "B", "C"]:
        r = results[key]
        for f in r["selected_features"]:
            feat_rows.append({"model": r["model_name"], "feature": f})
    for r in results["D"]:
        for f in r["selected_features"]:
            feat_rows.append({"model": r["model_name"], "feature": f})
    pd.DataFrame(feat_rows).to_csv(OUTPUT_DIR / "selected_features.csv", index=False)

    # ── ROC Curves ──
    fig, ax = plt.subplots(figsize=(10, 8))
    clrs = {
        "Clinical Only": "green", "Combined (no FS)": "red",
        "A_L1_LR": "blue", "B_ElasticNet": "darkorange", "C_RFECV": "purple",
    }
    ls_map = {"Clinical Only": "--", "Combined (no FS)": "--"}

    for key in ["clinical", "baseline", "A", "B", "C"]:
        r = results[key]
        ax.plot(r["roc"]["fpr"], r["roc"]["tpr"],
                color=clrs.get(r["model_name"], "gray"), lw=2,
                ls=ls_map.get(r["model_name"], "-"),
                label=f"{r['model_name']} ({r['n_selected']}f, AUC={r['roc']['auc']:.3f})")

    best_d = max(results["D"], key=lambda x: x["metrics"]["roc_auc"])
    ax.plot(best_d["roc"]["fpr"], best_d["roc"]["tpr"],
            color="brown", lw=2, label=f"{best_d['model_name']} ({best_d['n_selected']}f, AUC={best_d['roc']['auc']:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, ls="--", label="Chance")
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves - Regularized Combined Models vs Baselines", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)

    # Model D sensitivity
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = [r["n_selected"] for r in results["D"]]
    taucs = [r["metrics"]["roc_auc"] for r in results["D"]]
    cvaucs = [r["cv"]["roc_auc"]["mean"] for r in results["D"]]
    ax.plot(ks, taucs, "o-", color="brown", lw=2, label="Test ROC-AUC")
    ax.plot(ks, cvaucs, "s--", color="sienna", lw=2, label="CV ROC-AUC")
    clin_auc = results["clinical"]["metrics"]["roc_auc"]
    bas_auc = results["baseline"]["metrics"]["roc_auc"]
    ax.axhline(clin_auc, color="green", ls=":", lw=1.5, label=f"Clinical ({clin_auc:.3f})")
    ax.axhline(bas_auc, color="red", ls=":", lw=1.5, label=f"Combined no FS ({bas_auc:.3f})")
    ax.set_xlabel("K (selected features)"); ax.set_ylabel("ROC-AUC")
    ax.set_title("Model D: Mutual Information - K Sensitivity")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    for x, y in zip(ks, taucs):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_D_sensitivity.png", dpi=150)
    plt.close(fig)

    # ── Report ──
    best = max(all_list, key=lambda x: x["metrics"]["roc_auc"])
    clin_auc = results["clinical"]["metrics"]["roc_auc"]
    bas_auc = results["baseline"]["metrics"]["roc_auc"]

    report = f"""# Reviewer Experiment: Regularized Combined Model

## Objective
Address Reviewer Comment #2: combined model (70 features, AUC=0.703) underperforms
clinical-only (6 features, AUC=0.772). Test whether feature selection / regularization resolves this.

## Protocol
- Identical split: seed=42, test_size=0.20, stratified (394 train / 99 test)
- Identical preprocessing: SimpleImputer(median) -> StandardScaler -> SVM(RBF, balanced)
- Identical GridSearchCV: 5-fold stratified, scoring=roc_auc
- Identical repeated stability: RepeatedStratifiedKFold (5x10 = 50 runs)
- Bootstrap CI: 5000 resamples

## Results

{cdf.to_markdown(index=False)}

## Best Overall Model

- **{best['model_name']}**
- Retained features: {best['n_selected']}
- Test ROC-AUC: {best['metrics']['roc_auc']:.4f}
- 95% CI: [{best['bootstrap']['ci_lower']:.4f}, {best['bootstrap']['ci_upper']:.4f}]
- CV ROC-AUC: {best['cv']['roc_auc']['mean']:.4f} +/- {best['cv']['roc_auc']['std']:.4f}

## Comparison with Baselines

| Comparison | Delta AUC |
|---|---|
| Best vs Clinical-Only ({clin_auc:.4f}) | {best['metrics']['roc_auc'] - clin_auc:+.4f} |
| Best vs Combined-no-FS ({bas_auc:.4f}) | {best['metrics']['roc_auc'] - bas_auc:+.4f} |

## Conclusion

"""
    if best["metrics"]["roc_auc"] > clin_auc:
        report += "Feature selection **resolves** the combined model's degradation. The regularized model outperforms clinical-only.\n"
    elif best["metrics"]["roc_auc"] > bas_auc:
        report += "Feature selection improves the combined model but does **not** fully close the gap with clinical-only.\n"
    else:
        report += "Feature selection does **not** resolve the degradation. Clinical-only remains superior.\n"
        report += "The issue is not merely overfitting from dimensionality; spatial features add noise.\n"

    report += f"\n*Generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*\n"
    (OUTPUT_DIR / "report.md").write_text(report, encoding="utf-8")

    log(f"\n{'='*60}")
    log("  FINAL RESULTS")
    log(f"  Best model:         {best['model_name']}")
    log(f"  Retained features:  {best['n_selected']}")
    log(f"  Test ROC-AUC:       {best['metrics']['roc_auc']:.4f}")
    log(f"  CV ROC-AUC:         {best['cv']['roc_auc']['mean']:.4f} +/- {best['cv']['roc_auc']['std']:.4f}")
    log(f"  Clinical-only:      {clin_auc:.4f}")
    log(f"  Combined no-FS:     {bas_auc:.4f}")
    log(f"  Outputs: {OUTPUT_DIR}")
    log(f"  Total time: {time.time()-T0:.0f}s")
    log(f"{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
