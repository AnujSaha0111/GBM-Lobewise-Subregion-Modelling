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
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
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
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ──
ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "outputs" / "multimodal_lobewise" / "merged_features_with_metadata.csv"
OUTPUT_DIR = ROOT / "outputs" / "baselines"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_BOOTSTRAP = 5000
N_SPLITS = 5
N_REPEATS = 5

SPATIAL_FEATURE_PREFIXES = ("T1_", "T2_", "T1GD_", "FLAIR_")
METADATA_FEATURE_COLS = ["age", "sex", "idh", "mgmt", "who_grade", "eor"]

FEATURE_SETS = {
    "spatial": {"name": "Spatial Only", "key": "spatial"},
    "clinical": {"name": "Clinical + Molecular", "key": "clinical"},
    "combined": {"name": "Combined", "key": "combined"},
}

MODEL_CONFIGS = [
    {
        "name": "Logistic Regression",
        "key": "lr",
        "step": "clf",
        "color": "#1b9e77",
        "estimator": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE, class_weight="balanced"),
        "param_grid": {
            "clf__C": [0.01, 0.1, 1, 10],
        },
    },
    {
        "name": "Random Forest",
        "key": "rf",
        "step": "clf",
        "color": "#d95f02",
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        "param_grid": {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [3, 5, 10, None],
        },
    },
    {
        "name": "XGBoost",
        "key": "xgb",
        "step": "clf",
        "color": "#7570b3",
        "estimator": XGBClassifier(
            eval_metric="logloss", random_state=RANDOM_STATE,
            scale_pos_weight=1.0,
        ),
        "param_grid": {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [3, 5],
            "clf__learning_rate": [0.03, 0.05, 0.1],
        },
    },
    {
        "name": "SVM (RBF)",
        "key": "svm",
        "step": "svc",
        "color": "#e7298a",
        "estimator": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
        "param_grid": {
            "svc__C": [0.01, 0.1, 1, 10, 100],
            "svc__gamma": ["scale", 0.001, 0.01, 0.1, 1],
        },
    },
]

# ── Data ──
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
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
        )
        X_train_dict[key] = X_tr
        X_test_dict[key] = X_te
        y_train_dict[key] = y_tr
        y_test_dict[key] = y_te
        if y_test_vals is None:
            y_test_vals = y_te.values

    return feature_map, X_train_dict, X_test_dict, y_train_dict, y_test_dict, y_test_vals


# ── Evaluation helpers ──
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
    }, y_proba, y_pred


def bootstrap_auc_ci(y_true, y_prob, n_bootstraps, random_state):
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    boot_aucs = []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        pb = y_prob[idx]
        if len(np.unique(yb)) < 2:
            continue
        boot_aucs.append(roc_auc_score(yb, pb))
    boot_aucs = np.array(boot_aucs)
    return {
        "mean_auc": float(np.mean(boot_aucs)),
        "ci_lower": float(np.percentile(boot_aucs, 2.5)),
        "ci_upper": float(np.percentile(boot_aucs, 97.5)),
        "n_valid_bootstrap": len(boot_aucs),
    }


def repeated_cv(X, y, mc, best_params, n_splits, n_repeats, random_state, step_name="clf"):
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state,
    )
    rows = []
    total = n_splits * n_repeats
    for repeat_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        if (repeat_idx + 1) % 10 == 0:
            print(f"      CV fold {repeat_idx+1}/{total} ...")
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (step_name, mc["estimator"]),
        ])
        pipe.set_params(**best_params)
        pipe.fit(X_tr, y_tr)

        y_pr = pipe.predict(X_te)
        y_prb = pipe.predict_proba(X_te)[:, 1]
        rows.append({
            "repeat": repeat_idx + 1,
            "fold": (repeat_idx % n_splits) + 1,
            "roc_auc": float(roc_auc_score(y_te, y_prb)),
            "accuracy": float(accuracy_score(y_te, y_pr)),
            "precision": float(precision_score(y_te, y_pr, zero_division=0)),
            "recall": float(recall_score(y_te, y_pr, zero_division=0)),
            "f1_score": float(f1_score(y_te, y_pr, zero_division=0)),
        })

    results_df = pd.DataFrame(rows)
    summary = {}
    for col in ["roc_auc", "accuracy", "precision", "recall", "f1_score"]:
        vals = results_df[col].values
        summary[col] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return summary, results_df


# ── Main ──
def main():
    print("=" * 60)
    print("  BASELINE CLASSIFIER COMPARISON")
    print("=" * 60)

    feature_map, X_tr, X_te, y_tr, y_te, y_test_vals = load_and_split()
    print(f"\nTest set: {len(y_test_vals)} samples ({y_test_vals.sum()} high-risk)")

    all_results = []
    roc_data = []
    stored_best_params = {}

    for fs_key in ["spatial", "clinical", "combined"]:
        fs_info = FEATURE_SETS[fs_key]
        print(f"\n{'=' * 50}")
        print(f"  Feature Set: {fs_info['name']}")
        print(f"{'=' * 50}")

        for mc in MODEL_CONFIGS:
            print(f"\n  --- {mc['name']} ---")

            step_name = mc.get("step", "clf")
            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (step_name, mc["estimator"]),
            ])

            if mc["param_grid"] is not None:
                gs_cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
                gs = GridSearchCV(
                    pipeline, mc["param_grid"],
                    scoring="roc_auc",
                    cv=gs_cv,
                    n_jobs=-1, verbose=0,
                )
                gs.fit(X_tr[fs_key], y_tr[fs_key])
                best_model = gs.best_estimator_
                best_params = gs.best_params_
                best_cv_auc = gs.best_score_
                print(f"    Best CV AUC: {best_cv_auc:.4f}  params: {best_params}")
            else:
                pipeline.set_params(**mc["fixed_params"])
                pipeline.fit(X_tr[fs_key], y_tr[fs_key])
                best_model = pipeline
                best_params = mc["fixed_params"]
                best_cv_auc = None
                print(f"    Fixed params: {best_params}")

            stored_best_params[(fs_key, mc["key"])] = best_params

            metrics, y_proba, y_pred = evaluate(best_model, X_te[fs_key], y_te[fs_key])
            print(f"    Test ROC-AUC: {metrics['roc_auc']:.4f}  Acc: {metrics['accuracy']:.4f}  F1: {metrics['f1_score']:.4f}")

            boot = bootstrap_auc_ci(y_te[fs_key].values, y_proba, N_BOOTSTRAP, RANDOM_STATE)
            print(f"    Bootstrap 95% CI: [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]")

            cv_summary, cv_df = repeated_cv(
                pd.concat([X_tr[fs_key], X_te[fs_key]]),
                pd.concat([y_tr[fs_key], y_te[fs_key]]),
                mc, best_params,
                N_SPLITS, N_REPEATS, RANDOM_STATE,
                step_name=step_name,
            )
            cv_roc = cv_summary["roc_auc"]
            print(f"    CV ROC-AUC: {cv_roc['mean']:.4f} +- {cv_roc['std']:.4f}")

            result_row = {
                "feature_set": fs_info["name"],
                "model": mc["name"],
                "roc_auc": metrics["roc_auc"],
                "ci_lower": boot["ci_lower"],
                "ci_upper": boot["ci_upper"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "cv_auc_mean": cv_roc["mean"],
                "cv_auc_std": cv_roc["std"],
                "best_params": str(best_params),
            }
            all_results.append(result_row)

            fpr, tpr, _ = roc_curve(y_te[fs_key].values, y_proba)
            roc_data.append({
                "feature_set": fs_info["name"],
                "model": mc["name"],
                "model_key": mc["key"],
                "color": mc["color"],
                "fpr": fpr,
                "tpr": tpr,
                "auc": metrics["roc_auc"],
            })

            # Save per-model per-feature-set metrics
            model_dir = OUTPUT_DIR / mc["key"]
            model_dir.mkdir(exist_ok=True)
            mpath = model_dir / f"metrics_{fs_key}.json"
            with mpath.open("w", encoding="utf-8") as f:
                json.dump({**metrics, "bootstrap_ci": boot, "cv_summary": cv_summary, "best_params": best_params}, f, indent=2)

    results_df = pd.DataFrame(all_results)

    # ── baseline_metrics.csv ──
    csv_path = OUTPUT_DIR / "baseline_metrics.csv"
    cols_csv = ["feature_set", "model", "roc_auc", "ci_lower", "ci_upper", "accuracy", "precision", "recall", "f1_score", "cv_auc_mean", "cv_auc_std"]
    results_df[cols_csv].to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path.name}")
    print(results_df[cols_csv].to_string(index=False))

    # ── baseline_cv_metrics.csv ──
    print("\n  Repeated CV ...")
    cv_rows = []
    for fs_key in ["spatial", "clinical", "combined"]:
        for mc in MODEL_CONFIGS:
            print(f"    {mc['name']} / {fs_key} ...")
            bp = stored_best_params.get((fs_key, mc["key"]), {})
            step_name = mc.get("step", "clf")
            _, cv_df = repeated_cv(
                pd.concat([X_tr[fs_key], X_te[fs_key]]),
                pd.concat([y_tr[fs_key], y_te[fs_key]]),
                mc, bp,
                N_SPLITS, N_REPEATS, RANDOM_STATE,
                step_name=step_name,
            )
            cv_df["feature_set"] = FEATURE_SETS[fs_key]["name"]
            cv_df["model"] = mc["name"]
            cv_rows.append(cv_df)
    cv_all = pd.concat(cv_rows, ignore_index=True)
    cv_path = OUTPUT_DIR / "baseline_cv_metrics.csv"
    cv_all.to_csv(cv_path, index=False)
    print(f"\nSaved {cv_path.name}")

    # ── ROC comparison figure ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)

    for idx, fs_key in enumerate(["spatial", "clinical", "combined"]):
        ax = axes[idx]
        for rd in roc_data:
            if rd["feature_set"] != FEATURE_SETS[fs_key]["name"]:
                continue
            ax.plot(rd["fpr"], rd["tpr"], color=rd["color"], lw=2,
                    label=f"{rd['model']} (AUC={rd['auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.set_xlabel("False Positive Rate", fontsize=9)
        if idx == 0:
            ax.set_ylabel("True Positive Rate", fontsize=9)
        ax.set_title(FEATURE_SETS[fs_key]["name"], fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.2)

    fig.suptitle("ROC Curves — Baseline Classifier Comparison", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "baseline_roc_comparison.png", dpi=150)
    plt.close(fig)
    print("Saved baseline_roc_comparison.png")

    # ── Boxplot of CV AUC ──
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = []
    labels = []
    colors_used = []
    x = 0
    for fs_key in ["spatial", "clinical", "combined"]:
        for mc in MODEL_CONFIGS:
            subset = cv_all[(cv_all["feature_set"] == FEATURE_SETS[fs_key]["name"]) & (cv_all["model"] == mc["name"])]
            if len(subset):
                bp = ax.boxplot(subset["roc_auc"].values, positions=[x], widths=0.6,
                                patch_artist=True,
                                boxprops=dict(facecolor=mc["color"], alpha=0.7),
                                medianprops=dict(color="black", lw=1.5))
                positions.append(x)
                labels.append(f"{mc['key']}")
                colors_used.append(mc["color"])
                x += 1
        x += 0.5  # gap between feature sets

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8, rotation=45)
    ax.set_ylabel("ROC-AUC", fontsize=10)
    ax.set_title("Repeated Stratified CV (5x10) — ROC-AUC Distribution", fontsize=11, fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle="--", lw=1, alpha=0.5)
    ax.grid(axis="y", alpha=0.2)

    # Add feature set separators
    sep_positions = [4.5, 9.0]
    for sp in sep_positions:
        ax.axvline(sp, color="gray", lw=1, linestyle=":", alpha=0.5)
    # Feature set labels at the top
    for i, fs_key in enumerate(["spatial", "clinical", "combined"]):
        center = 1.75 + i * 4.5
        ax.text(center, ax.get_ylim()[1] * 0.98, FEATURE_SETS[fs_key]["name"],
                ha="center", fontsize=9, fontweight="bold", color="gray")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "baseline_boxplot.png", dpi=150)
    plt.close(fig)
    print("Saved baseline_boxplot.png")

    # ── LaTeX table ──
    tex_lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Baseline Classifier Comparison Across Feature Sets}",
        r"\label{tab:baseline_comparison}",
        r"\small",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"Feature Set & Model & ROC-AUC & 95\% CI & Accuracy & Precision & Recall & F1 & CV AUC \\",
        r"\midrule",
    ]

    for fs_key in ["spatial", "clinical", "combined"]:
        fs_name = FEATURE_SETS[fs_key]["name"]
        first_in_group = True
        for mc in MODEL_CONFIGS:
            row = results_df[(results_df["feature_set"] == fs_name) & (results_df["model"] == mc["name"])]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            fs_cell = fs_name if first_in_group else ""
            first_in_group = False
            ci_str = f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
            cv_str = f"{r['cv_auc_mean']:.3f} $\\pm$ {r['cv_auc_std']:.3f}" if pd.notna(r.get("cv_auc_mean")) else "---"
            tex_lines.append(
                f"  {fs_cell} & {mc['name']} & {r['roc_auc']:.3f} & {ci_str} & "
                f"{r['accuracy']:.3f} & {r['precision']:.3f} & "
                f"{r['recall']:.3f} & {r['f1_score']:.3f} & {cv_str} \\\\"
            )
        if fs_key != "combined":
            tex_lines.append(r"  \cmidrule{1-9}")

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    tex_path = OUTPUT_DIR / "baseline_comparison_table.tex"
    with tex_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"Saved {tex_path.name}")

    # ── Report ──
    def best_model_for_fs(fs_name, metric="roc_auc"):
        subset = results_df[results_df["feature_set"] == fs_name]
        if len(subset) == 0:
            return None, None
        best_idx = subset[metric].idxmax()
        return subset.loc[best_idx]["model"], float(subset.loc[best_idx][metric])

    def svm_rank_for_fs(fs_name, metric="roc_auc"):
        subset = results_df[results_df["feature_set"] == fs_name]
        subset = subset.sort_values(metric, ascending=False)
        svm_row = subset[subset["model"] == "SVM (RBF)"]
        if len(svm_row) == 0:
            return None
        rank = subset.index.get_loc(svm_row.index[0]) + 1
        return rank

    # SVM comparison
    svm_comp = []
    for fs_key in ["spatial", "clinical", "combined"]:
        rank = svm_rank_for_fs(FEATURE_SETS[fs_key]["name"])
        svm_comp.append(f"{FEATURE_SETS[fs_key]['name']}: rank {rank}/4")
    report += f"SVM details: {'. '.join(svm_comp)}.\n"

    # Best overall
    all_best_idx = results_df["roc_auc"].idxmax()
    all_best = results_df.iloc[all_best_idx]

    print(f"\n{'=' * 60}")
    print(f"  All outputs in: {OUTPUT_DIR}")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
