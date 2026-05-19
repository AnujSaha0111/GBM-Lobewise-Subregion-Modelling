#!/usr/bin/env python3
"""Research-grade GBM risk classifier training workflow."""

from __future__ import annotations

import argparse
import inspect
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

xgb.set_config(verbosity=0)

FEATURE_COLS = [
    "global_nc_en_ratio",
    "global_ed_en_ratio",
    "global_ed_total_ratio",
    "tumor_burden_index",
    *(f"{lb}_{sub}_ratio" for lb in ("frontal", "temporal", "parietal", "occipital")
      for sub in ("ed", "en", "nc")),
]

TARGET_COL = "risk_label"
LABEL_NAMES = ["low-risk", "high-risk"]
MODALITIES = ("T1", "T2", "T1GD", "FLAIR")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the GBM risk classification training workflow.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
        help="Path to config.json",
    )
    parser.add_argument(
        "--modality",
        choices=MODALITIES,
        type=str.upper,
        default=None,
        help="Modality for default inputs/outputs (T1, T2, T1GD, FLAIR).",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Processed features CSV (overrides config/modality default).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Root output directory (overrides config/modality default).",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def resolve_path(root: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (root / p)


def ensure_directories(outputs_root: Path) -> dict[str, Path]:
    dirs = {
        "root": outputs_root,
        "figures": outputs_root / "figures",
        "metrics": outputs_root / "metrics",
        "tables": outputs_root / "tables",
        "models": outputs_root / "models",
        "logs": outputs_root / "logs",
        "reports": outputs_root / "reports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def save_json(payload: dict | list, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_markdown(content: str, path: Path) -> None:
    path.write_text(content, encoding="utf-8")


def scale_pos_weight(y: pd.Series) -> float:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        return 1.0
    return float(n_neg) / float(n_pos)


def detect_xgboost_device(random_state: int) -> dict[str, object]:
    summary = {
        "selected_device": "cpu",
        "gpu_available": False,
        "reason": "GPU not detected",
    }
    try:
        model = xgb.XGBClassifier(
            n_estimators=1,
            max_depth=1,
            tree_method="hist",
            device="cuda",
            random_state=random_state,
        )
        X_tmp = np.array([[0.0], [1.0]])
        y_tmp = np.array([0, 1])
        model.fit(X_tmp, y_tmp, verbose=False)
        summary = {
            "selected_device": "cuda",
            "gpu_available": True,
            "reason": "GPU training available",
        }
    except Exception:
        pass
    return summary


def xgb_base_params(device: str, random_state: int, n_jobs: int) -> dict[str, object]:
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": random_state,
        "n_jobs": n_jobs,
    }
    if device == "cuda":
        params.update({"tree_method": "hist", "device": "cuda"})
    else:
        params.update({"tree_method": "hist"})
    return params


def fit_xgb_with_optional_early_stopping(
    model: xgb.XGBClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weights: np.ndarray,
    eval_set: list[tuple[pd.DataFrame, pd.Series]],
    early_stopping_rounds: int,
) -> None:
    params = inspect.signature(model.fit).parameters
    kwargs: dict[str, object] = {
        "sample_weight": sample_weights,
    }
    if "eval_set" in params:
        kwargs["eval_set"] = eval_set
    if "verbose" in params:
        kwargs["verbose"] = False
    if early_stopping_rounds > 0:
        if "early_stopping_rounds" in params:
            kwargs["early_stopping_rounds"] = early_stopping_rounds
        elif "callbacks" in params:
            kwargs["callbacks"] = [
                xgb.callback.EarlyStopping(rounds=early_stopping_rounds)
            ]
    model.fit(X_train, y_train, **kwargs)


def load_dataset(input_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict[str, object]]:
    df = pd.read_csv(input_csv)
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input CSV: {missing}")

    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    valid_mask = y.isin([0, 1])
    df = df[valid_mask].copy()
    y = y[valid_mask].astype(int)

    X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    total_nulls = int(X.isnull().sum().sum())
    X = X.fillna(X.median())

    summary = {
        "rows_after_label_drop": int(len(df)),
        "n_features": int(len(FEATURE_COLS)),
        "class_distribution": y.value_counts().to_dict(),
        "total_feature_nulls": total_nulls,
    }
    return df, X, y, summary


def plot_class_distribution(labels: pd.Series, figure_path: Path) -> None:
    counts = labels.value_counts().reindex([0, 1]).fillna(0)
    ax = counts.plot(kind="bar", color=["green", "red"], figsize=(7, 5))
    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")
    ax.set_xticklabels(LABEL_NAMES, rotation=0)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close()


def plot_feature_distributions(X: pd.DataFrame, labels: pd.Series, figure_path: Path) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(18, 14))
    axes = axes.flatten()
    for index, feature in enumerate(FEATURE_COLS):
        for label_value, label_name in enumerate(LABEL_NAMES):
            axes[index].hist(
                X.loc[labels == label_value, feature],
                bins=20,
                alpha=0.5,
                label=label_name,
            )
        axes[index].set_title(feature, fontsize=8)
        axes[index].legend(fontsize=6)

    for index in range(len(FEATURE_COLS), len(axes)):
        axes[index].set_visible(False)

    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(X: pd.DataFrame, figure_path: Path) -> pd.DataFrame:
    corr = X.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close()
    return corr


def high_correlation_pairs(corr_matrix: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    upper = corr_matrix.abs().where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    rows: list[dict[str, object]] = []
    for col in upper.columns:
        for row in upper.index:
            value = upper.loc[row, col]
            if pd.notna(value) and value > threshold:
                rows.append(
                    {
                        "feature_a": row,
                        "feature_b": col,
                        "absolute_correlation": float(value),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["feature_a", "feature_b", "absolute_correlation"])
    return pd.DataFrame(rows).sort_values("absolute_correlation", ascending=False)


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
    test_size: float,
    validation_fraction_of_trainval: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=validation_fraction_of_trainval,
        random_state=random_state,
        stratify=y_trainval,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_split_assignments(
    df: pd.DataFrame,
    y: pd.Series,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    output_path: Path,
) -> None:
    split_series = pd.Series(index=y.index, dtype="string")
    split_series.loc[X_train.index] = "train"
    split_series.loc[X_val.index] = "validation"
    split_series.loc[X_test.index] = "test"
    indices = split_series.dropna().index
    output = pd.DataFrame({
        "row_index": indices,
        TARGET_COL: y.loc[indices].values,
        "split": split_series.dropna().astype(str).values,
    })
    if "patient_id" in df.columns:
        output.insert(1, "patient_id", df.loc[indices, "patient_id"].astype(str).values)
    output.to_csv(output_path, index=False)


def run_baselines(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    output_path: Path,
) -> pd.DataFrame:
    scaled_models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "SVM (RBF)": SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=42,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }

    tree_models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            random_state=42,
        ),
    }

    vt = VarianceThreshold()
    scaler = RobustScaler()

    X_train_scaled = scaler.fit_transform(vt.fit_transform(X_train))
    X_val_scaled = scaler.transform(vt.transform(X_val))

    rows = []

    for name, model in scaled_models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_val_scaled)
        rows.append(
            {
                "model": name,
                "balanced_accuracy": balanced_accuracy_score(y_val, preds),
                "macro_f1": f1_score(y_val, preds, average="macro"),
            }
        )

    for name, model in tree_models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rows.append(
            {
                "model": name,
                "balanced_accuracy": balanced_accuracy_score(y_val, preds),
                "macro_f1": f1_score(y_val, preds, average="macro"),
            }
        )

    baseline_df = pd.DataFrame(rows).sort_values(
        ["balanced_accuracy", "macro_f1"],
        ascending=False,
    )
    baseline_df.to_csv(output_path, index=False)
    return baseline_df


def fit_validation_xgb(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    device: str,
    random_state: int,
    early_stopping_rounds: int,
) -> xgb.XGBClassifier:
    params = xgb_base_params(
        device=device,
        random_state=random_state,
        n_jobs=-1 if device == "cpu" else 1,
    )
    params.update(
        {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 4,
            "gamma": 0.0,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "scale_pos_weight": scale_pos_weight(y_train),
        }
    )
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    model = xgb.XGBClassifier(**params)
    fit_xgb_with_optional_early_stopping(
        model,
        X_train,
        y_train,
        sample_weights,
        [(X_val, y_val)],
        early_stopping_rounds,
    )
    return model


def tune_xgboost(
    X_trainval: pd.DataFrame,
    y_trainval: pd.Series,
    random_state: int,
    n_trials: int,
    n_splits: int,
    device: str,
    early_stopping_rounds: int,
    trials_output_path: Path,
) -> tuple[optuna.Study, dict[str, object]]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = xgb_base_params(
            device=device,
            random_state=random_state,
            n_jobs=-1 if device == "cpu" else 1,
        )
        params.update(
            {
                "n_estimators": trial.suggest_int("n_estimators", 100, 700),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            }
        )

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scores: list[float] = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_trainval, y_trainval), start=1):
            X_tr = X_trainval.iloc[train_idx]
            X_va = X_trainval.iloc[val_idx]
            y_tr = y_trainval.iloc[train_idx]
            y_va = y_trainval.iloc[val_idx]
            params["scale_pos_weight"] = scale_pos_weight(y_tr)
            sample_weights = compute_sample_weight(class_weight="balanced", y=y_tr)
            model = xgb.XGBClassifier(**params)
            fit_xgb_with_optional_early_stopping(
                model,
                X_tr,
                y_tr,
                sample_weights,
                [(X_va, y_va)],
                early_stopping_rounds,
            )
            preds = model.predict(X_va)
            score = balanced_accuracy_score(y_va, preds)
            scores.append(score)
            trial.report(float(np.mean(scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(trials_output_path, index=False)
    best_params = study.best_params.copy()
    return study, best_params


def fit_best_model(
    X_trainval: pd.DataFrame,
    y_trainval: pd.Series,
    best_params: dict[str, object],
    device: str,
    random_state: int,
) -> xgb.XGBClassifier:
    params = xgb_base_params(
        device=device,
        random_state=random_state,
        n_jobs=-1 if device == "cpu" else 1,
    )
    params.update(best_params)
    params["scale_pos_weight"] = scale_pos_weight(y_trainval)
    model = xgb.XGBClassifier(**params)
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_trainval)
    model.fit(X_trainval, y_trainval, sample_weight=sample_weights, verbose=False)
    return model


def evaluate_model(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    figures_dir: Path,
    metrics_dir: Path,
) -> dict[str, object]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report_dict).transpose().to_csv(metrics_dir / "test_classification_report.csv")

    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "quadratic_kappa": float(cohen_kappa_score(y_test, y_pred, weights="quadratic")),
        "auc": float(roc_auc_score(y_test, y_proba[:, 1])),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    save_json(metrics, metrics_dir / "test_metrics.json")

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=LABEL_NAMES,
        cmap="Blues",
        ax=ax,
    )
    plt.title("Confusion Matrix - Test Set")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    return metrics


def save_feature_importance(
    model: xgb.XGBClassifier,
    feature_names: list[str],
    figures_dir: Path,
    tables_dir: Path,
) -> None:
    importances = model.feature_importances_
    if len(feature_names) != len(importances):
        min_len = min(len(feature_names), len(importances))
        feature_names = feature_names[:min_len]
        importances = importances[:min_len]
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=True)
    importance_df.to_csv(tables_dir / "feature_importance.csv", index=False)

    plt.figure(figsize=(8, 7))
    plt.barh(importance_df["feature"], importance_df["importance"], color="steelblue")
    plt.xlabel("Importance")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_importance.png", dpi=150)
    plt.close()


def save_shap_outputs(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    feature_names: list[str],
    figures_dir: Path,
    logs_dir: Path,
) -> None:
    error_path = logs_dir / "shap_error.txt"
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        shap.summary_plot(
            shap_values,
            X_test,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
        )
        plt.title("SHAP Importance - High Risk")
        plt.tight_layout()
        plt.savefig(figures_dir / "shap_importance.png", dpi=150)
        plt.close()

        shap.summary_plot(
            shap_values,
            X_test,
            feature_names=feature_names,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(figures_dir / "shap_summary.png", dpi=150)
        plt.close()
        if error_path.exists():
            error_path.unlink()
    except Exception as exc:
        error_path.write_text(str(exc), encoding="utf-8")


def final_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    best_params: dict[str, object],
    random_state: int,
    n_splits: int,
    device: str,
    output_path: Path,
) -> dict[str, object]:
    params = xgb_base_params(
        device=device,
        random_state=random_state,
        n_jobs=-1 if device == "cpu" else 1,
    )
    params.update(best_params)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows: list[dict[str, float]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_tr = X.iloc[train_idx]
        X_te = X.iloc[test_idx]
        y_tr = y.iloc[train_idx]
        y_te = y.iloc[test_idx]

        params["scale_pos_weight"] = scale_pos_weight(y_tr)
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_tr)
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, sample_weight=sample_weights, verbose=False)
        proba = model.predict_proba(X_te)[:, 1]
        preds = (proba >= 0.5).astype(int)

        bal_acc = float(balanced_accuracy_score(y_te, preds))
        auc = float(roc_auc_score(y_te, proba))

        rows.append(
            {
                "fold": fold_idx,
                "balanced_accuracy": bal_acc,
                "auc": auc,
            }
        )

    cv_df = pd.DataFrame(rows)
    cv_df.to_csv(output_path, index=False)
    auc_mean = float(cv_df["auc"].mean())
    auc_std = float(cv_df["auc"].std())

    n = len(cv_df)
    if n >= 2:
        auc_ci_low, auc_ci_high = stats.t.interval(
            0.95,
            df=n - 1,
            loc=auc_mean,
            scale=stats.sem(cv_df["auc"]),
        )
        t_stat, p_two = stats.ttest_1samp(cv_df["auc"], popmean=0.5)
        if np.isnan(p_two):
            auc_p = float("nan")
        elif auc_mean > 0.5:
            auc_p = p_two / 2.0
        else:
            auc_p = 1.0 - (p_two / 2.0)
    else:
        auc_ci_low, auc_ci_high, auc_p = float("nan"), float("nan"), float("nan")

    return {
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "auc_ci": [auc_ci_low, auc_ci_high],
        "auc_p": auc_p,
    }


def get_feature_summary(X: pd.DataFrame, y: pd.Series) -> dict[str, object]:
    return {
        "n_samples": int(len(X)),
        "n_features_total": int(X.shape[1]),
        "feature_names": list(X.columns),
        "class_distribution": y.value_counts().to_dict(),
        "feature_mean": X.mean().to_dict(),
        "feature_std": X.std().to_dict(),
        "missing_values": int(X.isnull().sum().sum()),
        "correlation_max": float(
            X.corr().abs().values[np.triu_indices(X.shape[1], 1)].max()
        ),
    }


def save_model_artifacts(model: xgb.XGBClassifier, outputs_root: Path) -> None:
    joblib.dump(model, outputs_root / "models" / "best_xgb_risk_classifier.pkl")
    model.save_model(str(outputs_root / "final_model.json"))
    with (outputs_root / "feature_list.json").open("w", encoding="utf-8") as handle:
        json.dump({"features": FEATURE_COLS}, handle, indent=2)


def build_markdown_report(
    dataset_summary: dict[str, object],
    device_summary: dict[str, object],
    baseline_df: pd.DataFrame,
    validation_metrics: dict[str, object],
    test_metrics: dict[str, object],
    cv_summary: dict[str, object],
    best_params: dict[str, object],
    optuna_best_value: float,
    config: dict[str, object],
) -> str:
    best_baseline = baseline_df.iloc[0].to_dict() if not baseline_df.empty else {}
    return f"""# GBM Training Summary

## Dataset

- Rows after label cleaning: `{dataset_summary['rows_after_label_drop']}`
- Imaging features used: `{dataset_summary['n_features']}`
- Class distribution: `{dataset_summary['class_distribution']}`
- Total feature nulls at training input: `{dataset_summary['total_feature_nulls']}`

## Runtime

- Selected device: `{device_summary['selected_device']}`
- GPU available: `{device_summary['gpu_available']}`
- Device note: `{device_summary['reason']}`
- Optuna trials run: `{config['optuna_trials']}`

## Baselines

- Best validation baseline: `{best_baseline.get('model', 'n/a')}`
- Best baseline balanced accuracy: `{best_baseline.get('balanced_accuracy', float('nan')):.4f}`
- Best baseline macro F1: `{best_baseline.get('macro_f1', float('nan')):.4f}`

## XGBoost

- Validation balanced accuracy: `{validation_metrics['balanced_accuracy']:.4f}`
- Validation macro F1: `{validation_metrics['macro_f1']:.4f}`
- Best Optuna CV balanced accuracy: `{optuna_best_value:.4f}`
- Best params: `{best_params}`

## Final Test Metrics

- Balanced accuracy: `{test_metrics['balanced_accuracy']:.4f}`
- Macro F1: `{test_metrics['macro_f1']:.4f}`
- Quadratic kappa: `{test_metrics['quadratic_kappa']:.4f}`
- AUC: `{test_metrics['auc']:.4f}`

## Final Cross-Validation

- AUC: `{cv_summary['auc_mean']:.4f}`
- 95% CI: `{cv_summary['auc_ci'][0]:.4f}` to `{cv_summary['auc_ci'][1]:.4f}`
- One-sided p-value vs 0.5: `{cv_summary['auc_p']:.6g}`
"""


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(resolve_path(root, str(args.config)))
    training_cfg = cfg.get("training", {})

    input_cfg = training_cfg.get("input_csv", "outputs/features_processed.csv")
    output_cfg = training_cfg.get("output_dir", "outputs")

    if args.modality:
        modality = args.modality.upper()
        modality_lc = modality.lower()
        default_input = f"outputs/features_processed_{modality_lc}.csv"
        default_output = f"outputs/{modality}"
    else:
        default_input = input_cfg
        default_output = output_cfg

    input_csv = args.input or default_input
    output_root = args.output or default_output

    random_state = int(training_cfg.get("random_seed", 42))
    test_size = float(training_cfg.get("test_size", 0.15))
    val_frac = float(training_cfg.get("validation_fraction_of_trainval", 0.176470588))
    optuna_trials = int(training_cfg.get("optuna_trials", 50))
    optuna_cv_folds = int(training_cfg.get("optuna_cv_folds", 5))
    final_cv_folds = int(training_cfg.get("final_cv_folds", training_cfg.get("cv_folds", 10)))
    early_stopping_rounds = int(training_cfg.get("xgboost_early_stopping_rounds", 25))

    input_csv = resolve_path(root, input_csv)
    outputs_root = resolve_path(root, output_root)

    paths = ensure_directories(outputs_root)

    clean_df, X, y, dataset_summary = load_dataset(input_csv)
    save_json(dataset_summary, paths["metrics"] / "dataset_summary.json")

    feature_summary = get_feature_summary(X, y)
    save_json(feature_summary, paths["metrics"] / "feature_summary.json")

    device_summary = detect_xgboost_device(random_state=random_state)
    save_json(device_summary, paths["logs"] / "gpu_status.json")

    plot_class_distribution(y, paths["figures"] / "class_distribution.png")
    plot_feature_distributions(X, y, paths["figures"] / "feature_distributions.png")
    corr_matrix = plot_correlation_heatmap(X, paths["figures"] / "correlation_heatmap.png")
    high_corr_df = high_correlation_pairs(corr_matrix, threshold=0.9)
    high_corr_df.to_csv(paths["tables"] / "high_correlation_pairs.csv", index=False)
    X.describe().transpose().to_csv(paths["tables"] / "feature_summary.csv")

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X=X,
        y=y,
        random_state=random_state,
        test_size=test_size,
        validation_fraction_of_trainval=val_frac,
    )
    X_trainval = pd.concat([X_train, X_val]).sort_index()
    y_trainval = pd.concat([y_train, y_val]).sort_index()

    save_split_assignments(
        clean_df,
        y,
        X_train,
        X_val,
        X_test,
        paths["tables"] / "split_assignments.csv",
    )

    baseline_df = run_baselines(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        output_path=paths["metrics"] / "baseline_results.csv",
    )

    validation_model = fit_validation_xgb(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        device=str(device_summary["selected_device"]),
        random_state=random_state,
        early_stopping_rounds=early_stopping_rounds,
    )
    val_preds = validation_model.predict(X_val)
    validation_metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_val, val_preds)),
        "macro_f1": float(f1_score(y_val, val_preds, average="macro")),
    }
    save_json(validation_metrics, paths["metrics"] / "validation_xgboost_metrics.json")

    study, best_params = tune_xgboost(
        X_trainval=X_trainval,
        y_trainval=y_trainval,
        random_state=random_state,
        n_trials=optuna_trials,
        n_splits=optuna_cv_folds,
        device=str(device_summary["selected_device"]),
        early_stopping_rounds=early_stopping_rounds,
        trials_output_path=paths["logs"] / "optuna_trials.csv",
    )
    save_json(best_params, paths["models"] / "best_params.json")

    best_model = fit_best_model(
        X_trainval=X_trainval,
        y_trainval=y_trainval,
        best_params=best_params,
        device=str(device_summary["selected_device"]),
        random_state=random_state,
    )

    test_metrics = evaluate_model(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        figures_dir=paths["figures"],
        metrics_dir=paths["metrics"],
    )
    save_feature_importance(best_model, list(X_trainval.columns), paths["figures"], paths["tables"])
    save_shap_outputs(best_model, X_test, list(X_test.columns), paths["figures"], paths["logs"])

    cv_summary = final_cross_validation(
        X=X,
        y=y,
        best_params=best_params,
        random_state=random_state,
        n_splits=final_cv_folds,
        device=str(device_summary["selected_device"]),
        output_path=paths["metrics"] / "final_cv_fold_metrics.csv",
    )
    save_json(cv_summary, paths["metrics"] / "final_cv_summary.json")
    save_json(
        {
            "fold_aucs": pd.read_csv(paths["metrics"] / "final_cv_fold_metrics.csv")["auc"].tolist(),
            "mean_auc": cv_summary["auc_mean"],
            "ci_low": cv_summary["auc_ci"][0],
            "ci_high": cv_summary["auc_ci"][1],
            "p_value": cv_summary["auc_p"],
            "cv_folds": final_cv_folds,
        },
        outputs_root / "cv_results.json",
    )

    save_model_artifacts(best_model, outputs_root)

    report = build_markdown_report(
        dataset_summary=dataset_summary,
        device_summary=device_summary,
        baseline_df=baseline_df,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        cv_summary=cv_summary,
        best_params=best_params,
        optuna_best_value=float(study.best_value),
        config={
            "optuna_trials": optuna_trials,
        },
    )
    save_markdown(report, paths["reports"] / "training_summary.md")

    report_lines = [
        "GBM Survival Risk Stratification - Training Report",
        f"Samples: {len(clean_df)}",
        f"Positives: {(y == 1).sum()}  Negatives: {(y == 0).sum()}",
        f"Mean CV AUC: {cv_summary['auc_mean']:.4f}",
        f"95% CI: [{cv_summary['auc_ci'][0]:.4f}, {cv_summary['auc_ci'][1]:.4f}]",
        f"One-sided p-value vs 0.5: {cv_summary['auc_p']:.6g}",
    ]
    (outputs_root / "training_report.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    print("Training completed successfully.")
    print("Summary:")
    print(f"  Samples: {len(clean_df)}")
    print(f"  Positives: {(y == 1).sum()}  Negatives: {(y == 0).sum()}")
    print(f"  Validation balanced accuracy: {validation_metrics['balanced_accuracy']:.4f}")
    print(f"  Validation macro F1: {validation_metrics['macro_f1']:.4f}")
    print(f"  Test balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"  Test macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"  Test AUC: {test_metrics['auc']:.4f}")
    print(f"  Final CV mean AUC: {cv_summary['auc_mean']:.4f}")
    print(f"  Final CV 95% CI: [{cv_summary['auc_ci'][0]:.4f}, {cv_summary['auc_ci'][1]:.4f}]")
    print(f"  Final CV p-value vs 0.5: {cv_summary['auc_p']:.6g}")


if __name__ == "__main__":
    main()
