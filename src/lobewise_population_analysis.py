#!/usr/bin/env python3
"""Build population-level lobewise probabilistic risk atlas from modality feature CSVs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, pearsonr, spearmanr

MODALITIES = ("t1", "t2", "t1gd", "flair")
INPUT_DIR = Path("outputs")
OUTPUT_DIR = Path("outputs/population_atlas")

LOBE_RATIO_COLS = [
    f"{lobe}_{sub}_ratio"
    for lobe in ("frontal", "temporal", "parietal", "occipital")
    for sub in ("ed", "en", "nc")
]

KEEP_COLS = ["patient_id", "OS_months", *LOBE_RATIO_COLS]

INVOLVEMENT_THRESHOLD = 0.01

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_modality_csv(modality: str) -> pd.DataFrame:
    path = INPUT_DIR / f"features_raw_{modality}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    df = pd.read_csv(path)
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV {path} missing columns: {missing}")
    return df[KEEP_COLS].copy()


def add_risk_label(df: pd.DataFrame) -> pd.DataFrame:
    os_numeric = pd.to_numeric(df["OS_months"], errors="coerce")
    df = df[os_numeric.notna()].copy()
    df["OS_months"] = os_numeric[os_numeric.notna()]
    df["risk_label"] = (df["OS_months"] <= 12).astype(int)
    return df


def cohens_d(x: pd.Series, y: pd.Series) -> float:
    n1, n2 = len(x), len(y)
    mean1, mean2 = x.mean(), y.mean()
    var1, var2 = x.var(ddof=1), y.var(ddof=1)
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float((mean1 - mean2) / pooled)


def safe_mw(high: pd.Series, low: pd.Series) -> tuple[float, float]:
    stat, pval = mannwhitneyu(high, low, alternative="two-sided")
    return float(stat), float(pval)


def safe_odds_ratio(a: int, b: int, c: int, d: int) -> float:
    if b == 0 or c == 0:
        return float("inf")
    return float((a * d) / (b * c))


def compute_population_statistics(modality: str, df: pd.DataFrame) -> None:
    high = df[df["risk_label"] == 1]
    low = df[df["risk_label"] == 0]
    n_high, n_low = len(high), len(low)

    stat_rows: list[dict] = []
    occur_rows: list[dict] = []
    risk_rows: list[dict] = []

    for feat in LOBE_RATIO_COLS:
        m_high = float(high[feat].mean())
        m_low = float(low[feat].mean())
        mean_diff = m_high - m_low
        d = cohens_d(high[feat], low[feat])
        u_stat, pval = safe_mw(high[feat], low[feat])

        stat_rows.append({
            "feature": feat,
            "high_risk_mean": m_high,
            "low_risk_mean": m_low,
            "mean_difference": mean_diff,
            "cohens_d": d,
            "mannwhitney_u": u_stat,
            "p_value": pval,
        })

        involved = df[feat] > INVOLVEMENT_THRESHOLD
        overall_pct = float(involved.mean() * 100)

        high_inv = high[feat] > INVOLVEMENT_THRESHOLD
        low_inv = low[feat] > INVOLVEMENT_THRESHOLD
        high_pct = float(high_inv.mean() * 100)
        low_pct = float(low_inv.mean() * 100)

        a = int(high_inv.sum())
        b = int(low_inv.sum())
        c = n_high - a
        d_val = n_low - b
        or_val = safe_odds_ratio(a, b, c, d_val)

        occur_rows.append({
            "feature": feat,
            "overall_pct": overall_pct,
            "high_risk_pct": high_pct,
            "low_risk_pct": low_pct,
            "odds_ratio": or_val,
            "p_value": pval,
            "cohens_d": d,
        })

        total_inv = a + b
        risk_prob = float(a / total_inv) if total_inv > 0 else 0.0
        risk_rows.append({
            "feature": feat,
            "risk_probability": risk_prob,
        })

    stat_df = pd.DataFrame(stat_rows).sort_values(
        ["p_value", "cohens_d"], ascending=[True, False]
    )
    occur_df = pd.DataFrame(occur_rows).sort_values(
        ["p_value", "cohens_d"], ascending=[True, False]
    )
    risk_df = pd.DataFrame(risk_rows)

    base = OUTPUT_DIR / modality
    stat_df.to_csv(base / "statistical_tests.csv", index=False)
    occur_df.to_csv(base / "occurrence_frequency.csv", index=False)
    risk_df.to_csv(base / "risk_probability_map.csv", index=False)

    print(f"\n  [{modality.upper()}] Statistical analysis saved:")

    risk_assoc = stat_df[stat_df["cohens_d"] > 0].head(5)
    protective = stat_df[stat_df["cohens_d"] < 0].head(5)
    most_prevalent = occur_df.sort_values("overall_pct", ascending=False).head(5)

    print(f"  Top 5 risk-associated regions (highest Cohen's d > 0):")
    for _, r in risk_assoc.iterrows():
        print(f"    {r['feature']:25s}  d={r['cohens_d']:.4f}  p={r['p_value']:.6g}")

    print(f"  Top 5 protective regions (most negative Cohen's d):")
    for _, r in protective.iterrows():
        print(f"    {r['feature']:25s}  d={r['cohens_d']:.4f}  p={r['p_value']:.6g}")

    print(f"  Top 5 most prevalent regions (highest overall %):")
    for _, r in most_prevalent.iterrows():
        print(f"    {r['feature']:25s}  {r['overall_pct']:6.2f}%")


def _load_stat_df(mod: str) -> pd.DataFrame:
    p = OUTPUT_DIR / mod / "statistical_tests.csv"
    df = pd.read_csv(p)
    df["modality"] = mod.upper()
    return df


def _load_occur_df(mod: str) -> pd.DataFrame:
    p = OUTPUT_DIR / mod / "occurrence_frequency.csv"
    df = pd.read_csv(p)
    df["modality"] = mod.upper()
    return df


def compute_multimodal_consensus() -> None:
    combined_dir = OUTPUT_DIR / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    stat_all = pd.concat([_load_stat_df(m) for m in MODALITIES], ignore_index=True)
    occur_all = pd.concat([_load_occur_df(m) for m in MODALITIES], ignore_index=True)

    # --- 1. Combined modality dataframe (modality_agreement.csv) ---
    INFCAP = 1e6
    occur_all["odds_ratio_safe"] = occur_all["odds_ratio"].replace(float("inf"), INFCAP)

    agg = stat_all.groupby("feature").agg(
        mean_cohens_d=("cohens_d", "mean"),
        std_cohens_d=("cohens_d", "std"),
        mean_p_value=("p_value", "mean"),
    )
    or_agg = occur_all.groupby("feature")["odds_ratio_safe"].mean().rename("mean_odds_ratio")
    prev_agg = occur_all.groupby("feature")["overall_pct"].mean().rename("mean_prevalence")

    agreement_rows = []
    for feat in LOBE_RATIO_COLS:
        n_agree = 0
        for mod in MODALITIES:
            row = stat_all[(stat_all["feature"] == feat) & (stat_all["modality"] == mod.upper())]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            if r["cohens_d"] > 0 and r["p_value"] < 0.05:
                n_agree += 1
        agreement_rows.append({
            "feature": feat,
            "modality_consistency": n_agree / len(MODALITIES),
        })

    agreement_df = pd.DataFrame(agreement_rows)
    combined = (
        agg.join(or_agg)
        .join(prev_agg)
        .join(agreement_df.set_index("feature"))
        .reset_index()
    )
    combined = combined[["feature", "mean_cohens_d", "std_cohens_d", "mean_p_value",
                         "mean_odds_ratio", "mean_prevalence", "modality_consistency"]]

    combined.to_csv(combined_dir / "modality_agreement.csv", index=False)
    print("\n[MULTIMODAL] Modality agreement saved -> combined/modality_agreement.csv")

    # --- 2. Cross-modality correlations (correlation_matrix.csv) ---
    corr_rows: list[dict] = []
    metrics = {
        "cohens_d": stat_all,
        "odds_ratio": occur_all,
        "prevalence": occur_all,
    }
    metric_col_map = {
        "cohens_d": "cohens_d",
        "odds_ratio": "odds_ratio_safe",
        "prevalence": "overall_pct",
    }
    for metric_name, df_src in [
        ("cohens_d", stat_all),
        ("odds_ratio", occur_all),
        ("prevalence", occur_all),
    ]:
        col = metric_col_map[metric_name]
        vecs: dict[str, np.ndarray] = {}
        for mod in MODALITIES:
            sub = df_src[df_src["modality"] == mod.upper()].set_index("feature")
            vecs[mod] = sub.loc[LOBE_RATIO_COLS, col].values.astype(float)

        for i, m1 in enumerate(MODALITIES):
            for m2 in MODALITIES[i + 1:]:
                v1, v2 = vecs[m1], vecs[m2]
                finite_mask = np.isfinite(v1) & np.isfinite(v2)
                v1f, v2f = v1[finite_mask], v2[finite_mask]
                if len(v1f) < 3:
                    continue
                pr, pp = pearsonr(v1f, v2f)
                sr, sp = spearmanr(v1f, v2f)
                corr_rows.append({
                    "modality_1": m1.upper(),
                    "modality_2": m2.upper(),
                    "metric": metric_name,
                    "pearson_r": float(pr),
                    "pearson_p": float(pp),
                    "spearman_r": float(sr),
                    "spearman_p": float(sp),
                })
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(combined_dir / "correlation_matrix.csv", index=False)
    print("  Cross-modality correlations saved -> combined/correlation_matrix.csv")

    # --- 3. Consensus ranking (consensus_rankings.csv) ---
    consensus = combined.copy()
    consensus["consensus_score"] = consensus["mean_cohens_d"] * consensus["modality_consistency"]
    consensus = consensus.sort_values("consensus_score", ascending=False).reset_index(drop=True)
    consensus.to_csv(combined_dir / "consensus_rankings.csv", index=False)
    print("  Consensus rankings saved -> combined/consensus_rankings.csv")

    # --- 4. Multimodal summary (multimodal_summary.csv) ---
    summary_rows = []
    for feat in LOBE_RATIO_COLS:
        sub_s = stat_all[stat_all["feature"] == feat]
        sub_o = occur_all[occur_all["feature"] == feat]
        d_signed = sub_s["cohens_d"].values
        p_vals = sub_s["p_value"].values
        n_pos = int(np.sum((d_signed > 0) & (p_vals < 0.05)))
        n_neg = int(np.sum((d_signed < 0) & (p_vals < 0.05)))
        n_ns = len(MODALITIES) - n_pos - n_neg
        summary_rows.append({
            "feature": feat,
            "mean_cohens_d": float(sub_s["cohens_d"].mean()),
            "modalities_risk": n_pos,
            "modalities_protective": n_neg,
            "modalities_ns": n_ns,
            "mean_prevalence": float(sub_o["overall_pct"].mean()),
            "mean_odds_ratio": float(sub_o["odds_ratio_safe"].mean()),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(combined_dir / "multimodal_summary.csv", index=False)
    print("  Multimodal summary saved -> combined/multimodal_summary.csv")

    # --- 5. Console summary ---
    stable = consensus.iloc[0]
    print(f"\n  Most stable prognostic region: {stable['feature']} "
          f"(consensus={stable['consensus_score']:.4f}, "
          f"consistency={stable['modality_consistency']:.2f})")

    inconsistent = combined.loc[combined["modality_consistency"].idxmin()]
    print(f"  Most inconsistent region: {inconsistent['feature']} "
          f"(consistency={inconsistent['modality_consistency']:.2f})")

    strongest = consensus.iloc[0]
    print(f"  Strongest multimodal consensus feature: {strongest['feature']} "
          f"(d={strongest['mean_cohens_d']:.4f}, "
          f"score={strongest['consensus_score']:.4f})")

    max_std_row = combined.loc[combined["std_cohens_d"].idxmax()]
    print(f"  Strongest modality disagreement: {max_std_row['feature']} "
          f"(std d={max_std_row['std_cohens_d']:.4f})")

    # --- 6. Plots ---
    _plot_correlation_heatmap(corr_df, combined_dir)
    _plot_consensus_barplot(consensus, combined_dir)
    _plot_agreement_heatmap(stat_all, combined_dir)


def _plot_correlation_heatmap(corr_df: pd.DataFrame, out_dir: Path) -> None:
    for metric in ["cohens_d", "odds_ratio", "prevalence"]:
        sub = corr_df[corr_df["metric"] == metric].copy()
        if sub.empty:
            continue
        mods_upper = [m.upper() for m in MODALITIES]
        mat = pd.DataFrame(np.nan, index=mods_upper, columns=mods_upper)
        np.fill_diagonal(mat.values, 1.0)
        for _, r in sub.iterrows():
            mat.loc[r["modality_1"], r["modality_2"]] = r["pearson_r"]
            mat.loc[r["modality_2"], r["modality_1"]] = r["pearson_r"]

        fig, ax = plt.subplots(figsize=(5, 4.5))
        sns.heatmap(mat, annot=True, fmt=".3f", cmap="RdBu_r", vmin=-1, vmax=1,
                    square=True, cbar_kws={"shrink": 0.8}, ax=ax)
        label_map = {"cohens_d": "Cohen's d", "odds_ratio": "Odds Ratio", "prevalence": "Prevalence"}
        ax.set_title(f"Cross-modality Pearson r — {label_map[metric]}")
        ax.set_ylabel("")
        ax.set_xlabel("")
        fig.tight_layout()
        path = out_dir / f"correlation_heatmap_{metric}.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved plot -> combined/{path.name}")


def _plot_consensus_barplot(consensus: pd.DataFrame, out_dir: Path) -> None:
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in consensus["mean_cohens_d"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(range(len(consensus)), consensus["consensus_score"].values,
                   color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(consensus)))
    ax.set_yticklabels(consensus["feature"].values, fontsize=9)
    ax.set_xlabel("Consensus Score (mean Cohen's d × consistency)")
    ax.set_title("Multimodal Consensus Ranking of Lobe-Subregion Features")
    ax.axvline(0, color="grey", linestyle="-", linewidth=0.5)
    fig.tight_layout()
    path = out_dir / "consensus_barplot.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved plot -> combined/{path.name}")


def _plot_agreement_heatmap(stat_all: pd.DataFrame, out_dir: Path) -> None:
    mat = pd.DataFrame(index=[m.upper() for m in MODALITIES], columns=LOBE_RATIO_COLS, dtype=float)
    for mod_upper in mat.index:
        for feat in LOBE_RATIO_COLS:
            row = stat_all[(stat_all["modality"] == mod_upper) & (stat_all["feature"] == feat)]
            if len(row) == 0:
                mat.loc[mod_upper, feat] = np.nan
            else:
                r = row.iloc[0]
                mat.loc[mod_upper, feat] = r["cohens_d"]

    annot_mat = mat.copy()
    for col in annot_mat.columns:
        for row_idx in annot_mat.index:
            v = mat.loc[row_idx, col]
            if pd.isna(v):
                annot_mat.loc[row_idx, col] = ""
            elif v > 0:
                r = stat_all[(stat_all["modality"] == row_idx) & (stat_all["feature"] == col)]
                p = r["p_value"].iloc[0] if len(r) > 0 else 1.0
                annot_mat.loc[row_idx, col] = "+" if p >= 0.05 else "*"
            else:
                r = stat_all[(stat_all["modality"] == row_idx) & (stat_all["feature"] == col)]
                p = r["p_value"].iloc[0] if len(r) > 0 else 1.0
                annot_mat.loc[row_idx, col] = "-" if p >= 0.05 else "x"

    fig, ax = plt.subplots(figsize=(10, 3.5))
    vmax = max(abs(mat.values[~np.isnan(mat.values)].min()), abs(mat.values[~np.isnan(mat.values)].max()), 0.01)
    sns.heatmap(mat, annot=annot_mat, fmt="", cmap="RdBu_r", center=0,
                vmin=-vmax, vmax=vmax, linewidths=0.5, cbar_kws={"shrink": 0.8, "label": "Cohen's d"},
                ax=ax)
    ax.set_title("Modality Agreement Across Lobe-Subregion Features\n(* = sig. risk, x = sig. protective, +/- = ns)")
    ax.set_ylabel("Modality")
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    path = out_dir / "modality_agreement_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved plot -> combined/{path.name}")


def main() -> int:
    loaded: dict[str, pd.DataFrame] = {}
    for mod in MODALITIES:
        loaded[mod] = load_modality_csv(mod)
        print(f"Loaded {mod}: {len(loaded[mod])} patients")

    id_sets = [set(df["patient_id"].unique()) for df in loaded.values()]
    shared_ids = id_sets[0]
    for s in id_sets[1:]:
        shared_ids &= s
    print(f"Patients shared across all 4 modalities: {len(shared_ids)}")

    for mod in MODALITIES:
        before = len(loaded[mod])
        loaded[mod] = loaded[mod][loaded[mod]["patient_id"].isin(shared_ids)]
        print(f"  {mod}: {before} -> {len(loaded[mod])} (after intersection)")

    for mod in MODALITIES:
        df = loaded[mod]
        df = add_risk_label(df)
        out_path = OUTPUT_DIR / mod / "clean_population_features.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        risk_counts = df["risk_label"].value_counts().to_dict()
        df.to_csv(out_path, index=False)
        print(f"\n{mod.upper()}:")
        print(f"  Patients: {len(df)}")
        print(f"  High risk (OS <= 12mo): {risk_counts.get(1, 0)}")
        print(f"  Low risk  (OS > 12mo):  {risk_counts.get(0, 0)}")
        print(f"  Saved -> {out_path}")

        compute_population_statistics(mod, df)

    compute_multimodal_consensus()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
