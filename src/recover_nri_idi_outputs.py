#!/usr/bin/env python3
"""Recover NRI/IDI deliverables from completed analysis outputs.

This script does not rerun model fitting or bootstrap resampling.
It reads the already-written full-cohort and IDH-wildtype JSON outputs,
then rebuilds the combined requested deliverables and the missing plots.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs" / "survival_incremental_value"
DOCS_DIR = ROOT / "docs"

RESULTS_FULL = OUTPUT_DIR / "nri_idi_results.json"
RESULTS_WT = OUTPUT_DIR / "nri_idi_results_idh_wildtype.json"
BOOT_FULL = OUTPUT_DIR / "nri_idi_bootstrap.json"
BOOT_WT = OUTPUT_DIR / "nri_idi_bootstrap_idh_wildtype.json"

TIME_ORDER = ["12m", "24m", "36m"]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(payload: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _format_p(p: float) -> str:
    if p < 0.0001:
        return "<0.0001"
    return f"{p:.4f}"


def _cohort_label(label: str) -> str:
    return "IDH-wildtype" if label.lower().startswith("idh") or label.lower().startswith("idh-") else label


def _ordered_results(rows: list[dict]) -> list[dict]:
    order = {time: idx for idx, time in enumerate(TIME_ORDER)}
    return sorted(rows, key=lambda row: order[row["time"]])


def _build_combined_results(full_payload: dict, wt_payload: dict) -> list[dict]:
    rows = []
    for payload in [full_payload, wt_payload]:
        for row in _ordered_results(payload["results"]):
            item = dict(row)
            item["cohort"] = _cohort_label(item["cohort"])
            rows.append(item)
    return rows


def _build_combined_bootstrap(full_boot: dict, wt_boot: dict) -> list[dict]:
    rows = []
    for cohort, boot in [("Full cohort", full_boot), ("IDH-wildtype", wt_boot)]:
        for time in TIME_ORDER:
            rows.append(
                {
                    "cohort": cohort,
                    "time": time,
                    "n_bootstrap": boot["n_bootstrap"],
                    "c_nri_mean": boot[f"c_nri_{time}_mean"],
                    "c_nri_std": boot[f"c_nri_{time}_std"],
                    "c_nri_ci_lower": boot[f"c_nri_{time}_ci_lower"],
                    "c_nri_ci_upper": boot[f"c_nri_{time}_ci_upper"],
                    "c_nri_p_value": boot[f"c_nri_{time}_p_value"],
                    "nri_case_mean": boot[f"nri_case_{time}_mean"],
                    "nri_case_std": boot[f"nri_case_{time}_std"],
                    "nri_case_ci_lower": boot[f"nri_case_{time}_ci_lower"],
                    "nri_case_ci_upper": boot[f"nri_case_{time}_ci_upper"],
                    "nri_case_p_value": boot[f"nri_case_{time}_p_value"],
                    "nri_control_mean": boot[f"nri_control_{time}_mean"],
                    "nri_control_std": boot[f"nri_control_{time}_std"],
                    "nri_control_ci_lower": boot[f"nri_control_{time}_ci_lower"],
                    "nri_control_ci_upper": boot[f"nri_control_{time}_ci_upper"],
                    "nri_control_p_value": boot[f"nri_control_{time}_p_value"],
                    "idi_mean": boot[f"idi_{time}_mean"],
                    "idi_std": boot[f"idi_{time}_std"],
                    "idi_ci_lower": boot[f"idi_{time}_ci_lower"],
                    "idi_ci_upper": boot[f"idi_{time}_ci_upper"],
                    "idi_p_value": boot[f"idi_{time}_p_value"],
                }
            )
    return rows


def _make_plots(full_rows: list[dict], full_boot: dict, wt_rows: list[dict], wt_boot: dict) -> None:
    x = np.arange(len(TIME_ORDER))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def _series(rows: list[dict], key: str) -> list[float]:
        ordered = _ordered_results(rows)
        return [row[key] for row in ordered]

    def _errors(rows: list[dict], boot: dict, metric: str) -> np.ndarray:
        lows = []
        highs = []
        for row in _ordered_results(rows):
            time = row["time"]
            value = row[metric]
            lows.append(value - boot[f"{metric}_{time}_ci_lower"])
            highs.append(boot[f"{metric}_{time}_ci_upper"] - value)
        return np.array([lows, highs])

    cnri_full = _series(full_rows, "c_nri")
    cnri_wt = _series(wt_rows, "c_nri")
    idi_full = _series(full_rows, "idi")
    idi_wt = _series(wt_rows, "idi")

    axes[0].bar(
        x - width / 2,
        cnri_full,
        width,
        yerr=_errors(full_rows, full_boot, "c_nri"),
        capsize=4,
        color="#356fb3",
        label="Full cohort",
    )
    axes[0].bar(
        x + width / 2,
        cnri_wt,
        width,
        yerr=_errors(wt_rows, wt_boot, "c_nri"),
        capsize=4,
        color="#c44e52",
        label="IDH-wildtype",
    )
    axes[0].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_title("Continuous NRI")
    axes[0].set_xlabel("Evaluation time")
    axes[0].set_ylabel("cNRI")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(TIME_ORDER)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].bar(
        x - width / 2,
        idi_full,
        width,
        yerr=_errors(full_rows, full_boot, "idi"),
        capsize=4,
        color="#356fb3",
        label="Full cohort",
    )
    axes[1].bar(
        x + width / 2,
        idi_wt,
        width,
        yerr=_errors(wt_rows, wt_boot, "idi"),
        capsize=4,
        color="#c44e52",
        label="IDH-wildtype",
    )
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title("Integrated Discrimination Improvement")
    axes[1].set_xlabel("Evaluation time")
    axes[1].set_ylabel("IDI")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(TIME_ORDER)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "nri_idi_plots.png", dpi=200)
    fig.savefig(OUTPUT_DIR / "nri_idi_plots.pdf")
    plt.close(fig)


def _write_report(full_rows: list[dict], full_boot: dict, wt_rows: list[dict], wt_boot: dict) -> None:
    lines = [
        "# NRI and IDI Report\n",
        "## Incremental Value of Spatial Features\n",
        "Time-dependent cNRI and IDI were evaluated at 12, 24, and 36 months. Cases were defined as patients with an observed event by the evaluation time, controls were patients known to be event-free beyond that time, and patients censored before that time were excluded from that time-point-specific comparison. Bootstrap inference used 5,000 resamples.\n",
    ]

    for title, rows, boot in [
        ("Full cohort", full_rows, full_boot),
        ("IDH-wildtype subgroup", wt_rows, wt_boot),
    ]:
        lines.extend(
            [
                f"## {title}\n",
                "| Time | n valid | n case | n control | cNRI | 95% CI | p | IDI | 95% CI | p |",
                "|---|---:|---:|---:|---:|---|---:|---:|---|---:|",
            ]
        )
        for row in _ordered_results(rows):
            time = row["time"]
            lines.append(
                "| "
                f"{time} | {row['n_valid']} | {row['n_case']} | {row['n_control']} | "
                f"{row['c_nri']:.4f} | [{boot[f'c_nri_{time}_ci_lower']:.4f}, {boot[f'c_nri_{time}_ci_upper']:.4f}] | {_format_p(boot[f'c_nri_{time}_p_value'])} | "
                f"{row['idi']:.4f} | [{boot[f'idi_{time}_ci_lower']:.4f}, {boot[f'idi_{time}_ci_upper']:.4f}] | {_format_p(boot[f'idi_{time}_p_value'])} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation\n",
            "Across the full cohort, adding spatial features produced consistently positive and statistically significant cNRI and IDI values at 12, 24, and 36 months. The IDH-wildtype subgroup showed the same directional pattern, with significant cNRI and IDI at all three evaluation times, although uncertainty was wider at later time points because fewer controls remained.\n",
            "The reclassification signal is substantial on the cNRI scale, whereas the absolute discrimination gain on the IDI scale is modest. Together, these results support a significant reclassification benefit with modest incremental discrimination improvement from the spatial feature block.\n",
        ]
    )

    (OUTPUT_DIR / "nri_idi_report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_summary(full_rows: list[dict], full_boot: dict, wt_rows: list[dict], wt_boot: dict) -> None:
    def _result_line(row: dict, boot: dict) -> str:
        time = row["time"]
        return (
            f"{time}: cNRI {row['c_nri']:.4f} "
            f"(95% CI {boot[f'c_nri_{time}_ci_lower']:.4f} to {boot[f'c_nri_{time}_ci_upper']:.4f}, p {_format_p(boot[f'c_nri_{time}_p_value'])}); "
            f"IDI {row['idi']:.4f} "
            f"(95% CI {boot[f'idi_{time}_ci_lower']:.4f} to {boot[f'idi_{time}_ci_upper']:.4f}, p {_format_p(boot[f'idi_{time}_p_value'])})"
        )

    full_text = "; ".join(_result_line(row, full_boot) for row in _ordered_results(full_rows))
    wt_text = "; ".join(_result_line(row, wt_boot) for row in _ordered_results(wt_rows))

    text = f"""# NRI/IDI Summary

## Numerical Findings

Full cohort (n=493): {full_text}.

IDH-wildtype subgroup (n=391): {wt_text}.

## Interpretation

Spatial features provide a significant reclassification benefit beyond the clinical-molecular model. The cNRI results are consistently positive and statistically significant in the full cohort and in the IDH-wildtype subgroup at 12, 24, and 36 months. The IDI results are also significantly positive at all evaluated time points, but the absolute gains are small in magnitude, indicating modest incremental discrimination improvement.

Recommended overall wording: significant reclassification benefit with modest absolute discrimination gain.

## Recommended Manuscript Wording

### Abstract

Adding spatial features to the clinical-molecular Cox model improved time-dependent risk prediction at 12, 24, and 36 months. In the full cohort, continuous net reclassification improvement ranged from 0.4453 to 0.5663 and integrated discrimination improvement ranged from 0.0210 to 0.0250, with bootstrap-supported significance at all time points. Similar findings were observed in the IDH-wildtype subgroup, supporting incremental prognostic value of the spatial feature block.

### Methods

Incremental predictive value of spatial features was assessed by comparing a clinical-molecular Cox model with an expanded clinical-molecular-spatial Cox model in the same 493-patient cohort. Time-dependent continuous net reclassification improvement and integrated discrimination improvement were computed at 12, 24, and 36 months using model-based event probabilities. At each time point, cases were defined as patients with an observed event by that time, controls as patients known to be event-free beyond that time, and patients censored before that time were excluded. Uncertainty was quantified using 5,000 bootstrap resamples to derive 95% confidence intervals and bootstrap p-values. The analysis was repeated in the IDH-wildtype subgroup.

### Results

In the full cohort, adding spatial features improved reclassification at all evaluation times, with cNRI values of 0.4595 at 12 months, 0.4453 at 24 months, and 0.5663 at 36 months; all corresponding bootstrap confidence intervals excluded zero. IDI was also significantly positive at each time point, ranging from 0.0210 to 0.0250. In the IDH-wildtype subgroup, cNRI values ranged from 0.3921 to 0.5631 and IDI values ranged from 0.0159 to 0.0226, again with bootstrap evidence of improvement across all three time points.

### Discussion

These findings indicate that spatial features add complementary prognostic information beyond established clinical and molecular variables. The effect is more pronounced for patient-level reclassification than for absolute separation of predicted risk distributions, suggesting that spatial features primarily improve rank ordering and directional risk assignment rather than producing large shifts in average predicted risk.

### Conclusion

Spatial features confer significant incremental prognostic value when added to the clinical-molecular survival model, with robust gains in time-dependent reclassification and modest but significant improvements in discrimination in both the full cohort and the IDH-wildtype subgroup.
"""

    (DOCS_DIR / "nri_idi_summary.md").write_text(text, encoding="utf-8")


def main() -> int:
    full_payload = _load_json(RESULTS_FULL)
    wt_payload = _load_json(RESULTS_WT)
    full_boot = _load_json(BOOT_FULL)
    wt_boot = _load_json(BOOT_WT)

    full_rows = _ordered_results(full_payload["results"])
    wt_rows = _ordered_results(wt_payload["results"])
    combined_results = _build_combined_results(full_payload, wt_payload)
    combined_bootstrap = _build_combined_bootstrap(full_boot, wt_boot)

    pd.DataFrame(combined_results).to_csv(OUTPUT_DIR / "nri_idi_results.csv", index=False)
    pd.DataFrame(combined_bootstrap).to_csv(OUTPUT_DIR / "nri_idi_bootstrap.csv", index=False)

    _save_json(
        {
            "cohorts": {
                "full_cohort": full_payload,
                "idh_wildtype": wt_payload,
            }
        },
        OUTPUT_DIR / "nri_idi_results.json",
    )
    _save_json(
        {
            "cohorts": {
                "full_cohort": full_boot,
                "idh_wildtype": wt_boot,
            },
            "long_format": combined_bootstrap,
        },
        OUTPUT_DIR / "nri_idi_bootstrap.json",
    )

    _write_report(full_rows, full_boot, wt_rows, wt_boot)
    _write_summary(full_rows, full_boot, wt_rows, wt_boot)
    _make_plots(full_rows, full_boot, wt_rows, wt_boot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
