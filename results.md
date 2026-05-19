# GBM Survival Risk Stratification - Results (Run 3)

Date: 16 May 2026

## 1. Data Summary

- Input: outputs/features_processed.csv
- Samples: 493
- Positives (risk_label=1): 217
- Negatives (risk_label=0): 276
- Feature set: 16 radiomic features

## 2. Holdout Performance (Train/Val/Test)

- Validation balanced accuracy: 0.5714
- Validation macro F1: 0.5716
- Test balanced accuracy: 0.6438
- Test macro F1: 0.6146
- Test AUC: 0.7103

## 3. Final Cross-Validation Performance

- CV: Stratified 10-fold
- Mean AUC: 0.6432
- 95% CI (t-interval): [0.5441, 0.7423]
- One-sided p-value vs AUC=0.5: 0.00485937

## 4. Interpretation vs Abstract Target

The observed mean AUC (0.6432) is close to the abstract target of ~0.652
and is statistically significant above chance at the 0.05 level.
Remaining gap may reflect registration quality or cohort differences.

## 5. Artifacts Produced

- outputs/cv_results.json
- outputs/training_report.txt
- outputs/final_model.json
- outputs/feature_list.json
- outputs/figures/shap_summary.png
- outputs/figures/shap_importance.png
- outputs/figures/confusion_matrix.png
- outputs/metrics/final_cv_summary.json
- outputs/metrics/final_cv_fold_metrics.csv
- outputs/metrics/test_metrics.json
- outputs/metrics/test_classification_report.csv
- outputs/metrics/validation_xgboost_metrics.json
- outputs/metrics/baseline_results.csv
- outputs/reports/training_summary.md



