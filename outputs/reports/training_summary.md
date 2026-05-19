# GBM Training Summary

## Dataset

- Rows after label cleaning: `493`
- Imaging features used: `16`
- Class distribution: `{0: 276, 1: 217}`
- Total feature nulls at training input: `0`

## Runtime

- Selected device: `cuda`
- GPU available: `True`
- Device note: `GPU training available`
- Optuna trials run: `50`

## Baselines

- Best validation baseline: `LogisticRegression`
- Best baseline balanced accuracy: `0.6347`
- Best baseline macro F1: `0.6356`

## XGBoost

- Validation balanced accuracy: `0.5714`
- Validation macro F1: `0.5716`
- Best Optuna CV balanced accuracy: `0.6071`
- Best params: `{'n_estimators': 249, 'learning_rate': 0.011569046886474444, 'max_depth': 8, 'subsample': 0.7347049478829597, 'colsample_bytree': 0.5496480501205234, 'min_child_weight': 5, 'gamma': 1.3170773995337703, 'reg_alpha': 3.717642135537067, 'reg_lambda': 0.0023232967562617536}`

## Final Test Metrics

- Balanced accuracy: `0.6438`
- Macro F1: `0.6146`
- Quadratic kappa: `0.2730`
- AUC: `0.7103`

## Final Cross-Validation

- AUC: `0.6432`
- 95% CI: `0.5441` to `0.7423`
- One-sided p-value vs 0.5: `0.00485937`
