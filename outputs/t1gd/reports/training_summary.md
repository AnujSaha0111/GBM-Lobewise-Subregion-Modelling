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

- Validation balanced accuracy: `0.5796`
- Validation macro F1: `0.5795`
- Best Optuna CV balanced accuracy: `0.5965`
- Best params: `{'n_estimators': 283, 'learning_rate': 0.013940346079873234, 'max_depth': 8, 'subsample': 0.7200762468698007, 'colsample_bytree': 0.5610191174223894, 'min_child_weight': 5, 'gamma': 0.17194260557609198, 'reg_alpha': 3.5204810455260365, 'reg_lambda': 0.0019674328025306126}`

## Final Test Metrics

- Balanced accuracy: `0.6349`
- Macro F1: `0.6205`
- Quadratic kappa: `0.2605`
- AUC: `0.6644`

## Final Cross-Validation

- AUC: `0.6412`
- 95% CI: `0.5334` to `0.7490`
- One-sided p-value vs 0.5: `0.0079553`
