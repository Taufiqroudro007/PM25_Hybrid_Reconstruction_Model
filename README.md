# Beijing PM2.5 hybrid forecasting reconstruction

## 1. What this project does

This is a clean, reproducible reconstruction of the experiments described in
the submitted IEEE manuscript. It does **not** claim to reproduce the old
numerical tables because the original implementation and prediction files
were unavailable.

The code:
- downloads the official UCI Beijing PM2.5 dataset (dataset ID 381);
- audits row count, timestamp continuity, missingness and chronological splits;
- keeps PM2.5 target values missing in the source as missing (no target imputation);
- builds 24-hour observed PM2.5 history windows;
- uses meteorological variables measured at the forecast origin, not future
  observed weather;
- trains direct forecasting models separately for 1, 3, 6, 12 and 24 hours;
- compares persistence, an empirical temporal Ridge baseline, standalone ML
  models and baseline-plus-residual variants;
- scores every model at a given horizon on identical target timestamps;
- generates expanding-window out-of-fold residuals for hybrid models;
- selects a residual correction multiplier using validation RMSE only;
- writes metrics, aligned predictions, residual-fold audit files, and figures.

## 2. Environment

Recommended: Python 3.10 or 3.11.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## 3. Run

From this project directory:

```bash
python src/run_experiments.py
```

The script downloads the UCI dataset on first run and stores it under
`data/raw/`. If the network or UCI API is unavailable, download the CSV
`PRSA_data_2010.1.1-2014.12.31.csv` from the UCI page and place it in
`data/raw/`, then run again.

## 4. Output files

- `results/dataset_audit.json`: raw dataset and split audit.
- `results/missingness.csv`: missing-value counts by variable.
- `results/all_model_metrics.csv`: all metrics by model and horizon.
- `results/rmse_summary_table.csv`: RMSE pivot table.
- `results/predictions_1h.csv` ... `predictions_24h.csv`: aligned predictions
  and target timestamps for every model.
- `results/rmse_by_horizon.png`: multi-horizon RMSE plot.
- `results/residual_oof_audit_*h.csv`: fold sizes used to construct training residuals.
- `results/preliminary_in_sample/`: outputs from the earlier implementation; these are retained for traceability and must not be reported as results from the corrected pipeline.
- `results/one_hour_model_comparison.png`: one-hour model comparison.

## 5. Important methodological notes

### Chronological split

The script implements the date boundaries stated in the submitted manuscript:
- training: 2010-01-01 through 2013-10-31;
- validation: 2013-11-01 through 2014-05-31;
- test: 2014-06-01 through 2014-12-31.

These dates do not correspond to 70/15/15 of 43,824 raw hourly rows. The audit
reports the actual row counts and percentages. Do not call them 70/15/15.

### Missing target values

The UCI source contains missing PM2.5 labels. The script does not interpolate
or otherwise manufacture target values. A sample is used only if its full
24-hour PM2.5 history and its forecast target are observed. Consequently,
the valid sample count differs by horizon and split and must be reported from
the generated output.

### Forecast-time inputs

For a forecast issued at time t, predictors include PM2.5 observations through
t, meteorological measurements available at t, and calendar variables for
the target time. The script does not use weather measurements observed at
t+h. In an operational deployment, weather forecasts would be needed to use
future meteorology.

### Mathematical interpretation

The Ridge model is an empirical autoregressive/meteorological baseline
motivated by transport dynamics. It is not a spatial finite-difference
discretization of the 3-D advection-diffusion-reaction PDE. The single-station
dataset does not contain spatial concentration gradients needed to validate
that discretization.

### Residual hybrid

For each horizon, Ridge residual targets are generated using expanding-window,
time-ordered out-of-fold predictions. Each OOF block is predicted by a Ridge
pipeline fitted only on earlier training observations; therefore, a training
observation is not used to fit the baseline that predicts it. The residual
learner is fitted to these OOF residuals. The final Ridge baseline is then
refitted on the complete training period.

A residual correction multiplier is selected from `{0, 0.25, 0.50, 0.75, 1.0}`
by minimizing validation RMSE. The selected multiplier is applied unchanged
to test predictions. Test labels are not used for fitting or model selection.
The OOF fold sizes and positions are saved in a horizon-specific audit CSV.

### Metrics

MAPE is calculated only for nonzero observed targets because percentage error
is undefined at zero. The output includes the number of nonzero observations
used for MAPE. MAE, RMSE and R² are calculated on the same target timestamps
for every model at each horizon.

### Limits of this implementation

The base estimators use fixed, transparent hyperparameters; the validation
set selects only the residual correction multiplier, not the estimator
hyperparameters. This implementation does not yet include XGBoost, LSTM, or
rolling-origin evaluation. Do not retain the old LSTM metrics or claim that
these experiments reproduce them.

## 6. Dataset citation

Chen, S. (2015). *Beijing PM2.5*. UCI Machine Learning Repository.
https://doi.org/10.24432/C5JS49
