# Residual Learning with a Meteorologically Informed Temporal Baseline for Short-Term PM2.5 Forecasting

## 1. Overview
This repository provides the reproducible implementation, audit logs, and experimental outputs for the IEEE conference paper:

> **Residual Learning with a Meteorologically Informed Temporal Baseline for Short-Term PM2.5 Forecasting**  
> M Taufiqul Islam Roudro and Mahmudul Hasan Zihad  
> *2026 International Conference on Recent Progresses in Science, Engineering and Technology (ICRPSET)*

The pipeline:
- Downloads and audits the official UCI Beijing PM2.5 dataset (Dataset ID 381; 43,824 hourly observations from January 2010 through December 2014).
- Retains missing target values as missing without interpolation; samples are included only when the full 24-hour PM2.5 history and target observation are available.
- Uses meteorological measurements available at the forecast origin $t$ together with deterministic calendar encodings at the target time $t+h$ (no future weather observations are used).
- Trains direct forecasting models separately for 1-, 3-, 6-, 12-, and 24-hour horizons.
- Evaluates 14 approaches on identical target timestamps: Persistence, Temporal Ridge Baseline, six standalone learners (SVR, Random Forest, Extra Trees, HistGradientBoosting, XGBoost, and LSTM), and six corresponding residual-corrected hybrid models.
- Generates expanding-window chronological out-of-fold (OOF) residuals with temporal purging to prevent target leakage, and selects the residual correction multiplier $\lambda_h \in \{0, 0.25, 0.50, 0.75, 1.0\}$ using validation RMSE only.
- Performs paired moving-block bootstrap comparisons across 24-, 48-, and 72-hour block lengths (2,000 replicates, random seed `42`).

---

## 2. Environment Setup
Recommended: Python 3.10 or 3.11.

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt

```

---

## 3. Running the Experiments

From the repository root directory:

1. **Train all models, run chronological validation, and generate test predictions:**
```bash
python src/run_experiments_lstm.py

```


The script automatically downloads the UCI dataset on first run and stores it under `data/raw/`. If the network or UCI API is unavailable, download `PRSA_data_2010.1.1-2014.12.31.csv` from the UCI dataset page and place it in `data/raw/` before running.
2. **Run paired moving-block bootstrap statistical comparisons:**
```bash
python src/paired_time_block_bootstrap.py

```


3. **Generate Fig. 1 ($\Delta$RMSE relative to Temporal Ridge):**
```bash
python src/plot_rmse_delta_onecol.py

```



---

## 4. Repository Structure & Output Files

* `src/run_experiments_lstm.py`: Main training, chronological OOF residual generation, validation multiplier selection, and test evaluation pipeline.
* `src/paired_time_block_bootstrap.py`: Paired moving-block bootstrap resampling across 24-, 48-, and 72-hour blocks.
* `src/plot_rmse_delta_onecol.py`: Generates the IEEE single-column $\Delta$RMSE comparison figure (`figure_rmse_delta_onecol.pdf` and `figure_rmse_delta_onecol.png`).
* `results_lstm/dataset_audit.json`: Raw dataset and chronological split audit.
* `results_lstm/missingness.csv`: Missing-value counts and percentages by variable.
* `results_lstm/all_model_metrics.csv`: MAE, RMSE, and $R^2$ across all 14 approaches and 5 horizons.
* `results_lstm/rmse_summary_table.csv`: Pivot summary table of test RMSE values.
* `results_lstm/predictions_1h.csv` ... `predictions_24h.csv`: Aligned test predictions and ground-truth targets for every model.
* `results_lstm/residual_oof_audit_1h.csv` ... `residual_oof_audit_24h.csv`: Chronological OOF fold sizes, boundaries, and purge counts.
* `results_lstm/paired_block_bootstrap_recomputed.csv`: Paired bootstrap $\Delta$RMSE estimates and 95% confidence intervals across all tested block lengths.

---

## 5. Methodological Notes

* **Chronological Partitioning:**
* **Training:** `2010-01-01` through `2013-10-31` (33,600 raw hourly rows; 76.67%)
* **Validation:** `2013-11-01` through `2014-05-31` (5,088 raw hourly rows; 11.61%)
* **Testing:** `2014-06-01` through `2014-12-31` (5,136 raw hourly rows; 11.72%)


* **Missing Target Handling:** Missing PM2.5 labels in the UCI dataset are never interpolated. A supervised sample is retained only when all 24 historical PM2.5 values and the target observation at $t+h$ are present, resulting in horizon-specific valid sample counts reported in `results_lstm/dataset_audit.json`.
* **Temporal Purging:** For each expanding-window OOF block, a copy of the Ridge baseline is fitted solely on earlier training samples whose target timestamp $t+h$ is strictly earlier than the first forecast origin of that block, removing up to $h$ preceding origin positions so baseline training targets never overlap OOF forecast periods.
* **Fixed Hyperparameters:** All learners use fixed, transparent hyperparameters with random seed `42`; validation data is used strictly to select $\lambda_h \in \{0, 0.25, 0.50, 0.75, 1.0\}$.

---

## 6. Dataset Citation

Chen, S. (2015). *Beijing PM2.5* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5JS49

```

```
