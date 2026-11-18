"""
Reproducible reconstruction of the Beijing PM2.5 forecasting experiments.

Design decisions:
- Direct forecasts are trained separately for each horizon (1, 3, 6, 12, 24 h).
- Only PM2.5 values actually observed in the source data are used as labels.
- No interpolation of the target is performed.
- Predictors use PM2.5 history through the forecast origin and meteorology at
  the forecast origin. Calendar variables at the target time are known in advance.
- Every model for a given horizon is evaluated on the same target timestamps.
- The mathematical component is an empirical autoregressive baseline, not a
  spatial finite-difference solution of the atmospheric PDE.

Run from the project root:
    python src/run_experiments.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 3, 6, 12, 24]
LOOKBACK = 24

# Chronological boundaries stated in the submitted manuscript.
TRAIN_END = pd.Timestamp("2013-10-31 23:00:00")
VAL_START = pd.Timestamp("2013-11-01 00:00:00")
VAL_END = pd.Timestamp("2014-05-31 23:00:00")
TEST_START = pd.Timestamp("2014-06-01 00:00:00")
TEST_END = pd.Timestamp("2014-12-31 23:00:00")

WIND_CATEGORIES = ["NE", "NW", "SE", "SW", "cv"]


def load_uci_dataset() -> pd.DataFrame:
    """Download the authoritative UCI dataset using ucimlrepo."""
    csv_path = DATA_DIR / "PRSA_data_2010.1.1-2014.12.31.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        dataset = fetch_ucirepo(id=381)
        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()
        df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        df.to_csv(csv_path, index=False)

    # Normalize column names without changing their meaning.
    df.columns = [str(c).strip() for c in df.columns]
    if "pm2.5" not in df.columns:
        candidates = [c for c in df.columns if c.lower().replace("_", "").replace(".", "") in
                      {"pm25", "pm2.5"}]
        if candidates:
            df = df.rename(columns={candidates[0]: "pm2.5"})
    required = ["year", "month", "day", "hour", "pm2.5",
                "DEWP", "TEMP", "PRES", "cbwd", "Iws", "Is", "Ir"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    df["datetime"] = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=df["day"], hour=df["hour"]),
        errors="coerce"
    )
    df["pm2.5"] = pd.to_numeric(df["pm2.5"], errors="coerce")
    for c in ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def audit_dataset(df: pd.DataFrame) -> None:
    """Write factual data checks before any model fitting."""
    timestamps = df["datetime"]
    hourly_diffs = timestamps.diff().dropna()
    audit = {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "first_timestamp": str(timestamps.min()),
        "last_timestamp": str(timestamps.max()),
        "duplicate_timestamps": int(timestamps.duplicated().sum()),
        "missing_pm25": int(df["pm2.5"].isna().sum()),
        "missing_pm25_percent": float(100 * df["pm2.5"].isna().mean()),
        "missing_by_column": {k: int(v) for k, v in df.isna().sum().items()},
        "non_hourly_gaps": int((hourly_diffs != pd.Timedelta(hours=1)).sum()),
        "unique_wind_directions": sorted(df["cbwd"].dropna().astype(str).unique().tolist()),
    }

    # Raw chronological split counts, independent of target availability.
    periods = {
        "train": (timestamps <= TRAIN_END),
        "validation": ((timestamps >= VAL_START) & (timestamps <= VAL_END)),
        "test": ((timestamps >= TEST_START) & (timestamps <= TEST_END)),
    }
    audit["chronological_split"] = {
        name: {
            "rows": int(mask.sum()),
            "percentage_of_full_dataset": float(100 * mask.mean()),
            "valid_pm25_targets": int(df.loc[mask, "pm2.5"].notna().sum()),
        }
        for name, mask in periods.items()
    }
    with open(RESULTS_DIR / "dataset_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    pd.DataFrame({
        "column": df.columns,
        "missing_count": [int(df[c].isna().sum()) for c in df.columns],
        "missing_percent": [100 * df[c].isna().mean() for c in df.columns],
    }).to_csv(RESULTS_DIR / "missingness.csv", index=False)

    print("\nDATASET AUDIT")
    print(json.dumps(audit, indent=2))


def make_feature_row(df: pd.DataFrame, origin_idx: int, target_idx: int) -> dict:
    """Features available at forecast origin; target-time calendar is known."""
    origin = df.iloc[origin_idx]
    target = df.iloc[target_idx]

    row = {}
    # The most recent observation is lag 1; oldest is lag 24.
    for lag in range(1, LOOKBACK + 1):
        row[f"pm_lag_{lag}"] = float(df.iloc[origin_idx - lag + 1]["pm2.5"])

    # Weather is restricted to the forecast origin; no future observed weather.
    for col in ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]:
        row[f"{col}_origin"] = float(origin[col])

    wind = str(origin["cbwd"])
    for category in WIND_CATEGORIES:
        row[f"cbwd_{category}"] = float(wind == category)

    # Calendar features at the target time are deterministic and known.
    hour = int(target["datetime"].hour)
    month = int(target["datetime"].month)
    row["target_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    row["target_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    row["target_month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    row["target_month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    return row


def build_supervised_data(df: pd.DataFrame, horizon: int):
    """Create direct-horizon examples; discard missing target/history windows."""
    X_rows, y_values, target_times, origin_times = [], [], [], []
    n = len(df)

    for origin_idx in range(LOOKBACK - 1, n - horizon):
        target_idx = origin_idx + horizon
        origin_time = df.iloc[origin_idx]["datetime"]
        target_time = df.iloc[target_idx]["datetime"]

        # Guard against any missing or irregular hourly timestamps.
        if target_time - origin_time != pd.Timedelta(hours=horizon):
            continue

        history = df.iloc[origin_idx - LOOKBACK + 1:origin_idx + 1]["pm2.5"]
        target_value = df.iloc[target_idx]["pm2.5"]

        # Do not interpolate PM2.5. A window is valid only when all 24
        # observed history values and the future label are present.
        if history.isna().any() or pd.isna(target_value):
            continue

        row = make_feature_row(df, origin_idx, target_idx)
        if not np.isfinite(list(row.values())).all():
            continue

        X_rows.append(row)
        y_values.append(float(target_value))
        target_times.append(target_time)
        origin_times.append(origin_time)

    X = pd.DataFrame(X_rows).astype(float)
    y = pd.Series(y_values, name="pm25_target", dtype=float)
    meta = pd.DataFrame({
        "origin_time": origin_times,
        "target_time": target_times,
    })
    return X, y, meta


def split_by_target_time(X, y, meta):
    t = meta["target_time"]
    train = t <= TRAIN_END
    val = (t >= VAL_START) & (t <= VAL_END)
    test = (t >= TEST_START) & (t <= TEST_END)
    return {
        "train": (X.loc[train].reset_index(drop=True),
                  y.loc[train].reset_index(drop=True),
                  meta.loc[train].reset_index(drop=True)),
        "validation": (X.loc[val].reset_index(drop=True),
                      y.loc[val].reset_index(drop=True),
                      meta.loc[val].reset_index(drop=True)),
        "test": (X.loc[test].reset_index(drop=True),
                 y.loc[test].reset_index(drop=True),
                 meta.loc[test].reset_index(drop=True)),
    }


def model_definitions():
    """Fixed, transparent model configurations; no test-set tuning."""
    scaled = lambda estimator: Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", estimator),
    ])
    unscaled = lambda estimator: Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", estimator),
    ])

    return {
        "Temporal_Ridge": scaled(Ridge(alpha=1.0)),
        "Standalone_SVR": scaled(SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")),
        "Standalone_RF": unscaled(RandomForestRegressor(
            n_estimators=300, max_depth=18, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        )),
        "Standalone_ExtraTrees": unscaled(ExtraTreesRegressor(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        )),
        "Standalone_HistGB": unscaled(HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=250, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=42
        )),
    }


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    nonzero = np.abs(y_true) > 1e-8
    mape_nonzero = (
        float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
        if nonzero.any() else np.nan
    )
    return {
        "n": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_nonzero_percent": mape_nonzero,
        "n_nonzero_for_MAPE": int(nonzero.sum()),
    }


def run_one_horizon(df, horizon):
    X, y, meta = build_supervised_data(df, horizon)
    splits = split_by_target_time(X, y, meta)

    Xtr, ytr, mtr = splits["train"]
    Xv, yv, mv = splits["validation"]
    Xte, yte, mte = splits["test"]

    if min(len(Xtr), len(Xv), len(Xte)) == 0:
        raise RuntimeError(f"Empty split at horizon {horizon}; inspect data/split definitions.")

    print(f"\nHORIZON {horizon} h: train={len(Xtr)}, val={len(Xv)}, test={len(Xte)}")

    predictions = []
    metrics = []

    models = model_definitions()

    # Fit baseline on training only.
    baseline = models["Temporal_Ridge"]
    baseline.fit(Xtr, ytr)
    base_train_pred = baseline.predict(Xtr)
    base_val_pred = baseline.predict(Xv)
    base_test_pred = baseline.predict(Xte)

    # Baseline test prediction.
    pred_frame = mte.copy()
    pred_frame["y_true"] = yte.to_numpy()
    pred_frame["Temporal_Ridge"] = base_test_pred
    metrics.append({
        "horizon_hours": horizon,
        "model": "Temporal_Ridge",
        **calculate_metrics(yte, base_test_pred)
    })

    # Standalone ML models and baseline-plus-residual variants.
    for name, model in models.items():
        if name == "Temporal_Ridge":
            continue

        model.fit(Xtr, ytr)
        val_pred = model.predict(Xv)
        test_pred = model.predict(Xte)

        pred_frame[name] = test_pred
        metrics.append({
            "horizon_hours": horizon,
            "model": name,
            **calculate_metrics(yte, test_pred)
        })

        # Residual learner is trained against baseline residuals on training data.
        # The residual target is calculated only from training labels and
        # in-sample baseline predictions. Validation is used for diagnosis only.
        residual_train = ytr.to_numpy() - base_train_pred
        residual_model = clone(model)
        residual_model.fit(Xtr, residual_train)
        hybrid_test_pred = base_test_pred + residual_model.predict(Xte)

        hybrid_name = name.replace("Standalone_", "Hybrid_")
        pred_frame[hybrid_name] = hybrid_test_pred
        metrics.append({
            "horizon_hours": horizon,
            "model": hybrid_name,
            **calculate_metrics(yte, hybrid_test_pred)
        })

    # Persistence forecast: last observed PM2.5 at forecast origin.
    persistence = Xte["pm_lag_1"].to_numpy()
    pred_frame["Persistence"] = persistence
    metrics.append({
        "horizon_hours": horizon,
        "model": "Persistence",
        **calculate_metrics(yte, persistence)
    })

    # Ensure every model is scored on exactly the same target timestamps.
    assert pred_frame["target_time"].is_monotonic_increasing
    assert pred_frame["target_time"].is_unique

    pred_frame.to_csv(RESULTS_DIR / f"predictions_{horizon}h.csv", index=False)
    return metrics, pred_frame


def make_figures(metrics_df):
    # RMSE by forecast horizon.
    fig, ax = plt.subplots(figsize=(9, 5))
    for model in ["Temporal_Ridge", "Standalone_RF", "Standalone_HistGB",
                  "Hybrid_RF", "Hybrid_HistGB", "Persistence"]:
        sub = metrics_df[metrics_df["model"] == model].sort_values("horizon_hours")
        if not sub.empty:
            ax.plot(sub["horizon_hours"], sub["RMSE"], marker="o", label=model)
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("RMSE (µg/m³)")
    ax.set_title("Direct multi-horizon PM2.5 forecasting")
    ax.set_xticks(HORIZONS)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "rmse_by_horizon.png", dpi=300)
    plt.close(fig)

    # One-hour comparison.
    one = metrics_df[metrics_df["horizon_hours"] == 1].sort_values("RMSE")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(one["model"], one["RMSE"])
    ax.set_ylabel("RMSE (µg/m³)")
    ax.set_title("One-hour-ahead model comparison")
    ax.tick_params(axis="x", rotation=55)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "one_hour_model_comparison.png", dpi=300)
    plt.close(fig)


def main():
    df = load_uci_dataset()
    audit_dataset(df)

    # No target imputation: missing labels remain missing and are excluded
    # from supervised samples. This avoids creating synthetic ground truth.
    all_metrics = []
    for horizon in HORIZONS:
        horizon_metrics, _ = run_one_horizon(df, horizon)
        all_metrics.extend(horizon_metrics)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(RESULTS_DIR / "all_model_metrics.csv", index=False)

    # Compact table for paper use.
    pivot = metrics_df.pivot_table(
        index="model", columns="horizon_hours", values="RMSE", aggfunc="first"
    )
    pivot.to_csv(RESULTS_DIR / "rmse_summary_table.csv")

    make_figures(metrics_df)

    print("\nRESULTS SAVED TO:", RESULTS_DIR)
    print(metrics_df.sort_values(["horizon_hours", "RMSE"]).to_string(index=False))
    print("\nIMPORTANT: inspect dataset_audit.json and the prediction CSVs before "
          "writing any manuscript sample counts or performance claims.")


if __name__ == "__main__":
    main()
