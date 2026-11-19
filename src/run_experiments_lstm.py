
"""
Reproducible reconstruction of the Beijing PM2.5 forecasting experiments.

Design decisions:
- Direct forecasts are trained separately for each horizon (1, 3, 6, 12, 24 h).
- Only PM2.5 values actually observed in the source data are used as labels.
- No interpolation of the target is performed.
- Predictors use PM2.5 history through the forecast origin and meteorology at
  the forecast origin. Calendar variables at the target time are known in advance.
- Hybrid residual learners use expanding-window, time-ordered out-of-fold residuals.
- A residual correction multiplier is selected using causally valid validation data.
- Every model for a given horizon is evaluated on the same target timestamps.
- LSTM internal early stopping uses chronological validation with a
  horizon-dependent temporal purge.
- The mathematical component is an empirical autoregressive baseline, not a
  spatial finite-difference solution of the atmospheric PDE.

Run from the project root:
    python src/run_experiments_lstm.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ucimlrepo import fetch_ucirepo

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from xgboost import XGBRegressor


warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results_lstm"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 3, 6, 12, 24]
LOOKBACK = 24

TRAIN_END = pd.Timestamp("2013-10-31 23:00:00")

VAL_START = pd.Timestamp("2013-11-01 00:00:00")
VAL_END = pd.Timestamp("2014-05-31 23:00:00")

TEST_START = pd.Timestamp("2014-06-01 00:00:00")
TEST_END = pd.Timestamp("2014-12-31 23:00:00")

WIND_CATEGORIES = ["NE", "NW", "SE", "SW", "cv"]

RANDOM_STATE = 42


# ============================================================
# 2. DATA LOADING AND AUDIT
# ============================================================

def load_uci_dataset() -> pd.DataFrame:
    """Download or load the Beijing PM2.5 dataset."""

    csv_path = DATA_DIR / "PRSA_data_2010.1.1-2014.12.31.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        dataset = fetch_ucirepo(id=381)

        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()

        df = pd.concat(
            [X.reset_index(drop=True), y.reset_index(drop=True)],
            axis=1,
        )

        df.to_csv(csv_path, index=False)

    df.columns = [str(c).strip() for c in df.columns]

    if "pm2.5" not in df.columns:
        candidates = [
            c for c in df.columns
            if c.lower().replace("_", "").replace(".", "")
            in {"pm25", "pm2.5"}
        ]

        if candidates:
            df = df.rename(columns={candidates[0]: "pm2.5"})

    required = [
        "year", "month", "day", "hour", "pm2.5",
        "DEWP", "TEMP", "PRES", "cbwd", "Iws", "Is", "Ir",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Dataset is missing expected columns: {missing}"
        )

    df["datetime"] = pd.to_datetime(
        dict(
            year=df["year"],
            month=df["month"],
            day=df["day"],
            hour=df["hour"],
        ),
        errors="coerce",
    )

    df["pm2.5"] = pd.to_numeric(
        df["pm2.5"], errors="coerce"
    )

    for col in ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    return df


def audit_dataset(df: pd.DataFrame) -> None:
    """Write factual dataset checks before model fitting."""

    timestamps = df["datetime"]
    hourly_diffs = timestamps.diff().dropna()

    audit = {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "first_timestamp": str(timestamps.min()),
        "last_timestamp": str(timestamps.max()),
        "duplicate_timestamps": int(timestamps.duplicated().sum()),
        "missing_pm25": int(df["pm2.5"].isna().sum()),
        "missing_pm25_percent": float(
            100 * df["pm2.5"].isna().mean()
        ),
        "missing_by_column": {
            k: int(v) for k, v in df.isna().sum().items()
        },
        "non_hourly_gaps": int(
            (hourly_diffs != pd.Timedelta(hours=1)).sum()
        ),
        "unique_wind_directions": sorted(
            df["cbwd"].dropna().astype(str).unique().tolist()
        ),
    }

    periods = {
        "train": timestamps <= TRAIN_END,
        "validation": (
            (timestamps >= VAL_START)
            & (timestamps <= VAL_END)
        ),
        "test": (
            (timestamps >= TEST_START)
            & (timestamps <= TEST_END)
        ),
    }

    audit["chronological_split"] = {
        name: {
            "rows": int(mask.sum()),
            "percentage_of_full_dataset": float(
                100 * mask.mean()
            ),
            "valid_pm25_targets": int(
                df.loc[mask, "pm2.5"].notna().sum()
            ),
        }
        for name, mask in periods.items()
    }

    with open(
        RESULTS_DIR / "dataset_audit.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(audit, f, indent=2)

    pd.DataFrame({
        "column": df.columns,
        "missing_count": [
            int(df[c].isna().sum()) for c in df.columns
        ],
        "missing_percent": [
            100 * df[c].isna().mean() for c in df.columns
        ],
    }).to_csv(
        RESULTS_DIR / "missingness.csv",
        index=False,
    )

    print("\nDATASET AUDIT")
    print(json.dumps(audit, indent=2))


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

def make_feature_row(
    df: pd.DataFrame,
    origin_idx: int,
    target_idx: int,
) -> dict:
    """Construct features available at the forecast origin."""

    origin = df.iloc[origin_idx]
    target = df.iloc[target_idx]

    row = {}

    # PM2.5 history: lag 1 is the most recent observation.
    for lag in range(1, LOOKBACK + 1):
        row[f"pm_lag_{lag}"] = float(
            df.iloc[origin_idx - lag + 1]["pm2.5"]
        )

    # Meteorological variables available at forecast origin.
    for col in ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]:
        row[f"{col}_origin"] = float(origin[col])

    # Wind direction at forecast origin.
    wind = str(origin["cbwd"])

    for category in WIND_CATEGORIES:
        row[f"cbwd_{category}"] = float(wind == category)

    # Deterministic calendar variables at target time.
    hour = int(target["datetime"].hour)
    month = int(target["datetime"].month)

    row["target_hour_sin"] = np.sin(
        2 * np.pi * hour / 24
    )
    row["target_hour_cos"] = np.cos(
        2 * np.pi * hour / 24
    )

    row["target_month_sin"] = np.sin(
        2 * np.pi * (month - 1) / 12
    )
    row["target_month_cos"] = np.cos(
        2 * np.pi * (month - 1) / 12
    )

    return row


def build_supervised_data(
    df: pd.DataFrame,
    horizon: int,
):
    """Build direct-horizon supervised examples."""

    X_rows = []
    y_values = []
    target_times = []
    origin_times = []

    n = len(df)

    for origin_idx in range(LOOKBACK - 1, n - horizon):

        target_idx = origin_idx + horizon

        origin_time = df.iloc[origin_idx]["datetime"]
        target_time = df.iloc[target_idx]["datetime"]

        # Verify the requested forecast horizon.
        if target_time - origin_time != pd.Timedelta(
            hours=horizon
        ):
            continue

        history = df.iloc[
            origin_idx - LOOKBACK + 1:origin_idx + 1
        ]["pm2.5"]

        target_value = df.iloc[target_idx]["pm2.5"]

        # No target interpolation or synthetic labels.
        if history.isna().any() or pd.isna(target_value):
            continue

        row = make_feature_row(
            df,
            origin_idx,
            target_idx,
        )

        if not np.isfinite(list(row.values())).all():
            continue

        X_rows.append(row)
        y_values.append(float(target_value))
        target_times.append(target_time)
        origin_times.append(origin_time)

    X = pd.DataFrame(X_rows).astype(float)

    y = pd.Series(
        y_values,
        name="pm25_target",
        dtype=float,
    )

    meta = pd.DataFrame({
        "origin_time": origin_times,
        "target_time": target_times,
    })

    return X, y, meta


def split_by_target_time(X, y, meta):
    """Create chronological partitions based on target timestamps."""

    t = meta["target_time"]

    train = t <= TRAIN_END

    val = (
        (t >= VAL_START)
        & (t <= VAL_END)
    )

    test = (
        (t >= TEST_START)
        & (t <= TEST_END)
    )

    return {
        "train": (
            X.loc[train].reset_index(drop=True),
            y.loc[train].reset_index(drop=True),
            meta.loc[train].reset_index(drop=True),
        ),

        "validation": (
            X.loc[val].reset_index(drop=True),
            y.loc[val].reset_index(drop=True),
            meta.loc[val].reset_index(drop=True),
        ),

        "test": (
            X.loc[test].reset_index(drop=True),
            y.loc[test].reset_index(drop=True),
            meta.loc[test].reset_index(drop=True),
        ),
    }


# ============================================================
# 4. LSTM ARCHITECTURE
# ============================================================

class _LSTMNetwork(nn.Module):
    """LSTM sequence encoder with a separate static-feature branch."""

    def __init__(
        self,
        static_dim,
        hidden_size=32,
        dropout=0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.network = nn.Sequential(
            nn.Linear(hidden_size + static_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, sequence, static_features):

        _, (hidden, _) = self.lstm(sequence)

        temporal_features = hidden[-1]

        combined = torch.cat(
            [temporal_features, static_features],
            dim=1,
        )

        return self.network(combined)


# ============================================================
# 5. SKLEARN-COMPATIBLE LSTM REGRESSOR
# ============================================================

class TorchLSTMRegressor(RegressorMixin, BaseEstimator):
    """
    CPU LSTM regressor.

    Internal early stopping:
    - Uses the final 10% of the supplied training partition.
    - Applies a horizon-dependent temporal purge.
    - Fits preprocessing only on the internal training portion.
    - Refits preprocessing on the complete training partition
      before fitting the final network.
    """

    def __init__(
        self,
        hidden_size=32,
        dropout=0.2,
        learning_rate=0.001,
        batch_size=256,
        max_epochs=100,
        patience=12,
        random_state=42,
        horizon_hours=1,
    ):
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        self.horizon_hours = horizon_hours

    def _prepare_features(self, X, fit=False):

        sequence_columns = [
            f"pm_lag_{lag}"
            for lag in range(LOOKBACK, 0, -1)
        ]

        self.static_columns_ = [
            col for col in X.columns
            if col not in sequence_columns
        ]

        sequence = X[sequence_columns].to_numpy(
            dtype=np.float32
        )

        sequence = sequence.reshape(
            -1, LOOKBACK, 1
        )

        static = X[self.static_columns_].to_numpy(
            dtype=np.float32
        )

        if fit:
            self.sequence_scaler_ = StandardScaler()
            self.static_scaler_ = StandardScaler()

            self.sequence_scaler_.fit(
                sequence.reshape(-1, 1)
            )

            self.static_scaler_.fit(static)

        sequence = self.sequence_scaler_.transform(
            sequence.reshape(-1, 1)
        ).reshape(-1, LOOKBACK, 1)

        static = self.static_scaler_.transform(static)

        return (
            torch.tensor(
                sequence,
                dtype=torch.float32,
            ),
            torch.tensor(
                static,
                dtype=torch.float32,
            ),
        )

    def _new_network(self):

        torch.manual_seed(self.random_state)

        return _LSTMNetwork(
            static_dim=len(self.static_columns_),
            hidden_size=self.hidden_size,
            dropout=self.dropout,
        )

    def _train_network(
        self,
        sequence,
        static,
        targets,
        epochs,
    ):

        network = self._new_network()

        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=self.learning_rate,
        )

        loss_function = nn.MSELoss()

        dataset = TensorDataset(
            sequence,
            static,
            targets,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        for _ in range(epochs):

            network.train()

            for seq_batch, static_batch, y_batch in loader:

                optimizer.zero_grad()

                predictions = network(
                    seq_batch,
                    static_batch,
                )

                loss = loss_function(
                    predictions,
                    y_batch,
                )

                loss.backward()
                optimizer.step()

        return network

    def fit(self, X, y):

        torch.manual_seed(self.random_state)

        n = len(X)

        # Chronological internal validation: final 10%.
        validation_size = max(
            1,
            int(0.10 * n),
        )

        validation_start = n - validation_size

        # Remove at least horizon_hours supervised observations
        # immediately before the internal validation block.
        #
        # Because supervised samples are ordered and timestamps
        # are hourly, this ensures that the final internal training
        # target precedes the first internal validation origin.
        training_size = (
            validation_start - self.horizon_hours
        )

        if training_size < 2:
            raise ValueError(
                "Insufficient samples for purged LSTM validation."
            )

        y_array = np.asarray(
            y,
            dtype=np.float32,
        ).reshape(-1, 1)

        # ----------------------------------------------------
        # Internal early-stopping preprocessing
        # ----------------------------------------------------

        # Fit feature scalers only on the internal training data.
        train_seq, train_static = self._prepare_features(
            X.iloc[:training_size],
            fit=True,
        )

        # Transform internal validation using training scalers.
        val_seq, val_static = self._prepare_features(
            X.iloc[validation_start:],
            fit=False,
        )

        self.target_scaler_ = StandardScaler()

        train_targets = torch.tensor(
            self.target_scaler_.fit_transform(
                y_array[:training_size]
            ),
            dtype=torch.float32,
        )

        val_targets = torch.tensor(
            self.target_scaler_.transform(
                y_array[validation_start:]
            ),
            dtype=torch.float32,
        )

        train_dataset = TensorDataset(
            train_seq,
            train_static,
            train_targets,
        )

        loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        network = self._new_network()

        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=self.learning_rate,
        )

        loss_function = nn.MSELoss()

        best_loss = float("inf")
        best_epoch = 1
        patience_counter = 0

        # ----------------------------------------------------
        # Chronological early stopping
        # ----------------------------------------------------

        for epoch in range(1, self.max_epochs + 1):

            network.train()

            for seq_batch, static_batch, y_batch in loader:

                optimizer.zero_grad()

                predictions = network(
                    seq_batch,
                    static_batch,
                )

                loss = loss_function(
                    predictions,
                    y_batch,
                )

                loss.backward()
                optimizer.step()

            network.eval()

            with torch.no_grad():

                val_predictions = network(
                    val_seq,
                    val_static,
                )

                val_loss = loss_function(
                    val_predictions,
                    val_targets,
                ).item()

            if val_loss < best_loss - 1e-6:

                best_loss = val_loss
                best_epoch = epoch
                patience_counter = 0

            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        # ----------------------------------------------------
        # Final refit using the complete supplied training set
        # ----------------------------------------------------

        # Refit feature scalers using all available training samples.
        sequence, static = self._prepare_features(
            X,
            fit=True,
        )

        # Refit target scaler using all available training targets.
        self.target_scaler_ = StandardScaler()

        targets = torch.tensor(
            self.target_scaler_.fit_transform(y_array),
            dtype=torch.float32,
        )

        # Train final network for the selected number of epochs.
        self.network_ = self._train_network(
            sequence,
            static,
            targets,
            best_epoch,
        )

        self.best_epoch_ = best_epoch
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):

        self.network_.eval()

        sequence, static = self._prepare_features(
            X,
            fit=False,
        )

        with torch.no_grad():

            predictions = self.network_(
                sequence,
                static,
            ).numpy()

        return self.target_scaler_.inverse_transform(
            predictions
        ).ravel()


# ============================================================
# 6. MODEL DEFINITIONS
# ============================================================

def model_definitions(horizon):
    """Fixed model configurations; no test-set tuning."""

    def scaled(estimator):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", estimator),
        ])

    def unscaled(estimator):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ])

    return {

        "Temporal_Ridge": scaled(
            Ridge(alpha=1.0)
        ),

        "Standalone_SVR": scaled(
            SVR(
                kernel="rbf",
                C=10.0,
                epsilon=0.1,
                gamma="scale",
            )
        ),

        "Standalone_RF": unscaled(
            RandomForestRegressor(
                n_estimators=300,
                max_depth=18,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),

        "Standalone_ExtraTrees": unscaled(
            ExtraTreesRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),

        "Standalone_HistGB": unscaled(
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=250,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=RANDOM_STATE,
            )
        ),

        "Standalone_XGBoost": unscaled(
            XGBRegressor(
                objective="reg:squarederror",
                n_estimators=400,
                learning_rate=0.03,
                max_depth=5,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=5.0,
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
            )
        ),

        "Standalone_LSTM": TorchLSTMRegressor(
            hidden_size=32,
            dropout=0.2,
            learning_rate=0.001,
            batch_size=256,
            max_epochs=100,
            patience=12,
            random_state=RANDOM_STATE,
            horizon_hours=horizon,
        ),
    }


# ============================================================
# 7. EVALUATION METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    nonzero = np.abs(y_true) > 1e-8

    mape_nonzero = (
        float(
            np.mean(
                np.abs(
                    (y_true[nonzero] - y_pred[nonzero])
                    / y_true[nonzero]
                )
            ) * 100
        )
        if nonzero.any()
        else np.nan
    )

    return {
        "n": int(len(y_true)),

        "MAE": float(
            mean_absolute_error(y_true, y_pred)
        ),

        "RMSE": float(
            np.sqrt(
                mean_squared_error(y_true, y_pred)
            )
        ),

        "R2": float(
            r2_score(y_true, y_pred)
        ),

        "MAPE_nonzero_percent": mape_nonzero,

        "n_nonzero_for_MAPE": int(
            nonzero.sum()
        ),
    }


# ============================================================
# 8. EXPANDING-WINDOW OOF RESIDUAL GENERATION
# ============================================================

def expanding_window_oof_residuals(
    baseline,
    X,
    y,
    meta,
    horizon,
    n_splits=5,
):
    """
    Generate expanding-window OOF residuals with a horizon-dependent
    temporal purge.

    For every OOF block, baseline training targets must be strictly
    earlier than the first forecast origin in that block.
    """

    n = len(X)

    if not (len(y) == len(meta) == n):
        raise ValueError(
            "X, y and meta must have identical lengths."
        )

    if not meta["origin_time"].is_monotonic_increasing:
        raise ValueError(
            "Training origins must be chronologically sorted."
        )

    if not meta["target_time"].is_monotonic_increasing:
        raise ValueError(
            "Training targets must be chronologically sorted."
        )

    first_train = max(
        2,
        int(np.floor(0.50 * n)),
    )

    boundaries = np.linspace(
        first_train,
        n,
        n_splits + 1,
        dtype=int,
    )

    oof_indices = []
    oof_predictions = []
    fold_audit = []

    for fold_number, (
        block_start,
        block_end,
    ) in enumerate(
        zip(boundaries[:-1], boundaries[1:]),
        start=1,
    ):

        if block_end <= block_start:
            continue

        first_oof_origin = pd.Timestamp(
            meta.iloc[block_start]["origin_time"]
        )

        first_oof_target = pd.Timestamp(
            meta.iloc[block_start]["target_time"]
        )

        earlier_meta = meta.iloc[:block_start]

        eligible_mask = (
            pd.to_datetime(
                earlier_meta["target_time"]
            ) < first_oof_origin
        )

        train_positions = np.flatnonzero(
            eligible_mask.to_numpy()
        )

        if len(train_positions) < 2:
            raise RuntimeError(
                f"Insufficient purged training data "
                f"in fold {fold_number}."
            )

        train_end_exclusive = int(
            train_positions[-1] + 1
        )

        if not np.array_equal(
            train_positions,
            np.arange(train_end_exclusive),
        ):
            raise RuntimeError(
                f"Non-contiguous training prefix "
                f"in fold {fold_number}."
            )

        X_fold_train = X.iloc[train_positions]
        y_fold_train = y.iloc[train_positions]

        X_fold_oof = X.iloc[
            block_start:block_end
        ]

        fold_baseline = clone(baseline)

        fold_baseline.fit(
            X_fold_train,
            y_fold_train,
        )

        block_prediction = fold_baseline.predict(
            X_fold_oof
        )

        max_train_target = pd.Timestamp(
            meta.iloc[train_positions]["target_time"].max()
        )

        causal_ok = (
            max_train_target < first_oof_origin
        )

        if not causal_ok:
            raise RuntimeError(
                f"Temporal leakage detected "
                f"in fold {fold_number}."
            )

        oof_indices.extend(
            range(block_start, block_end)
        )

        oof_predictions.extend(
            block_prediction.tolist()
        )

        fold_audit.append({

            "fold": fold_number,

            "horizon_hours": int(horizon),

            "baseline_train_start_position": 0,

            "baseline_train_end_position_exclusive":
                train_end_exclusive,

            "baseline_train_n": int(
                len(train_positions)
            ),

            "purged_start_position":
                train_end_exclusive,

            "purged_end_position_exclusive":
                int(block_start),

            "purged_n": int(
                block_start - len(train_positions)
            ),

            "oof_block_start_position":
                int(block_start),

            "oof_block_end_position_exclusive":
                int(block_end),

            "oof_n": int(
                block_end - block_start
            ),

            "max_training_target_time":
                max_train_target,

            "first_oof_origin_time":
                first_oof_origin,

            "first_oof_target_time":
                first_oof_target,

            "causal_check": bool(causal_ok),
        })

    if not oof_indices:
        raise RuntimeError(
            "No out-of-fold residuals were generated."
        )

    X_oof = X.iloc[
        oof_indices
    ].reset_index(drop=True)

    y_oof = y.iloc[
        oof_indices
    ].reset_index(drop=True)

    residual_oof = (
        y_oof.to_numpy()
        - np.asarray(oof_predictions, dtype=float)
    )

    return X_oof, residual_oof, fold_audit


# ============================================================
# 9. RUN ONE FORECAST HORIZON
# ============================================================

def run_one_horizon(df, horizon):

    X, y, meta = build_supervised_data(
        df,
        horizon,
    )

    splits = split_by_target_time(
        X,
        y,
        meta,
    )

    Xtr, ytr, mtr = splits["train"]
    Xv, yv, mv = splits["validation"]
    Xte, yte, mte = splits["test"]

    if min(len(Xtr), len(Xv), len(Xte)) == 0:
        raise RuntimeError(
            f"Empty split at horizon {horizon}; "
            "inspect data and split definitions."
        )

    print(
        f"\nHORIZON {horizon} h: "
        f"train={len(Xtr)}, "
        f"val={len(Xv)}, "
        f"test={len(Xte)}"
    )

    predictions = []
    metrics = []
    residual_audit = []

    models = model_definitions(horizon)

    # --------------------------------------------------------
    # Final empirical temporal baseline
    # --------------------------------------------------------

    baseline = models["Temporal_Ridge"]

    baseline.fit(Xtr, ytr)

    base_val_pred = baseline.predict(Xv)
    base_test_pred = baseline.predict(Xte)

    pred_frame = mte.copy()

    pred_frame["y_true"] = yte.to_numpy()

    pred_frame["Temporal_Ridge"] = base_test_pred

    metrics.append({
        "horizon_hours": horizon,
        "model": "Temporal_Ridge",
        "residual_scale": np.nan,
        **calculate_metrics(yte, base_test_pred),
    })

    # --------------------------------------------------------
    # Generate purged OOF residuals
    # --------------------------------------------------------

    X_oof, residual_oof, fold_audit = (
        expanding_window_oof_residuals(
            models["Temporal_Ridge"],
            Xtr,
            ytr,
            mtr,
            horizon=horizon,
            n_splits=5,
        )
    )

    for row in fold_audit:
        residual_audit.append({
            "horizon_hours": horizon,
            **row,
        })

    residual_audit.append({

        "horizon_hours": horizon,

        "fold": "TOTAL",

        "baseline_train_start_position": 0,

        "baseline_train_end_position_exclusive":
            len(Xtr),

        "baseline_train_n": len(Xtr),

        "purged_start_position": "",

        "purged_end_position_exclusive": "",

        "purged_n": "",

        "oof_block_start_position": "",

        "oof_block_end_position_exclusive": "",

        "oof_n": len(X_oof),

        "max_training_target_time": "",

        "first_oof_origin_time": "",

        "first_oof_target_time": "",

        "causal_check": all(
            row["causal_check"]
            for row in fold_audit
        ),
    })

    # --------------------------------------------------------
    # Standalone and hybrid ML models
    # --------------------------------------------------------

    for name, model in models.items():

        if name == "Temporal_Ridge":
            continue

        # Standalone model.
        model.fit(Xtr, ytr)

        val_pred = model.predict(Xv)
        test_pred = model.predict(Xte)

        pred_frame[name] = test_pred

        metrics.append({
            "horizon_hours": horizon,
            "model": name,
            "residual_scale": np.nan,
            **calculate_metrics(yte, test_pred),
        })

        # Residual learner trained only on OOF baseline residuals.
        residual_model = clone(model)

        residual_model.fit(
            X_oof,
            residual_oof,
        )

        residual_val_pred = residual_model.predict(Xv)

        residual_test_pred = residual_model.predict(Xte)

        # ----------------------------------------------------
        # Causally valid validation samples
        # ----------------------------------------------------

        # A validation forecast may only be used for multiplier
        # selection when its forecast origin is later than the
        # final training target timestamp.
        causal_val_mask = (
            pd.to_datetime(mv["origin_time"]) > TRAIN_END
        ).to_numpy()

        if not causal_val_mask.any():
            raise RuntimeError(
                f"No causally valid validation samples "
                f"at horizon {horizon}."
            )

        candidate_scales = [
            0.0,
            0.25,
            0.50,
            0.75,
            1.0,
        ]

        validation_rmse = {

            scale: float(
                np.sqrt(
                    mean_squared_error(
                        yv.to_numpy()[causal_val_mask],

                        (
                            base_val_pred
                            + scale * residual_val_pred
                        )[causal_val_mask],
                    )
                )
            )

            for scale in candidate_scales
        }

        selected_scale = min(
            candidate_scales,
            key=validation_rmse.get,
        )

        # Final hybrid test predictions.
        hybrid_test_pred = (
            base_test_pred
            + selected_scale * residual_test_pred
        )

        hybrid_name = name.replace(
            "Standalone_",
            "Hybrid_",
        )

        pred_frame[hybrid_name] = hybrid_test_pred

        metrics.append({
            "horizon_hours": horizon,
            "model": hybrid_name,
            "residual_scale": float(selected_scale),
            **calculate_metrics(yte, hybrid_test_pred),
        })

        print(
            f"  {name}: "
            f"selected hybrid scale={selected_scale}"
        )

    # --------------------------------------------------------
    # Persistence baseline
    # --------------------------------------------------------

    persistence = Xte["pm_lag_1"].to_numpy()

    pred_frame["Persistence"] = persistence

    metrics.append({
        "horizon_hours": horizon,
        "model": "Persistence",
        "residual_scale": np.nan,
        **calculate_metrics(yte, persistence),
    })

    # --------------------------------------------------------
    # Verify identical test target timestamps
    # --------------------------------------------------------

    assert pred_frame["target_time"].is_monotonic_increasing

    assert pred_frame["target_time"].is_unique

    # Save test predictions.
    pred_frame.to_csv(
        RESULTS_DIR / f"predictions_{horizon}h.csv",
        index=False,
    )

    # Save OOF fold audit.
    pd.DataFrame(
        residual_audit
    ).to_csv(
        RESULTS_DIR / f"residual_oof_audit_{horizon}h.csv",
        index=False,
    )

    return metrics, pred_frame


# ============================================================
# 10. FIGURES
# ============================================================

def make_figures(metrics_df):

    # RMSE by forecast horizon.
    fig, ax = plt.subplots(figsize=(9, 5))

    selected_models = [
        "Temporal_Ridge",
        "Standalone_RF",
        "Standalone_HistGB",
        "Standalone_XGBoost",
        "Standalone_LSTM",
        "Hybrid_RF",
        "Hybrid_HistGB",
        "Hybrid_XGBoost",
        "Hybrid_LSTM",
        "Persistence",
    ]

    for model in selected_models:

        sub = metrics_df[
            metrics_df["model"] == model
        ].sort_values("horizon_hours")

        if not sub.empty:

            ax.plot(
                sub["horizon_hours"],
                sub["RMSE"],
                marker="o",
                label=model,
            )

    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("RMSE (µg/m³)")

    ax.set_title(
        "Direct multi-horizon PM2.5 forecasting"
    )

    ax.set_xticks(HORIZONS)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR / "rmse_by_horizon.png",
        dpi=300,
    )

    plt.close(fig)

    # One-hour comparison.
    one = metrics_df[
        metrics_df["horizon_hours"] == 1
    ].sort_values("RMSE")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(
        one["model"],
        one["RMSE"],
    )

    ax.set_ylabel("RMSE (µg/m³)")

    ax.set_title(
        "One-hour-ahead model comparison"
    )

    ax.tick_params(
        axis="x",
        rotation=55,
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR / "one_hour_model_comparison.png",
        dpi=300,
    )

    plt.close(fig)


# ============================================================
# 11. MAIN EXECUTION
# ============================================================

def main():

    # Load and audit the original dataset.
    df = load_uci_dataset()

    audit_dataset(df)

    all_metrics = []

    # Run all direct forecast horizons.
    for horizon in HORIZONS:

        horizon_metrics, _ = run_one_horizon(
            df,
            horizon,
        )

        all_metrics.extend(horizon_metrics)

    metrics_df = pd.DataFrame(all_metrics)

    # Save complete model metrics.
    metrics_df.to_csv(
        RESULTS_DIR / "all_model_metrics.csv",
        index=False,
    )

    # Save compact RMSE comparison table.
    pivot = metrics_df.pivot_table(
        index="model",
        columns="horizon_hours",
        values="RMSE",
        aggfunc="first",
    )

    pivot.to_csv(
        RESULTS_DIR / "rmse_summary_table.csv"
    )

    # Generate figures.
    make_figures(metrics_df)

    print("\nRESULTS SAVED TO:", RESULTS_DIR)

    print(
        metrics_df.sort_values(
            ["horizon_hours", "RMSE"]
        ).to_string(index=False)
    )

    print(
        "\nIMPORTANT: inspect dataset_audit.json, "
        "residual_oof_audit files, and prediction CSVs "
        "before writing manuscript sample counts "
        "or performance claims."
    )


if __name__ == "__main__":
    main()
