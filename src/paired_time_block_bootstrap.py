"""
Reproducible paired moving-block bootstrap for the final PM2.5 predictions.

Usage:
    python paired_time_block_bootstrap.py --predictions-dir results_lstm \
        --output paired_block_bootstrap_results_recomputed.csv

Method:
- Uses the complete hourly test timeline (2014-06-01 00:00 through
  2014-12-31 23:00), not merely consecutive valid prediction rows.
- Samples contiguous time blocks of 24, 48, and 72 hours with replacement.
- Keeps only timestamps for which the paired prediction file has a valid
  target and predictions; all compared models use the same sampled rows.
- Uses percentile 95% confidence intervals for Delta RMSE = RMSE(A)-RMSE(B).
- Uses 2,000 replicates and a fixed seed of 42.
- Positive Delta RMSE means model A has higher error; negative means A is better.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = [1, 3, 6, 12, 24]
BLOCK_LENGTHS = [24, 48, 72]
N_BOOTSTRAP = 2000
SEED = 42

TEST_START = pd.Timestamp("2014-06-01 00:00:00")
TEST_END = pd.Timestamp("2014-12-31 23:00:00")
TIME_COLS = {"origin_time", "target_time", "y_true"}


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-dir", type=Path, default=Path("results_lstm"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paired_block_bootstrap_results_recomputed.csv"),
    )
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    full_times = pd.date_range(TEST_START, TEST_END, freq="h")
    n_time = len(full_times)
    time_to_position = {timestamp: i for i, timestamp in enumerate(full_times)}

    results = []

    for horizon in HORIZONS:
        path = args.predictions_dir / f"predictions_{horizon}h.csv"
        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path, parse_dates=["origin_time", "target_time"])
        if df["target_time"].duplicated().any():
            raise ValueError(f"Duplicate target timestamps in {path}")
        if not df["target_time"].is_monotonic_increasing:
            raise ValueError(f"Unsorted target timestamps in {path}")
        if not (
            (df["target_time"] - df["origin_time"])
            .eq(pd.Timedelta(hours=horizon))
            .all()
        ):
            raise ValueError(f"Incorrect forecast offsets in {path}")

        model_names = [c for c in df.columns if c not in TIME_COLS]
        required = [
            "Standalone_SVR", "Hybrid_SVR",
            "Standalone_RF", "Hybrid_RF",
            "Standalone_ExtraTrees", "Hybrid_ExtraTrees",
            "Standalone_HistGB", "Hybrid_HistGB",
            "Standalone_XGBoost", "Hybrid_XGBoost",
            "Standalone_LSTM", "Hybrid_LSTM",
        ]
        missing = sorted(set(required) - set(model_names))
        if missing:
            raise ValueError(f"{path} is missing model columns: {missing}")

        target_times = pd.to_datetime(df["target_time"])
        if not target_times.isin(full_times).all():
            raise ValueError(f"Targets outside the declared test timeline in {path}")

        valid_positions = np.array(
            [time_to_position[t] for t in target_times], dtype=int
        )
        row_by_hour = np.full(n_time, -1, dtype=int)
        row_by_hour[valid_positions] = np.arange(len(df))

        y = df["y_true"].to_numpy(dtype=float)
        squared_errors = {
            name: (y - df[name].to_numpy(dtype=float)) ** 2
            for name in model_names
        }
        point_rmse = {
            name: float(np.sqrt(squared_errors[name].mean()))
            for name in model_names
        }

        ranked = sorted(model_names, key=lambda name: point_rmse[name])
        comparisons = [
            (f"Hybrid_{base}", f"Standalone_{base}", "Hybrid_vs_Standalone")
            for base in ["SVR", "RF", "ExtraTrees", "HistGB", "XGBoost", "LSTM"]
        ]
        comparisons.append((ranked[0], ranked[1], "Best_vs_Second_Best"))

        for block_hours in BLOCK_LENGTHS:
            n_blocks = int(np.ceil(n_time / block_hours))
            deltas = {comparison: [] for comparison in comparisons}
            sampled_valid_counts = []

            for _ in range(N_BOOTSTRAP):
                starts = rng.integers(
                    0, n_time - block_hours + 1, size=n_blocks
                )
                sampled_hours = np.concatenate(
                    [
                        np.arange(start, start + block_hours)
                        for start in starts
                    ]
                )[:n_time]

                sampled_rows = row_by_hour[sampled_hours]
                sampled_rows = sampled_rows[sampled_rows >= 0]
                if len(sampled_rows) == 0:
                    continue

                sampled_valid_counts.append(len(sampled_rows))

                for model_a, model_b, comparison_type in comparisons:
                    rmse_a = np.sqrt(squared_errors[model_a][sampled_rows].mean())
                    rmse_b = np.sqrt(squared_errors[model_b][sampled_rows].mean())
                    deltas[(model_a, model_b, comparison_type)].append(
                        float(rmse_a - rmse_b)
                    )

            for (model_a, model_b, comparison_type), values in deltas.items():
                values = np.asarray(values, dtype=float)
                lower, upper = np.quantile(values, [0.025, 0.975])
                point_delta = point_rmse[model_a] - point_rmse[model_b]

                results.append(
                    {
                        "horizon_hours": horizon,
                        "block_hours": block_hours,
                        "comparison_type": comparison_type,
                        "model_A": model_a,
                        "model_B": model_b,
                        "RMSE_A": point_rmse[model_a],
                        "RMSE_B": point_rmse[model_b],
                        "point_delta_RMSE": point_delta,
                        "bootstrap_mean_delta": float(values.mean()),
                        "CI_95_lower": float(lower),
                        "CI_95_upper": float(upper),
                        "CI_excludes_zero": bool(lower > 0 or upper < 0),
                        "n_test": len(df),
                        "mean_bootstrap_n": float(np.mean(sampled_valid_counts)),
                        "n_bootstrap": len(values),
                        "bootstrap_seed": SEED,
                        "resampling": "moving blocks on complete hourly test grid",
                    }
                )

    output = pd.DataFrame(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Saved {len(output)} comparisons to {args.output}")


if __name__ == "__main__":
    main()
