from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results_lstm"

OUTPUT_PDF = ROOT / "figure_rmse_delta_onecol.pdf"
OUTPUT_PNG = ROOT / "figure_rmse_delta_onecol.png"

HORIZONS = [1, 3, 6, 12, 24]

LEARNERS = [
    "SVR",
    "RF",
    "ExtraTrees",
    "HistGB",
    "XGBoost",
    "LSTM",
]

PRETTY_NAMES = {
    "SVR": "SVR",
    "RF": "RF",
    "ExtraTrees": "Extra Trees",
    "HistGB": "HistGB",
    "XGBoost": "XGBoost",
    "LSTM": "LSTM",
}

COLORS = {
    "SVR": "#1f77b4",
    "RF": "#ff7f0e",
    "ExtraTrees": "#2ca02c",
    "HistGB": "#d62728",
    "XGBoost": "#9467bd",
    "LSTM": "#8c564b",
}

MARKERS = {
    "SVR": "o",
    "RF": "s",
    "ExtraTrees": "^",
    "HistGB": "D",
    "XGBoost": "v",
    "LSTM": "P",
}


# ============================================================
# 2. IEEE FIGURE STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],

    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,

    "axes.linewidth": 0.8,
    "lines.linewidth": 1.0,

    # Embed TrueType fonts in PDF.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "savefig.dpi": 600,
})


# ============================================================
# 3. RMSE FUNCTION
# ============================================================

def calculate_rmse(y_true, y_pred):

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0:
        raise ValueError("Cannot calculate RMSE from empty data.")

    if len(y_true) != len(y_pred):
        raise ValueError("Actual and predicted lengths do not match.")

    if not np.isfinite(y_true).all():
        raise ValueError("Non-finite values found in actual observations.")

    if not np.isfinite(y_pred).all():
        raise ValueError("Non-finite values found in predictions.")

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ============================================================
# 4. LOAD AND VALIDATE FINAL TEST PREDICTIONS
# ============================================================

MODEL_COLUMNS = (
    ["Temporal_Ridge", "Persistence"]
    + [f"Standalone_{m}" for m in LEARNERS]
    + [f"Hybrid_{m}" for m in LEARNERS]
)

REQUIRED_COLUMNS = (
    ["target_time", "y_true"] + MODEL_COLUMNS
)

all_models = MODEL_COLUMNS

rmse_results = {
    model: [] for model in all_models
}

sample_counts = {}

print("\nLoading final test prediction files...\n")

for h in HORIZONS:

    csv_path = RESULTS_DIR / f"predictions_{h}h.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing prediction file: {csv_path}\n"
            "Run run_experiments_lstm.py first."
        )

    df = pd.read_csv(csv_path)

    # Check required columns.
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in {csv_path.name}: "
            f"{missing_columns}"
        )

    if df.empty:
        raise ValueError(
            f"{csv_path.name} contains no test predictions."
        )

    # Check timestamp integrity.
    if df["target_time"].isna().any():
        raise ValueError(
            f"Missing target timestamps in {csv_path.name}."
        )

    if df["target_time"].duplicated().any():
        raise ValueError(
            f"Duplicate target timestamps in {csv_path.name}."
        )

    parsed_times = pd.to_datetime(
        df["target_time"],
        errors="coerce"
    )

    if parsed_times.isna().any():
        raise ValueError(
            f"Invalid target timestamps in {csv_path.name}."
        )

    if not parsed_times.is_monotonic_increasing:
        raise ValueError(
            f"Target timestamps are not chronological "
            f"in {csv_path.name}."
        )

    # Convert actual and prediction columns to numeric.
    for col in ["y_true"] + MODEL_COLUMNS:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Reject missing and non-finite values.
    numeric_values = df[
        ["y_true"] + MODEL_COLUMNS
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            f"Missing or non-finite numerical values "
            f"in {csv_path.name}."
        )

    # Calculate test RMSE for every model.
    y_true = df["y_true"].to_numpy()

    for model in all_models:

        y_pred = df[model].to_numpy()

        rmse_results[model].append(
            calculate_rmse(y_true, y_pred)
        )

    sample_counts[h] = len(df)

    print(
        f"{h:2d}-hour horizon: "
        f"{len(df):,} test predictions loaded."
    )


# ============================================================
# 5. CALCULATE RMSE DIFFERENCES RELATIVE TO RIDGE
# ============================================================

ridge_rmse = np.asarray(
    rmse_results["Temporal_Ridge"],
    dtype=float
)

delta_results = {}

for prefix in ["Standalone", "Hybrid"]:

    for learner in LEARNERS:

        model_name = f"{prefix}_{learner}"

        model_rmse = np.asarray(
            rmse_results[model_name],
            dtype=float
        )

        delta_results[model_name] = (
            model_rmse - ridge_rmse
        )


# ============================================================
# 6. PRINT NUMERICAL RESULTS FOR VERIFICATION
# ============================================================

print("\nTEST-SET RMSE VALUES (microgram/m3)")
print("-" * 75)

print(
    f"{'Model':<24}"
    + "".join(f"{h:>10}h" for h in HORIZONS)
)

print("-" * 75)

for model in all_models:

    values = rmse_results[model]

    print(
        f"{model:<24}"
        + "".join(f"{v:>11.3f}" for v in values)
    )

print("\nRMSE DIFFERENCES RELATIVE TO TEMPORAL RIDGE")
print("-" * 75)

for model, values in delta_results.items():

    print(
        f"{model:<24}"
        + "".join(f"{v:>11.3f}" for v in values)
    )

print("-" * 75)


# ============================================================
# 7. GENERATE FIGURE
# ============================================================

x = np.arange(len(HORIZONS))

fig, axes = plt.subplots(
    2,
    1,
    figsize=(3.5, 4.3),
    sharex=True,
    sharey=True
)

panels = [
    ("Standalone", "(a) Standalone models"),
    ("Hybrid", "(b) Hybrid models"),
]

for ax, (prefix, title) in zip(axes, panels):

    # Temporal Ridge reference.
    ax.axhline(
        0,
        color="black",
        linewidth=1.0,
        zorder=2
    )

    # Plot all six learners.
    for learner in LEARNERS:

        model_name = f"{prefix}_{learner}"

        ax.plot(
            x,
            delta_results[model_name],
            marker=MARKERS[learner],
            color=COLORS[learner],
            linewidth=1.0,
            markersize=3.5,
            label=PRETTY_NAMES[learner],
            zorder=3
        )

    ax.set_title(
        title,
        loc="left",
        fontsize=7,
        pad=2
    )

    ax.set_ylabel(
        r"$\Delta$RMSE vs. Ridge ($\mu$g/m$^3$)"
    )

    ax.grid(
        True,
        alpha=0.3,
        linewidth=0.5
    )

    ax.tick_params(
        direction="out",
        width=0.7,
        length=2.5
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


# X-axis.
axes[1].set_xticks(x)
axes[1].set_xticklabels(
    [str(h) for h in HORIZONS]
)

axes[1].set_xlabel(
    "Forecast horizon (hours)"
)


# Widen x-axis margins slightly so the left-end Ridge label has clean space.
axes[0].set_xlim(-0.45, len(HORIZONS) - 1 + 0.25)

# Label the Ridge reference line cleanly on the left end.
axes[0].text(
    -0.38,
    0.08,
    "Ridge",
    va="bottom",
    ha="left",
    fontsize=6,
    zorder=4,
    bbox=dict(
        boxstyle="square,pad=0.1",
        facecolor="white",
        edgecolor="none",
        alpha=0.85
    )
)


# Shared legend.
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    frameon=False,
    columnspacing=1.0,
    handletextpad=0.4
)


# Layout.
fig.subplots_adjust(
    left=0.17,
    right=0.98,
    top=0.97,
    bottom=0.17,
    hspace=0.18
)


# ============================================================
# 8. SAVE FIGURE
# ============================================================

fig.savefig(
    OUTPUT_PDF,
    format="pdf",
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_PNG,
    format="png",
    dpi=600,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 9. ADDITIONAL PERSISTENCE CHECK
# ============================================================

persistence_delta = (
    np.asarray(rmse_results["Persistence"])
    - ridge_rmse
)

print("\nPERSISTENCE MINUS TEMPORAL RIDGE")
print(
    dict(zip(HORIZONS, np.round(persistence_delta, 4)))
)

print("\nFigure generation completed successfully.")

print(f"PDF: {OUTPUT_PDF}")
print(f"PNG: {OUTPUT_PNG}")