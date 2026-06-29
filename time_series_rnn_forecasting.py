import csv
import os
import time
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ============================
# FINAL SETTINGS 
# ============================
CSV_FILENAME = "airline-passengers.csv"
OUTPUT_DIR = "outputs"

WINDOW_LEN = 24          # L (past observations used as input)
SHOW_PLOTS = False       # set True only when you want extra live comparison plots
SAVE_OUTPUTS = True      # save clean project outputs for GitHub/README
ANOMALY_STD_MULTIPLIER = 1.5

# Training hyperparameters (same across all models for fair comparison)
BATCH_SIZE = 16
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.0
LR = 1e-3
MAX_EPOCHS = 300
PATIENCE = 30

SEED = 42


# ============================
# Reproducibility 
# ============================
def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Best-effort deterministic behavior 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================
# 1) Load dataset from CSV
# ============================
def load_airpassengers_csv(path=CSV_FILENAME):
    months = []
    values = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        # Expected columns: Month, Passengers
        for row in reader:
            months.append(row["Month"])
            values.append(float(row["Passengers"]))
    return months, np.array(values, dtype=np.float32)


# ============================
# 2) Scaling 
# ============================
class StandardScaler:
    def __init__(self):
        self.mean_ = 0.0
        self.std_ = 1.0

    def fit(self, x: np.ndarray):
        self.mean_ = float(np.mean(x))
        self.std_ = float(np.std(x) + 1e-8)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std_ + self.mean_


# ============================
# 3) Dataset: Level vs Differenced target
#    X: last L scaled LEVELS
#    y:
#      - no diff: next LEVEL (scaled)
#      - diff: next DELTA (scaled), then reconstruct level at eval time
#
# NOTE (important clarification):
# Differencing is computed in standardized space (on scaled values).
# This is mathematically valid; evaluation is still done in original units.
# ============================
class WindowedLevelOrDelta(Dataset):
    def __init__(self, series_scaled: np.ndarray, window_len: int, use_diff: bool):
        self.s = series_scaled.astype(np.float32)
        self.L = window_len
        self.use_diff = use_diff

        X, y, last_level = [], [], []
        T = len(self.s)

        # 1-step forecasting => for each window of length L, predict the next value (or next delta)
        for i in range(T - window_len):
            x_win = self.s[i:i + window_len]         # (L,)
            next_level = self.s[i + window_len]      # scalar (scaled)
            prev_level = self.s[i + window_len - 1]  # last element of window (scaled)

            X.append(x_win[:, None])                 # (L, 1)

            if use_diff:
                # delta in scaled space
                y.append(np.array([next_level - prev_level], dtype=np.float32))
            else:
                # level in scaled space
                y.append(np.array([next_level], dtype=np.float32))

            last_level.append(prev_level)

        self.X = np.stack(X, axis=0)                              # (N, L, 1)
        self.y = np.stack(y, axis=0)                              # (N, 1)
        self.last_level = np.array(last_level, dtype=np.float32)  # (N,)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.from_numpy(self.y[idx]),
            torch.tensor(self.last_level[idx], dtype=torch.float32),
        )


# ============================
# 4) Models: LSTM / GRU / BiLSTM
# ============================
class RNNForecaster(nn.Module):
    def __init__(self, rnn_type: str, hidden_size: int, num_layers: int,
                 bidirectional: bool, dropout: float = 0.0):
        super().__init__()
        rnn_type = rnn_type.lower()

        if rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0.0
            )
        elif rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0.0
            )
        else:
            raise ValueError("rnn_type must be 'lstm' or 'gru'")

        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1)  # output: next level or next delta (both in scaled space)
        )

    def forward(self, x):
        out, _ = self.rnn(x)       # (B, L, hidden*(1 or 2))
        last = out[:, -1, :]       # (B, hidden*(1 or 2))
        return self.head(last)     # (B, 1)


# ============================
# 5) Metrics
# ============================
def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred):
    safe_true = np.where(np.abs(y_true) < 1e-8, np.nan, y_true)
    return float(np.nanmean(np.abs((y_true - y_pred) / safe_true)) * 100)


def chronological_split_points(total_len: int):
    train_end = int(0.70 * total_len)
    val_end = int(0.85 * total_len)
    return train_end, val_end


def naive_forecast(series_raw: np.ndarray, start_idx: int):
    preds = []
    for idx in range(start_idx, len(series_raw)):
        preds.append(float(series_raw[idx - 1]))
    return np.array(preds, dtype=np.float32)


def moving_average_forecast(series_raw: np.ndarray, start_idx: int, window: int = 12):
    preds = []
    for idx in range(start_idx, len(series_raw)):
        left = max(0, idx - window)
        preds.append(float(np.mean(series_raw[left:idx])))
    return np.array(preds, dtype=np.float32)


def seasonal_naive_forecast(series_raw: np.ndarray, start_idx: int, season: int = 12):
    preds = []
    for idx in range(start_idx, len(series_raw)):
        if idx - season >= 0:
            preds.append(float(series_raw[idx - season]))
        else:
            preds.append(float(series_raw[idx - 1]))
    return np.array(preds, dtype=np.float32)


def linear_trend_forecast(series_raw: np.ndarray, train_end: int, start_idx: int):
    x_train = np.arange(train_end)
    y_train = series_raw[:train_end]
    slope, intercept = np.polyfit(x_train, y_train, deg=1)
    x_test = np.arange(start_idx, len(series_raw))
    return (slope * x_test + intercept).astype(np.float32)


def run_baseline_models(series_raw: np.ndarray):
    train_end, val_end = chronological_split_points(len(series_raw))
    y_true = series_raw[val_end:]

    candidates = [
        ("Naive", naive_forecast(series_raw, val_end)),
        ("MovingAvg-12", moving_average_forecast(series_raw, val_end, window=12)),
        ("SeasonalNaive-12", seasonal_naive_forecast(series_raw, val_end, season=12)),
        ("LinearTrend", linear_trend_forecast(series_raw, train_end, val_end)),
    ]

    rows = []
    plots = []
    for name, y_pred in candidates:
        rows.append((
            name,
            np.nan,
            mae(y_true, y_pred),
            rmse(y_true, y_pred),
            mape(y_true, y_pred),
            0.0,
        ))
        plots.append((name, y_true, y_pred))
    return rows, plots


# ============================
# 6) Training with early stopping
# ============================
def train_one_model(model, train_loader, val_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    bad_epochs = 0

    for _epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb, _lastb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb, _lastb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(criterion(pred, yb).item())

        val_loss = float(np.mean(val_losses))

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val


@torch.no_grad()
def predict_levels(model, loader, device, use_diff: bool):
    """
    Returns scaled LEVEL arrays:
      y_true_level_scaled, y_pred_level_scaled  (each shape: (N,))
    """
    model.eval()
    y_true_level = []
    y_pred_level = []

    for xb, yb, last_level in loader:
        xb = xb.to(device)

        yb_np = yb.numpy().squeeze(-1)                    # true target in scaled space (level or delta)
        last_np = last_level.numpy()                      # last level in scaled space
        pred_np = model(xb).cpu().numpy().squeeze(-1)     # predicted target in scaled space

        if use_diff:
            # reconstruct scaled level: x_{t+1} = x_t + delta
            true_level = last_np + yb_np
            pred_level = last_np + pred_np
        else:
            true_level = yb_np
            pred_level = pred_np

        y_true_level.append(true_level)
        y_pred_level.append(pred_level)

    y_true_level = np.concatenate([x.reshape(-1) for x in y_true_level], axis=0)
    y_pred_level = np.concatenate([x.reshape(-1) for x in y_pred_level], axis=0)
    return y_true_level, y_pred_level


# ============================
# 7) One condition runner (use_diff False/True)
# ============================
def run_condition(series_raw: np.ndarray, device, use_diff: bool):
    # chronological split
    T = len(series_raw)
    train_end, val_end = chronological_split_points(T)

    train_raw = series_raw[:train_end]
    val_raw = series_raw[train_end:val_end]
    test_raw = series_raw[val_end:]

    # scale (fit on train only)
    scaler = StandardScaler().fit(train_raw)
    train_s = scaler.transform(train_raw)
    val_s = scaler.transform(val_raw)
    test_s = scaler.transform(test_raw)

    # context stitching so val/test can form windows
    val_context = np.concatenate([train_s[-WINDOW_LEN:], val_s])
    test_context = np.concatenate([val_context[-WINDOW_LEN:], test_s])

    train_ds = WindowedLevelOrDelta(train_s, WINDOW_LEN, use_diff=use_diff)
    val_ds = WindowedLevelOrDelta(val_context, WINDOW_LEN, use_diff=use_diff)
    test_ds = WindowedLevelOrDelta(test_context, WINDOW_LEN, use_diff=use_diff)

    # training windows are supervised samples; shuffling windows is acceptable
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_ld = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    experiments = [
        ("LSTM",   dict(rnn_type="lstm", bidirectional=False)),
        ("GRU",    dict(rnn_type="gru",  bidirectional=False)),
        ("BiLSTM", dict(rnn_type="lstm", bidirectional=True)),
    ]

    rows = []   # (name, best_val_mse, test_mae, test_rmse, test_mape, time_sec)
    plots = []  # (name, y_true_level_original, y_pred_level_original)

    for name, cfg in experiments:
        model = RNNForecaster(
            rnn_type=cfg["rnn_type"],
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            bidirectional=cfg["bidirectional"],
            dropout=DROPOUT
        ).to(device)

        start = time.time()
        best_val_mse = train_one_model(model, train_ld, val_ld, device)
        elapsed = time.time() - start

        y_true_level_s, y_pred_level_s = predict_levels(model, test_ld, device, use_diff=use_diff)

        # back to original units for evaluation and plotting
        y_true_level = scaler.inverse_transform(y_true_level_s)
        y_pred_level = scaler.inverse_transform(y_pred_level_s)

        rows.append((
            name,
            best_val_mse,
            mae(y_true_level, y_pred_level),
            rmse(y_true_level, y_pred_level),
            mape(y_true_level, y_pred_level),
            elapsed
        ))
        plots.append((name, y_true_level, y_pred_level))

    return rows, plots


def print_table(title, rows):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"{'Model':<16} | {'Best Val MSE':<12} | {'MAE':<8} | {'RMSE':<9} | {'MAPE %':<8} | {'Time (s)':<8}")
    print("-" * 70)
    for (name, val_mse, t_mae, t_rmse, t_mape, t_sec) in rows:
        val_text = "N/A" if np.isnan(val_mse) else f"{val_mse:.6f}"
        print(f"{name:<16} | {val_text:<12} | {t_mae:<8.3f} | {t_rmse:<9.3f} | {t_mape:<8.2f} | {t_sec:<8.2f}")


def build_result_records(condition: str, rows, plots):
    records = []
    for row, plot in zip(rows, plots):
        name, val_mse, t_mae, t_rmse, t_mape, t_sec = row
        _plot_name, y_true, y_pred = plot
        records.append({
            "condition": condition,
            "model": name,
            "best_val_mse": "" if np.isnan(val_mse) else round(float(val_mse), 6),
            "test_mae": round(float(t_mae), 3),
            "test_rmse": round(float(t_rmse), 3),
            "test_mape_percent": round(float(t_mape), 2),
            "training_time_sec": round(float(t_sec), 2),
            "y_true": y_true,
            "y_pred": y_pred,
        })
    return records


def save_project_outputs(months, result_records):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metrics_rows = []
    for record in result_records:
        metrics_rows.append({
            "condition": record["condition"],
            "model": record["model"],
            "best_val_mse": record["best_val_mse"],
            "test_mae": record["test_mae"],
            "test_rmse": record["test_rmse"],
            "test_mape_percent": record["test_mape_percent"],
            "training_time_sec": record["training_time_sec"],
        })

    metrics_path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)

    best = min(result_records, key=lambda item: item["test_rmse"])
    _train_end, val_end = chronological_split_points(len(months))
    test_months = months[val_end:]
    y_true = best["y_true"]
    y_pred = best["y_pred"]
    abs_error = np.abs(y_true - y_pred)
    threshold = float(np.mean(abs_error) + ANOMALY_STD_MULTIPLIER * np.std(abs_error))
    anomaly_flags = abs_error >= threshold

    forecast_rows = []
    for idx, month in enumerate(test_months):
        forecast_rows.append({
            "month": month,
            "actual": round(float(y_true[idx]), 3),
            "forecast": round(float(y_pred[idx]), 3),
            "absolute_error": round(float(abs_error[idx]), 3),
            "model": best["model"],
            "condition": best["condition"],
        })

    forecast_path = os.path.join(OUTPUT_DIR, "forecast_comparison.csv")
    with open(forecast_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(forecast_rows[0].keys()))
        writer.writeheader()
        writer.writerows(forecast_rows)

    anomaly_rows = []
    for idx, month in enumerate(test_months):
        anomaly_rows.append({
            "month": month,
            "actual": round(float(y_true[idx]), 3),
            "forecast": round(float(y_pred[idx]), 3),
            "absolute_error": round(float(abs_error[idx]), 3),
            "error_threshold": round(threshold, 3),
            "high_error_flag": bool(anomaly_flags[idx]),
        })

    anomaly_path = os.path.join(OUTPUT_DIR, "anomaly_review.csv")
    with open(anomaly_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(anomaly_rows[0].keys()))
        writer.writeheader()
        writer.writerows(anomaly_rows)

    plot_path = os.path.join(OUTPUT_DIR, "forecast_plot.png")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(test_months, y_true, marker="o", label="Actual")
    ax.plot(test_months, y_pred, marker="o", label=f"Forecast: {best['model']} ({best['condition']})")

    flagged_months = [month for month, flag in zip(test_months, anomaly_flags) if flag]
    flagged_values = [value for value, flag in zip(y_true, anomaly_flags) if flag]
    if flagged_months:
        ax.scatter(flagged_months, flagged_values, color="red", label="High forecast error", zorder=5)

    ax.set_title("Best Forecast vs Actual Values with High-Error Review")
    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly demand / passenger count")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    return {
        "best_model": best["model"],
        "best_condition": best["condition"],
        "best_rmse": best["test_rmse"],
        "anomaly_count": int(np.sum(anomaly_flags)),
        "files": [metrics_path, forecast_path, anomaly_path, plot_path],
    }


# ============================
# 8) Main
# ============================
def main():
    set_seeds(SEED)

    _months, series = load_airpassengers_csv(CSV_FILENAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nDevice: {device}")
    print(f"Loaded {len(series)} points from {CSV_FILENAME}")
    print(f"Fixed setup: L={WINDOW_LEN}, 1-step forecasting")
    print("Models: LSTM, GRU, BiLSTM")
    print("Comparison: Baselines vs No differencing vs Differencing\n")

    # Practical baselines: important because deep learning should be compared with simple forecasting rules.
    rows_base, plots_base = run_baseline_models(series)
    print_table("Baseline Forecasting Methods", rows_base)

    # A) No differencing
    rows_a, plots_a = run_condition(series, device, use_diff=False)
    print_table("Condition A: NO DIFFERENCING (predict next level)", rows_a)

    # B) Differencing
    rows_b, plots_b = run_condition(series, device, use_diff=True)
    print_table("Condition B: DIFFERENCING (predict next delta, reconstruct next level)", rows_b)

    result_records = []
    result_records.extend(build_result_records("Baseline", rows_base, plots_base))
    result_records.extend(build_result_records("RNN - no differencing", rows_a, plots_a))
    result_records.extend(build_result_records("RNN - differencing", rows_b, plots_b))

    if SAVE_OUTPUTS:
        summary = save_project_outputs(_months, result_records)
        print("\nSaved project outputs:")
        for file_path in summary["files"]:
            print(f"  - {file_path}")
        print(
            f"\nBest result: {summary['best_model']} ({summary['best_condition']}) "
            f"with RMSE={summary['best_rmse']:.3f}"
        )
        print(f"High-error periods flagged: {summary['anomaly_count']}")
        print(
            "\nInterpretation: the same workflow can be reused for passenger demand, "
            "production demand, sensor trends, or quality-issue counts."
        )

    # Plots: 3 figures total (one per model), each with 2 panels side-by-side
    if SHOW_PLOTS:
        no_diff = {name: (y_true, y_pred) for (name, y_true, y_pred) in plots_a}
        diff = {name: (y_true, y_pred) for (name, y_true, y_pred) in plots_b}

        for model_name in ["LSTM", "GRU", "BiLSTM"]:
            y_true_a, y_pred_a = no_diff[model_name]
            y_true_b, y_pred_b = diff[model_name]

            fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
            fig.suptitle(f"{model_name} — Test: No differencing vs Differencing", fontsize=12)

            axes[0].plot(y_true_a, label="Actual")
            axes[0].plot(y_pred_a, label="Predicted")
            axes[0].set_title("No differencing")
            axes[0].set_xlabel("Test window index")
            axes[0].set_ylabel("Passengers")
            axes[0].legend()

            axes[1].plot(y_true_b, label="Actual")
            axes[1].plot(y_pred_b, label="Predicted")
            axes[1].set_title("Differencing")
            axes[1].set_xlabel("Test window index")
            axes[1].legend()

            plt.tight_layout()
            plt.show()

    print("\nDone.\n")


if __name__ == "__main__":
    main()
