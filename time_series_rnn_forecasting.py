import csv
import time
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# FINAL SETTINGS 
CSV_FILENAME = "airline-passengers.csv"

WINDOW_LEN = 24          # L (past observations used as input)
SHOW_PLOTS = True        # show live plots

# Training hyperparameters (same across all models for fair comparison)
BATCH_SIZE = 16
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.0
LR = 1e-3
MAX_EPOCHS = 300
PATIENCE = 30

SEED = 42


# Reproducibility 
def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Best-effort deterministic behavior 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 1) Load dataset from CSV
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


# 2) Scaling 
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


# 3) Dataset: Level vs Differenced target
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


# 4) Models: LSTM / GRU / BiLSTM
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


# 5) Metrics
def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# 6) Training with early stopping
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


# 7) One condition runner
def run_condition(series_raw: np.ndarray, device, use_diff: bool):
    # chronological split
    T = len(series_raw)
    train_end = int(0.70 * T)
    val_end = int(0.85 * T)

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

    rows = []   # (name, best_val_mse, test_mae, test_rmse, time_sec)
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
            elapsed
        ))
        plots.append((name, y_true_level, y_pred_level))

    return rows, plots


def print_table(title, rows):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"{'Model':<10} | {'Best Val MSE':<12} | {'Test MAE':<8} | {'Test RMSE':<9} | {'Time (s)':<8}")
    print("-" * 70)
    for (name, val_mse, t_mae, t_rmse, t_sec) in rows:
        print(f"{name:<10} | {val_mse:<12.6f} | {t_mae:<8.3f} | {t_rmse:<9.3f} | {t_sec:<8.2f}")


# 8) Main
def main():
    set_seeds(SEED)

    _months, series = load_airpassengers_csv(CSV_FILENAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nDevice: {device}")
    print(f"Loaded {len(series)} points from {CSV_FILENAME}")
    print(f"Fixed setup: L={WINDOW_LEN}, 1-step forecasting")
    print("Models: LSTM, GRU, BiLSTM")
    print("Comparison: No differencing vs Differencing\n")

    # A) No differencing
    rows_a, plots_a = run_condition(series, device, use_diff=False)
    print_table("Condition A: NO DIFFERENCING (predict next level)", rows_a)

    # B) Differencing
    rows_b, plots_b = run_condition(series, device, use_diff=True)
    print_table("Condition B: DIFFERENCING (predict next delta, reconstruct next level)", rows_b)

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
