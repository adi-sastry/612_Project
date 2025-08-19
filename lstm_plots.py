import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

TARGET_COLS = ["O3 Mean", "CO Mean", "SO2 Mean"]
COVAR_COLS = [
    "time_idx", "CityID", "Month", "DayOfWeek", "IsWeekend", "IsWedThur",
    "O3 Mean_lag1", "CO Mean_lag1", "SO2 Mean_lag1", "NO2 Mean_lag1", "Pollution_Avg",
]

class MultivarTimeSeriesDataset(Dataset):
    def __init__(self, df, seq_len=30, pred_len=7):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_cols = TARGET_COLS + COVAR_COLS
        df = df.sort_values(["CityID", "time_idx"]).reset_index(drop=True)
        self.X = df[self.feature_cols].values.astype("float32")
        self.city_ids = df["CityID"].values.astype("int64")
        self.starts = self._valid_starts(self.city_ids, len(df), seq_len + pred_len)

    @staticmethod
    def _valid_starts(city_ids, T, window):
        cut = np.where(np.diff(city_ids) != 0)[0] + 1
        segs = np.concatenate([[0], cut, [T]])
        starts = []
        for s, e in zip(segs[:-1], segs[1:]):
            if e - s >= window:
                starts.extend(range(s, e - window + 1))
        return np.array(starts, dtype=np.int64)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        idx = self.starts[i]
        x = self.X[idx: idx + self.seq_len]
        y = self.X[idx + self.seq_len: idx + self.seq_len + self.pred_len, :len(TARGET_COLS)]
        cid = self.city_ids[idx + self.seq_len - 1]
        return torch.from_numpy(x), torch.from_numpy(y), torch.tensor(cid, dtype=torch.long)

def lstm_preprocessing(csv, cities=None, city="Los Angeles"):
    df = pd.read_csv(csv)
    df["Date"] = pd.to_datetime(df["Date"])
    if cities:
        df = df[df["City"].isin(cities)].copy()
    else:
        df = df[df["City"] == city].copy()
    df = df.dropna().sort_values(["City", "Date"]).reset_index(drop=True)
    df["CityName"] = df["City"].astype(str)
    df["CityID"] = df["CityName"].astype("category").cat.codes
    df["time_idx"] = (df["Date"] - df["Date"].min()).dt.days
    keeps = ["Date", "CityName", "CityID", "time_idx"] + TARGET_COLS + [
        "Month","DayOfWeek","IsWeekend","IsWedThur",
        "O3 Mean_lag1","CO Mean_lag1","SO2 Mean_lag1","NO2 Mean_lag1","Pollution_Avg",
    ]
    missing = [c for c in keeps if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    df = df[keeps]
    df = df.loc[:, ~df.columns.duplicated()]
    return df

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.decoder_hidden = nn.Linear(output_dim, hidden_dim)

    def forward(self, x, pred_len=7):
        enc_out, _ = self.encoder(x)
        dec_in = enc_out[:, -1, :]
        outs = []
        for _ in range(pred_len):
            y_t = self.decoder(dec_in)
            outs.append(y_t)
            dec_in = torch.relu(self.decoder_hidden(y_t))
        return torch.stack(outs, dim=1)


# -----------------------------
# Plot helpers
# -----------------------------
def plot_horizon_mse(metrics_csv, out_png="outputs/lstm_horizon_mse.png"):
    m = pd.read_csv(metrics_csv)
    step = m[m["metric"] == "MSE_step"].copy()
    if step.empty:
        print("No MSE_step in metrics; skipping horizon plot.")
        return
    step = step.sort_values("horizon")
    plt.figure(figsize=(8, 4.5))
    plt.plot(step["horizon"].values, step["value"].values, marker="o")
    plt.xlabel("Forecast Horizon (days)")
    plt.ylabel("MSE")
    plt.title("LSTM Horizon-wise MSE")
    plt.tight_layout()
    Path("outputs").mkdir(exist_ok=True)
    plt.savefig(out_png)
    print(f"Saved: {out_png}")

def plot_per_pollutant_bars(metrics_csv, out_prefix="outputs/lstm_pollutant_"):
    m = pd.read_csv(metrics_csv)
    subset = m[m["metric"].isin(["MAE_target", "RMSE_target", "R2_target"])].copy()
    if subset.empty:
        print("No per-pollutant metrics found; skipping bar plots.")
        return
    for metric in ["MAE_target", "RMSE_target", "R2_target"]:
        sub = subset[subset["metric"] == metric]
        if sub.empty:
            continue
        pivot = sub.pivot(index="target", columns="metric", values="value")
        vals = pivot[metric].reindex(["O3 Mean","CO Mean","SO2 Mean"]).values
        labels = ["O3 Mean","CO Mean","SO2 Mean"]
        plt.figure(figsize=(7, 4))
        plt.bar(np.arange(len(vals)), vals)
        plt.xticks(np.arange(len(vals)), labels, rotation=0)
        plt.ylabel(metric.replace("_target",""))
        plt.title(f"LSTM Per-Target {metric.replace('_target','')}")
        plt.tight_layout()
        out_png = f"{out_prefix}{metric.replace('_target','').lower()}.png"
        Path("outputs").mkdir(exist_ok=True)
        plt.savefig(out_png)
        print(f"Saved: {out_png}")

def plot_cities_top_bottom(metrics_by_city_csv, out_prefix="outputs/lstm_cities_"):
    try:
        c = pd.read_csv(metrics_by_city_csv)
    except FileNotFoundError:
        print("metrics_by_city not found; skipping top/bottom city plots.")
        return
    rmse = c[c["metric"] == "RMSE"][["city","value"]].copy()
    if rmse.empty:
        print("No per-city RMSE found; skipping.")
        return
    rmse = rmse.sort_values("value")
    top = rmse.head(5)
    bot = rmse.tail(5)

    # Top-5 (lowest RMSE)
    plt.figure(figsize=(8, 4))
    plt.bar(top["city"].astype(str).values, top["value"].values)
    plt.xlabel("CityID")
    plt.ylabel("RMSE")
    plt.title("LSTM Top-5 Cities (Lowest RMSE)")
    plt.tight_layout()
    out1 = f"{out_prefix}top5_rmse.png"
    Path("outputs").mkdir(exist_ok=True)
    plt.savefig(out1)
    print(f"Saved: {out1}")

    # Bottom-5 (highest RMSE)
    plt.figure(figsize=(8, 4))
    plt.bar(bot["city"].astype(str).values, bot["value"].values)
    plt.xlabel("CityID")
    plt.ylabel("RMSE")
    plt.title("LSTM Bottom-5 Cities (Highest RMSE)")
    plt.tight_layout()
    out2 = f"{out_prefix}bottom5_rmse.png"
    plt.savefig(out2)
    print(f"Saved: {out2}")

def plot_best_and_residuals(
    csv, model_path, seq_len=30, pred_len=7, batch_size=64, cities=None, device=None
):
    # Rebuild val loader to get predictions and residuals
    df = lstm_preprocessing(csv=csv, cities=cities)
    # split inside each city: 80/20
    parts = []
    for cid, g in df.sort_values(["CityID","time_idx"]).groupby("CityID"):
        cut = int(len(g) * 0.8)
        parts.append((g.iloc[:cut].copy(), g.iloc[cut:].copy()))
    val_df = pd.concat([p[1] for p in parts], ignore_index=True)

    val_ds = MultivarTimeSeriesDataset(val_df, seq_len=seq_len, pred_len=pred_len)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    input_dim = len(TARGET_COLS) + len(COVAR_COLS)
    output_dim = len(TARGET_COLS)
    model = LSTMForecaster(input_dim=input_dim, hidden_dim=64, output_dim=output_dim, num_layers=2, dropout=0.2)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for xb, yb, _ in val_dl:
            xb = xb.to(device)
            yp = model(xb, pred_len=pred_len).cpu().numpy()
            preds.append(yp)
            trues.append(yb.numpy())
    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)

    # Best sample by SMAPE-like score
    eps = 1e-8
    s_err = 200.0 * np.mean(
        np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + eps),
        axis=(1, 2),
    )
    best_idx = int(np.argmin(s_err))

    # Plot best sample
    plt.figure(figsize=(10, 5))
    plt.plot(y_true[best_idx], label="True")
    plt.plot(y_pred[best_idx], label="Predicted")
    plt.legend()
    plt.title(f"LSTM Best Prediction (approx. SMAPE: {s_err[best_idx]:.2f})")
    plt.xlabel("Forecast step"); plt.ylabel("Target value")
    plt.tight_layout()
    Path("outputs").mkdir(exist_ok=True)
    plt.savefig("outputs/lstm_best_prediction_from_model.png")
    print("Saved: outputs/lstm_best_prediction_from_model.png")

    # Residual histogram (aggregate)
    res = (y_true - y_pred).ravel()
    plt.figure(figsize=(8, 4.5))
    plt.hist(res, bins=50)
    plt.title("LSTM Residual Histogram (Validation)")
    plt.xlabel("Residual"); plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("outputs/lstm_residual_hist_from_model.png")
    print("Saved: outputs/lstm_residual_hist_from_model.png")


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="cleaned_pollution_data.csv")
    ap.add_argument("--cities", type=str, default="", help='Comma-separated list OR empty for default "Los Angeles"')
    ap.add_argument("--seq_len", type=int, default=30)
    ap.add_argument("--pred_len", type=int, default=7)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--metrics_csv", type=str, default="outputs/lstm_metrics.csv")
    ap.add_argument("--metrics_by_city_csv", type=str, default="outputs/lstm_metrics_by_city.csv")
    ap.add_argument("--model_path", type=str, default="lstm_outputs/best_lstm_model.pth")
    return ap.parse_args()


def main():
    args = parse_args()
    Path("outputs").mkdir(exist_ok=True)

    # 1) Plots from saved CSV metrics
    plot_horizon_mse(args.metrics_csv)
    plot_per_pollutant_bars(args.metrics_csv)
    plot_cities_top_bottom(args.metrics_by_city_csv)

    # 2) Plots from running the saved model once on the val set
    if Path(args.model_path).exists():
        cities = None
        if args.cities.strip():
            cities = [c.strip() for c in args.cities.split(",")]
        plot_best_and_residuals(
            csv=args.csv,
            model_path=args.model_path,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            batch_size=args.batch_size,
            cities=cities,
        )
    else:
        print(f"Model checkpoint not found at {args.model_path} — skipping model-based plots.")


if __name__ == "__main__":
    main()
