import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pytorch_forecasting.metrics import SMAPE


# -----------------------------
# Dataset
# -----------------------------
class MultivarTimeSeriesDataset(Dataset):
    def __init__(self, df, target_cols, covar_cols, seq_len=30, pred_len=7):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_cols = target_cols
        self.covar_cols = covar_cols
        self.feature_cols = target_cols + covar_cols

        df = df.sort_values(["CityID", "time_idx"]).reset_index(drop=True)
        self.X = df[self.feature_cols].values.astype("float32")
        self.city_ids = df["CityID"].values.astype("int64")

        self.starts = self._valid_starts(self.city_ids, len(df), seq_len + pred_len)

    @staticmethod
    def _valid_starts(city_ids: np.ndarray, T: int, window: int) -> np.ndarray:
        # segment indices per city
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
        x = self.X[idx: idx + self.seq_len]  # [L, F]
        y = self.X[idx + self.seq_len: idx + self.seq_len + self.pred_len, :len(self.target_cols)]  # [H, T]
        cid = self.city_ids[idx + self.seq_len - 1]
        return torch.from_numpy(x), torch.from_numpy(y), torch.tensor(cid, dtype=torch.long)


# -----------------------------
# Data prep
# -----------------------------
def lstm_preprocessing(csv="cleaned_pollution_data.csv", city="Los Angeles", cities=None):
    df = pd.read_csv(csv)
    df["Date"] = pd.to_datetime(df["Date"])

    if cities is not None and len(cities) > 0:
        df = df[df["City"].isin(cities)].copy()
    else:
        df = df[df["City"] == city].copy()

    df = df.dropna().sort_values(["City", "Date"]).reset_index(drop=True)

    # Keep both readable and numeric city identifiers
    df["CityName"] = df["City"].astype(str)
    df["CityID"] = df["CityName"].astype("category").cat.codes
    df["time_idx"] = (df["Date"] - df["Date"].min()).dt.days

    target_cols = ["O3 Mean", "CO Mean", "SO2 Mean", "NO2 Mean"]

    keeps = ["Date", "CityName", "CityID", "time_idx"] + target_cols + [
        "Month", "DayOfWeek", "IsWeekend", "IsWedThur",
        "O3 Mean_lag1", "CO Mean_lag1", "SO2 Mean_lag1", "NO2 Mean_lag1", "Pollution_Avg",
    ]
    # guard if any expected cols are missing
    missing = [c for c in keeps if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df = df[keeps]
    df = df.loc[:, ~df.columns.duplicated()]  # safety

    return df


def train_val_split(df: pd.DataFrame, val_ratio=0.2):
    """Chronological split within each CityID, then concat."""
    parts = []
    for cid, g in df.sort_values(["CityID", "time_idx"]).groupby("CityID"):
        cut = int(len(g) * (1 - val_ratio))
        parts.append((g.iloc[:cut].copy(), g.iloc[cut:].copy()))
    train_df = pd.concat([p[0] for p in parts], ignore_index=True)
    val_df = pd.concat([p[1] for p in parts], ignore_index=True)
    return train_df, val_df


# -----------------------------
# Model
# -----------------------------
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
# Train / Eval
# -----------------------------
def lstm_train_eval(
    df,
    seq_len=30,
    pred_len=7,
    batch_size=64,
    epochs=20,
    hidden_dim=64,
    num_layers=2,
    dropout=0.2,
    lr=1e-3,
    grad_clip=1.0,
    patience=5,
    device=None,
):
    Path("lstm_outputs").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    target_cols = ["O3 Mean", "CO Mean", "SO2 Mean", "NO2 Mean"]
    covar_cols = [
        "time_idx", "CityID", "Month", "DayOfWeek", "IsWeekend", "IsWedThur",
        "O3 Mean_lag1", "CO Mean_lag1", "SO2 Mean_lag1", "NO2 Mean_lag1", "Pollution_Avg",
    ]

    train_df, val_df = train_val_split(df, val_ratio=0.2)
    train_ds = MultivarTimeSeriesDataset(train_df, target_cols, covar_cols, seq_len, pred_len)
    val_ds   = MultivarTimeSeriesDataset(val_df,   target_cols, covar_cols, seq_len, pred_len)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    input_dim = len(target_cols) + len(covar_cols)
    output_dim = len(target_cols)

    model = LSTMForecaster(input_dim, hidden_dim, output_dim, num_layers=num_layers, dropout=dropout).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    bad_epochs = 0

    loss_hist = {
        "epoch": [],
        "train_loss": [],
        "train_eval_loss": [],
        "val_loss": [],
    }

    for ep in range(1, epochs + 1):
        # ---- Train
        model.train()
        tr_losses = []
        for xb, yb, _ in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            yp = model(xb, pred_len=pred_len)
            loss = loss_fn(yp, yb)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tr_losses.append(loss.item())
        tr = float(np.mean(tr_losses)) if tr_losses else float("nan")

        # ---- Train (eval mode) to compare fairly
        model.eval()
        tr_eval_losses = []
        with torch.no_grad():
            for xb, yb, _ in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                yp = model(xb, pred_len=pred_len)
                tr_eval_losses.append(loss_fn(yp, yb).item())
        tr_eval = float(np.mean(tr_eval_losses)) if tr_eval_losses else float("nan")

        # ---- Val
        va_losses = []
        with torch.no_grad():
            for xb, yb, _ in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                yp = model(xb, pred_len=pred_len)
                va_losses.append(loss_fn(yp, yb).item())
        va = float(np.mean(va_losses)) if va_losses else float("nan")

        loss_hist["epoch"].append(ep)
        loss_hist["train_loss"].append(tr)
        loss_hist["train_eval_loss"].append(tr_eval)
        loss_hist["val_loss"].append(va)

        sch.step(va)
        print(f"Epoch {ep}/{epochs}  •  Train {tr:.4f} | Train(eval) {tr_eval:.4f} | Val {va:.4f}  (lr={opt.param_groups[0]['lr']:.2e})")

        if va + 1e-9 < best_val:
            best_val = va
            bad_epochs = 0
            torch.save(model.state_dict(), "lstm_outputs/best_lstm_model.pth")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {ep}  (best val {best_val:.4f})")
                break

    # ---- Final eval on val set
    model.load_state_dict(torch.load("lstm_outputs/best_lstm_model.pth", map_location=device))
    model.eval()

    pd.DataFrame(loss_hist).to_csv("outputs/lstm_loss_history.csv", index=False)

    preds, trues, cids = [], [], []
    with torch.no_grad():
        for xb, yb, cid in val_dl:
            xb = xb.to(device)
            yp = model(xb, pred_len=pred_len).cpu().numpy()
            preds.append(yp)
            trues.append(yb.numpy())
            cids.append(cid.numpy())

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)
    cids   = np.concatenate(cids, axis=0).astype(int)
    H = y_true.shape[1]
    T = y_true.shape[2]
    tgt_names = ["O3 Mean", "CO Mean", "SO2 Mean", "NO2 Mean"]

    # --- Aggregate metrics
    rows = []

    # Step-wise MSE across all targets
    for h in range(H):
        rows.append({
            "metric": "MSE_step", "horizon": h + 1, "target": "ALL",
            "value": mean_squared_error(y_true[:, h, :].ravel(), y_pred[:, h, :].ravel())
        })

    # Per-pollutant metrics
    for ti, tname in enumerate(tgt_names):
        yt = y_true[:, :, ti].ravel()
        yp = y_pred[:, :, ti].ravel()
        rows += [
            {"metric": "MAE_target",  "horizon": "ALL", "target": tname, "value": mean_absolute_error(yt, yp)},
            {"metric": "RMSE_target", "horizon": "ALL", "target": tname, "value": np.sqrt(mean_squared_error(yt, yp))},
            {"metric": "R2_target",   "horizon": "ALL", "target": tname, "value": r2_score(yt, yp)},
        ]

    # Aggregate RMSE
    rmse_agg = np.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel()))
    rows.append({"metric": "RMSE_agg", "horizon": "ALL", "target": "ALL", "value": rmse_agg})

    r2_agg = r2_score(y_true.reshape(-1), y_pred.reshape(-1))
    rows.append({"metric": "R2_agg", "horizon": "ALL", "target": "ALL", "value": r2_agg})

    # SMAPE (aggregate across all steps/targets)
    # Flatten to [N*H*T] for symmetric comparison
    smape_val = SMAPE()(
        torch.tensor(y_pred.reshape(-1, 1), dtype=torch.float32),
        torch.tensor(y_true.reshape(-1, 1), dtype=torch.float32),
    ).item()
    rows.append({"metric": "SMAPE_agg", "horizon": "ALL", "target": "ALL", "value": smape_val})

    # SMAPE per horizon
    for h in range(H):
        smape_step = SMAPE()(
            torch.tensor(y_pred[:, h, :].reshape(-1, 1), dtype=torch.float32),
            torch.tensor(y_true[:, h, :].reshape(-1, 1), dtype=torch.float32)
        ).item()
        rows.append({
            "metric": "SMAPE_step", "horizon": h + 1, "target": "ALL", "value": smape_step
        })

    # SMAPE per pollutant
    for ti, tname in enumerate(tgt_names):
        smape_target = SMAPE()(
            torch.tensor(y_pred[:, :, ti].reshape(-1, 1), dtype=torch.float32),
            torch.tensor(y_true[:, :, ti].reshape(-1, 1), dtype=torch.float32),
        ).item()
        rows.append({
            "metric": "SMAPE_target", "horizon": "ALL", "target": tname, "value": smape_target
        })

    Path("outputs").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv("outputs/lstm_metrics.csv", index=False)

    # --- Per-city metrics (aggregate)
    city_rows = []
    for cid in np.unique(cids):
        m = (cids == cid)
        yt_c = y_true[m].ravel()
        yp_c = y_pred[m].ravel()
        city_rows += [
            {"city": int(cid), "metric": "MAE",  "value": mean_absolute_error(yt_c, yp_c)},
            {"city": int(cid), "metric": "RMSE", "value": np.sqrt(mean_squared_error(yt_c, yp_c))},
            {"city": int(cid), "metric": "R2",   "value": r2_score(yt_c, yp_c)},
        ]
        for h in range(H):
            city_rows.append({
                "city": int(cid), "metric": "MSE_step", "horizon": h + 1,
                "value": mean_squared_error(y_true[m, h, :].ravel(), y_pred[m, h, :].ravel()),
            })
    pd.DataFrame(city_rows).to_csv("outputs/lstm_metrics_by_city.csv", index=False)

    print("Saved: outputs/lstm_metrics.csv, outputs/lstm_metrics_by_city.csv")
    print(f"[LSTM] Aggregate RMSE (val): {rmse_agg:.4f} | SMAPE: {smape_val:.4f}")
    return rmse_agg, smape_val


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", type=str, default="Los Angeles",
                    help='Comma-separated list OR "ALL" (e.g., "Los Angeles,New York" or "ALL")')
    ap.add_argument("--csv", type=str, default="cleaned_pollution_data.csv",
                    help="Path to cleaned dataset CSV")
    ap.add_argument("--seq_len", type=int, default=30)
    ap.add_argument("--pred_len", type=int, default=7)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=5)
    return ap.parse_args()


def main():
    args = parse_args()

    if args.cities.strip().upper() == "ALL":
        cities = sorted(pd.read_csv(args.csv)["City"].dropna().astype(str).unique().tolist())
        print(f"Discovered {len(cities)} cities.")
    else:
        cities = [c.strip() for c in args.cities.split(",")]

    df = lstm_preprocessing(csv=args.csv, cities=cities)
    rmse_agg, smape_val = lstm_train_eval(
        df=df,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        grad_clip=args.grad_clip,
        patience=args.patience,
    )
    print(f"[LSTM] Final — RMSE_agg: {rmse_agg:.4f} | SMAPE: {smape_val:.4f}")


if __name__ == "__main__":
    main()
