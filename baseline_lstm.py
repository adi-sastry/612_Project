import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import argparse
import torch.optim as optim
import os

#Preparing multivariate time series data
class MultivarTimeSeriesDataset(Dataset):
    def __init__(self,df, target_cols, covar_cols,seq_len=30, pred_len = 7):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_cols=target_cols
        self.covar_cols = covar_cols
        self.feature_cols = target_cols + covar_cols

        df = df.sort_values("time_idx").reset_index(drop=True)
        self.data = df[self.feature_cols].values.astype("float32")

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len
    
    def __getitem__(self,idx):
        x = self.data[idx: idx+self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len, :len(self.target_cols)]
        return torch.tensor(x), torch.tensor(y)
    
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.decoder_hidden = nn.Linear(output_dim, hidden_dim)

    #forward pass, takes input tensor x and time steps. x goes through encoder to get hidden time steps then initialize the decoder inputs as the last hidden state of output
    # from encoder
    def forward (self,x, pred_len=7):
        batch_size = x.size(0)
        encoder_output,_ = self.encoder(x)

        outputs = []
        decoder_input = encoder_output[:, -1 , :]

        #autoregressive loop - pass decoder to linear layer ang get output for given timestep and append predictio to output list
        #update decoder by passing the prediction through decoder hidden and Relu. Stack predicted outputs
        for _ in range(pred_len):
            out = self.decoder(decoder_input)
            outputs.append(out)

            decoder_input = torch.relu(self.decoder_hidden(out))

        outputs = torch.stack(outputs, dim=1)
        return outputs

#allows us to specify cities of interest and the csv path
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", type=str, default="Los Angeles",
                    help='Comma-separated list OR "ALL" (e.g., "Los Angeles,New York" or "ALL")')
    ap.add_argument("--csv", type=str, default="cleaned_pollution_data.csv",
                    help="Path to cleaned dataset CSV")
    return ap.parse_args()


def lstm_preprocessing(csv="cleaned_pollution_data.csv", city="Los Angeles", cities=None):
    pollution_cleaned = pd.read_csv(csv)
    pollution_cleaned["Date"] = pd.to_datetime(pollution_cleaned["Date"])

    if cities is not None and len(cities) > 0:
        df = pollution_cleaned[pollution_cleaned["City"].isin(cities)].copy()
    else:
        df = pollution_cleaned[pollution_cleaned["City"] == city].copy()

    df = df.dropna()
    df['City'] = df['City'].astype('category').cat.codes

    df["time_idx"] = (df["Date"] - df["Date"].min()).dt.days

    target_columns = ["O3 Mean", "CO Mean", "SO2 Mean"]
    covars=["time_idx","City","Month","DayOfWeek","IsWeekend","IsWedThur",
                 "O3 Mean_lag1","CO Mean_lag1","SO2 Mean_lag1","NO2 Mean_lag1","Pollution_Avg"]
    
    keeps = ["Date"] + target_columns + covars
    df = df[keeps]

    return df

#splitting data chronologically 
def train_val_split (df, val_ratio = 0.2):
    df = df.sort_values("time_idx").reset_index(drop=True)
    cut = int(len(df)*(1-val_ratio))
    train_df = df.iloc[:cut]
    val_df = df.iloc[cut:]
    return train_df, val_df

#Training loop - training and validation datasets and data loaders
def lstm_train(df,seq_len = 30, pred_len = 7, batch_size = 64, epochs =10):
    os.makedirs("lstm_outputs", exist_ok=True) 
    train_df, val_df =train_val_split(df)

    target_cols = ["O3 Mean", "CO Mean", "SO2 Mean"]
    covar_cols = ["time_idx","City","Month","DayOfWeek","IsWeekend","IsWedThur",
                 "O3 Mean_lag1","CO Mean_lag1","SO2 Mean_lag1","NO2 Mean_lag1","Pollution_Avg"]
    
    #Datasets
    train_ds = MultivarTimeSeriesDataset(train_df, target_cols, covar_cols,seq_len,pred_len)
    val_ds = MultivarTimeSeriesDataset(val_df, target_cols, covar_cols,seq_len,pred_len)    

    #data loaders
    train_dl = DataLoader(train_ds, batch_size=batch_size,shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=batch_size,shuffle=True, num_workers=4)

    input_dim = len(target_cols) +len(covar_cols)
    output_dim = len(target_cols)
    #model = LSTMForecaster(input_dim, hidden_dim=64, output_dim=output_dim)
    
    #using the following for quick testing
    model = LSTMForecaster(input_dim, hidden_dim=32, output_dim=output_dim, num_layers=2)

    optimizer =optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.MSELoss()

    best_val_loss = float("inf")

    for e in range(epochs):
        model.train()
        train_losses = []
        for xbatch, ybatch in train_dl:
            optimizer.zero_grad()
            y_pred = model(xbatch, pred_len=pred_len)
            loss = loss_func(y_pred, ybatch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        model.eval()
        val_losses =[]
        with torch.no_grad():
            for xbatch, ybatch in val_dl:
                y_pred = model(xbatch, pred_len=pred_len)
                loss = loss_func(y_pred, ybatch)
                val_losses.append(loss.item())
        print(f"Epoch {e+1}/{epochs} - Train Loss: {np.mean(train_losses):.4f}, Validation Loss:{np.mean(val_losses):.4f}")

        #savubg the best model 
        avg_val_loss = np.mean(val_losses)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "lstm_outputs/best_lstm_model.pth")

                

def main():
    args = parse_args()

    if args.cities.strip().upper() == "ALL":
        cities = sorted(pd.read_csv(args.csv)["City"].dropna().astype(str).unique().tolist())
        print(f"Discovered {len(cities)} cities.")
    else:
        cities = [c.strip() for c in args.cities.split(",")]
    
    df_lstm = lstm_preprocessing(csv=args.csv, cities=cities)
    #lstm_train(df = df_lstm, seq_len = 30, pred_len = 7, batch_size = 64, epochs =10)

    #using the following for quick testing
    lstm_train(df = df_lstm, seq_len = 10, pred_len = 7, batch_size = 16, epochs =1)


if __name__ == "__main__":
    main()