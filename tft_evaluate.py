import time
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import plotly.express as px

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import SMAPE
from tft import PollutionTFT

state_abbrev = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT',
    'Delaware': 'DE','District Of Columbia':'DC', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI',
    'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME',
    'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI',
    'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
    'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH',
    'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA',
    'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
    'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
    'Wisconsin': 'WI', 'Wyoming': 'WY'
}

def _ensure_outdir():
    Path("outputs").mkdir(exist_ok=True)


def _to_numpy(t):
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t


def evaluate_model(df_long, tft_dataset, bmp, batch_size=640):
    model = PollutionTFT.load_from_checkpoint(bmp)
    # model = TemporalFusionTransformer.load_from_checkpoint(bmp)

    # Build deterministic validation loader
    val_dataset = TimeSeriesDataSet.from_dataset(
        tft_dataset, df_long, predict=True, stop_randomization=True
    )
    val_dataloader = val_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    # ---- point predictions over full loader + total timing ----
    t0 = time.perf_counter()
    preds_tensor = model.predict(val_dataloader)  
    total_ms = (time.perf_counter() - t0) * 1000.0
    per_batch_ms = total_ms / max(len(val_dataloader), 1)

    acts_list, city_idx_list, tgt_idx_list = [], [], []
    for x, y in iter(val_dataloader):
        acts_list.append(y[0])                           
        city_idx_list.append(x["groups"][:, 0].cpu())    
        tgt_idx_list.append(x["groups"][:, 1].cpu())     

    acts_tensor = torch.cat(acts_list, dim=0)                 
    city_idx_all = torch.cat(city_idx_list, dim=0).numpy()    
    tgt_idx_all  = torch.cat(tgt_idx_list,  dim=0).numpy()   

    # Decode categories using encoders
    city_enc = val_dataset.categorical_encoders["City"]
    tgt_enc  = val_dataset.categorical_encoders["target_variable"]

    def _decode(enc, idx_np, fallback_series):
        try:
            return np.array(enc.inverse_transform(torch.tensor(idx_np)))
        except Exception:
            classes = np.array(getattr(enc, "classes_", []))
            if classes.ndim == 1 and len(classes) > 0:
                return classes[idx_np]
            cats = fallback_series.astype("category").cat.categories
            return np.array(cats)[idx_np]

    city_names_all = _decode(city_enc, city_idx_all, df_long["City"])
    tgt_names_all  = _decode(tgt_enc,  tgt_idx_all,  df_long["target_variable"])

    # Tensors with last-dim=1 for SMAPE
    preds_all = preds_tensor.unsqueeze(-1)  
    acts_all  = acts_tensor.unsqueeze(-1)  

    # ----- overall metrics -----
    smape = SMAPE()(preds_all.squeeze(-1), acts_all.squeeze(-1)).item()
    smape_per_pollutant = {}

    for tgt in np.unique(tgt_names_all):
        mask = (tgt_names_all == tgt)
        yt = acts_all[mask].squeeze(-1)  # shape: [samples, time_steps]
        yp = preds_all[mask].squeeze(-1) # shape: [samples, time_steps]
        
        # Convert directly to torch tensors (2D)
        yt_t = torch.tensor(yt, dtype=torch.float32)
        yp_t = torch.tensor(yp, dtype=torch.float32)
        
        smape_frac = SMAPE()(yp_t, yt_t).item()
        smape_per_pollutant[tgt] = smape_frac * 100

    print("SMAPE per pollutant (%):")
    for tgt, val in smape_per_pollutant.items():
        print(f"  {tgt}: {val:.2f}%")
    
    y_true = _to_numpy(acts_all)   
    y_pred = _to_numpy(preds_all)  

    rows = []
    H = y_true.shape[1]

    for tgt, smape in smape_per_pollutant.items():
            rows.append({
            "metric": "SMAPE_target",
            "horizon": "ALL",
            "target": tgt,
            "value": smape,
        })
    smape = SMAPE()(preds_all.squeeze(-1), acts_all.squeeze(-1)).item()
    rows.append({
        "metric": "SMAPE_agg",
        "horizon": "ALL",
        "target": "ALL",
        "value": smape * 100,
    })

    # step-wise MSE across all targets
    for h in range(H):
        mse_h = mean_squared_error(y_true[:, h, :].ravel(), y_pred[:, h, :].ravel())
        rows.append({"metric": "MSE_step", "horizon": h+1, "target": "ALL", "value": mse_h})

    # per-pollutant (overall)
    for tgt in np.unique(tgt_names_all):
        mask = (tgt_names_all == tgt)
        yt = y_true[mask, :, 0].ravel()
        yp = y_pred[mask, :, 0].ravel()
        rows += [
            {"metric": "MAE_target",  "horizon": "ALL", "target": tgt, "value": mean_absolute_error(yt, yp)},
            {"metric": "RMSE_target", "horizon": "ALL", "target": tgt, "value": mean_squared_error(yt, yp, squared=False)},
            {"metric": "R2_target",   "horizon": "ALL", "target": tgt, "value": r2_score(yt, yp)},
        ]

    # aggregate RMSE
    rmse_all = mean_squared_error(y_true.ravel(), y_pred.ravel(), squared=False)
    rows.append({"metric": "RMSE_agg", "horizon": "ALL", "target": "ALL", "value": rmse_all})

    _ensure_outdir()
    pd.DataFrame(rows).to_csv("outputs/metrics.csv", index=False)
    pd.DataFrame({"inference_ms_per_batch": [per_batch_ms]}).to_csv("outputs/inference_time.csv", index=False)

    # ----- metrics by City -----
    city_rows = []
    for city in np.unique(city_names_all):
        cmask = (city_names_all == city)
        yt_c = y_true[cmask, :, 0].ravel()
        yp_c = y_pred[cmask, :, 0].ravel()

        # per-city aggregate metrics
        city_rows += [
            {"city": city, "metric": "MAE",  "value": mean_absolute_error(yt_c, yp_c)},
            {"city": city, "metric": "RMSE", "value": mean_squared_error(yt_c, yp_c, squared=False)},
            {"city": city, "metric": "R2",   "value": r2_score(yt_c, yp_c)},
        ]

        for h in range(H):
            mse_h_c = mean_squared_error(y_true[cmask, h, :].ravel(), y_pred[cmask, h, :].ravel())
            city_rows.append({"city": city, "metric": "MSE_step", "horizon": h+1, "value": mse_h_c})

    pd.DataFrame(city_rows).to_csv("outputs/metrics_by_city.csv", index=False)

    # ----- Best-sample plot overall -----
    print(f"acts tensor: {acts_tensor}")
    print(f"preds tensor: {preds_tensor}")
    eps = 1e-8
    s_err = 200.0 * np.mean(
        np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + eps),
        axis=(1, 2),
    )
    best_idx = int(np.argmin(s_err))
    plt.figure(figsize=(10, 5))
    plt.plot(acts_tensor[best_idx].cpu().numpy(), label="True Target Values from Validation")
    plt.plot(preds_tensor[best_idx].cpu().numpy(), label="Predicted")
    plt.legend(); plt.title(f"Best Prediction (SMAPE: {s_err[best_idx]:.2f})")
    plt.xlabel("Time step"); plt.ylabel("Target")
    plt.savefig("outputs/best_prediction_plot.png")

    # ----- per-city best-sample plot -----
    for city in np.unique(city_names_all):
        mask = (city_names_all == city)
        if not np.any(mask):
            continue
        s_err_c = 200.0 * np.mean(
            np.abs(y_pred[mask] - y_true[mask]) / (np.abs(y_pred[mask]) + np.abs(y_true[mask]) + eps),
            axis=(1, 2),
        )
        bi = np.argmin(s_err_c)
        orig_idx = np.where(mask)[0][bi]
        plt.figure(figsize=(10, 5))
        plt.plot(acts_tensor[orig_idx].cpu().numpy(), label="True Target Values from Validation")
        plt.plot(preds_tensor[orig_idx].cpu().numpy(), label="Predicted")
        plt.legend(); plt.title(f"{city}: Best Prediction (SMAPE: {s_err_c[bi]:.2f})")
        plt.xlabel("Time step"); plt.ylabel("Target")
        safe_city = str(city).replace("/", "_").replace(" ", "_")
        plt.savefig(f"outputs/city_{safe_city}_best.png")

    
    # ----- Combined MSE and SMAPE per-step plots per pollutant -----
    mse_h_by_tgt = {}
    smape_h_by_tgt = {}

    for tgt in np.unique(tgt_names_all):
        mask = (tgt_names_all == tgt)
        yt = y_true[mask, :, 0]
        yp = y_pred[mask, :, 0]

        mse_list = []
        smape_list = []

        for h in range(H):
            yt_h = yt[:, h]
            yp_h = yp[:, h]

            mse_h = mean_squared_error(yt_h, yp_h)
            smape_h = 200.0 * np.mean(np.abs(yp_h - yt_h) / (np.abs(yp_h) + np.abs(yt_h) + 1e-8))

            mse_list.append(mse_h)
            smape_list.append(smape_h)

        mse_h_by_tgt[tgt] = mse_list
        smape_h_by_tgt[tgt] = smape_list

    # Plot MSE
    plt.figure(figsize=(10, 6))
    for tgt, mse_vals in mse_h_by_tgt.items():
        plt.plot(range(1, H + 1), mse_vals, marker="o", label=tgt)

    plt.title("MSE per Forecast Horizon (per Pollutant)")
    plt.xlabel("Forecast Horizon")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.legend(title="Pollutant")
    plt.tight_layout()
    plt.savefig("outputs/combined_mse_per_step.png")

    # Plot SMAPE
    plt.figure(figsize=(10, 6))
    for tgt, smape_vals in smape_h_by_tgt.items():
        plt.plot(range(1, H + 1), smape_vals, marker="o", label=tgt)

    plt.title("SMAPE per Forecast Horizon (per Pollutant)")
    plt.xlabel("Forecast Horizon")
    plt.ylabel("SMAPE (%)")
    plt.grid(True)
    plt.legend(title="Pollutant")
    plt.tight_layout()
    plt.savefig("outputs/combined_smape_per_step.png")


    # map
    raw_df = pd.read_csv("cleaned_pollution_data.csv")
    df_results = pd.DataFrame({
        "City": city_names_all,
        "Pollutant": tgt_names_all,
        "True": acts_all.squeeze(-1).mean(axis=1),  # mean across timesteps per sample
        "Pred": preds_all.squeeze(-1).mean(axis=1),
    })

    # Suppose raw_df has City and State info
    city_to_state = raw_df[['City', 'State']].drop_duplicates()

    # Merge city_to_state with your SMAPE results by City
    df_results_with_state = df_results.merge(city_to_state, on='City', how='left')

    eps = 1e-8
    df_results_with_state['SMAPE'] = 200 * np.abs(df_results_with_state['Pred'] - df_results_with_state['True']) / \
        (np.abs(df_results_with_state['Pred']) + np.abs(df_results_with_state['True']) + eps)
    smape_by_state = df_results_with_state.groupby(['State', 'Pollutant'])['SMAPE'].mean().reset_index()
    smape_by_state['State Code'] = smape_by_state['State'].map(state_abbrev)

    for pollutant in smape_by_state['Pollutant'].unique():
        df_plot = smape_by_state[smape_by_state['Pollutant'] == pollutant]
        fig = px.choropleth(
            df_plot,
            locations='State Code',
            locationmode='USA-states',
            color='SMAPE',
            color_continuous_scale='Reds',
            scope='usa',
            labels={'SMAPE': f'Average SMAPE for {pollutant} (%)'},
            title=f'SMAPE for {pollutant} by State'
        )
        fig.show()

    return smape