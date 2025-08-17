import os
import pandas as pd
from pathlib import Path
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from sklearn.preprocessing import StandardScaler

def preprocessing(csv="cleaned_pollution_data.csv", city="Los Angeles", cities=None, max_prediction_length=14, max_encoder_length=100):
    pollution_cleaned = pd.read_csv(csv)
    pollution_cleaned["Date"] = pd.to_datetime(pollution_cleaned["Date"])

    if cities is not None and len(cities) > 0:
        df_tft = pollution_cleaned[pollution_cleaned["City"].isin(cities)].copy()
    else:
        df_tft = pollution_cleaned[pollution_cleaned["City"] == city].copy()

    df_tft = df_tft.dropna()
    
    df_tft["City"] = df_tft["City"].astype(str)
    df_tft["time_idx"] = (df_tft["Date"] - df_tft["Date"].min()).dt.days
    
    Path("outputs").mkdir(exist_ok=True)
    (df_tft.groupby("City")["Date"].count()
          .rename("num_rows").reset_index()
          .to_csv("outputs/sample_counts.csv", index=False))

    target_columns = ["O3 Mean", "CO Mean", "SO2 Mean"]
    df_long = df_tft.melt(
        id_vars=["time_idx","City","Month","DayOfWeek","IsWeekend","IsWedThur",
                 "O3 Mean_lag1","CO Mean_lag1","SO2 Mean_lag1","NO2 Mean_lag1","Pollution_Avg"],
        value_vars=target_columns,
        var_name="target_variable",
        value_name="target"
    ).dropna(subset=["target"])

    training_cutoff = df_tft["time_idx"].max() - max_prediction_length

    tft_dataset = TimeSeriesDataSet(
        df_long[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="target",
        group_ids=["City", "target_variable"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["City", "target_variable"],
        time_varying_known_reals=["time_idx","Month","DayOfWeek","IsWeekend","IsWedThur"],
        time_varying_unknown_reals=["target","Pollution_Avg","O3 Mean_lag1","CO Mean_lag1","SO2 Mean_lag1","NO2 Mean_lag1"],
        target_normalizer=GroupNormalizer(groups=["City","target_variable"], transformation=None),
        allow_missing_timesteps=True
    )

    return df_long, tft_dataset