import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

def preprocessing(csv="cleaned_pollution_data.csv", city = "Los Angeles", max_prediction_length = 7, max_encoder_length=30):
    pollution_cleaned = pd.read_csv(csv)
    pollution_cleaned["Date"] = pd.to_datetime(pollution_cleaned["Date"])
    
    df_tft = pollution_cleaned[pollution_cleaned['City'] == "Los Angeles"].copy()
    df_tft = df_tft.dropna()
    
    df_tft["City"] = df_tft["City"].astype(str)
    df_tft["time_idx"] = (df_tft["Date"] - df_tft["Date"].min()).dt.days
    
    target_columns = ["O3 Mean", "CO Mean", "SO2 Mean"]
    df_long = df_tft.melt(
        id_vars=["time_idx", "City", "Month", "DayOfWeek", "IsWeekend", "IsWedThur",
             "O3 Mean_lag1", "CO Mean_lag1", "SO2 Mean_lag1", "NO2 Mean_lag1", "Pollution_Avg"],
        value_vars=target_columns,
        var_name="target_variable",
        value_name="target"
        )
    
    df_long = df_long.dropna(subset=["target"])
    training_cutoff = df_tft["time_idx"].max() - max_prediction_length

    tft_dataset = TimeSeriesDataSet(
        df_long[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="target",
        group_ids=["City", "target_variable"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["City", "target_variable"],
        time_varying_known_reals=["time_idx", "Month", "DayOfWeek", "IsWeekend", "IsWedThur"],
        time_varying_unknown_reals=["target", "Pollution_Avg", "O3 Mean_lag1", "CO Mean_lag1", "SO2 Mean_lag1", "NO2 Mean_lag1"],
        target_normalizer=GroupNormalizer(groups=["City", "target_variable"], transformation=None),
        allow_missing_timesteps=True
    )

    return df_long, tft_dataset