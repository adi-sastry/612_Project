import argparse
import pandas as pd
from tft_data_utils import preprocessing
from tft_training import train_model
from tft_evaluate import evaluate_model
from tft_hyperparameters import tft_hyparams

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", type=str, default="Los Angeles",
                    help='Comma-separated list OR "ALL" (e.g., "Los Angeles,New York" or "ALL")')
    ap.add_argument("--csv", type=str, default="cleaned_pollution_data.csv",
                    help="Path to cleaned dataset CSV")
    return ap.parse_args()

def main():
    args = parse_args()

    if args.cities.strip().upper() == "ALL":
        cities = sorted(pd.read_csv(args.csv)["City"].dropna().astype(str).unique().tolist())
        print(f"Discovered {len(cities)} cities.")
    else:
        cities = [c.strip() for c in args.cities.split(",")]

    df_long, tft_dataset = preprocessing(csv=args.csv, cities=cities)
    model, trainer, path_best_model = train_model(df_long, tft_dataset, tft_hyparams)

    smape = evaluate_model(df_long, tft_dataset, path_best_model)
    print(f"Validation SMAPE: {smape:.4f}")
    print("Saved: outputs/sample_counts.csv, outputs/metrics.csv, outputs/metrics_by_city.csv,",
          "outputs/inference_time.csv, outputs/best_prediction_plot.png, outputs/city_*_best.png")

if __name__ == "__main__":
    main()