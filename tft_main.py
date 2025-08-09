from tft_data_utils import preprocessing
from tft_training import train_model
from tft_evaluate import evaluate_model
from tft_hyperparameters import tft_hyparams

def main():
    df_long, tft_dataset = preprocessing()
    model, trainer, path_best_model = train_model(df_long, tft_dataset, tft_hyparams)
    smape = evaluate_model(df_long,tft_dataset,path_best_model)
    print(f"Symmetric Mean Absolute Percentage Error (SMAPE) on validation set: {smape}")

if __name__ == "__main__":
    main()