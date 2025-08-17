import optuna
import torch
from pytorch_forecasting import TimeSeriesDataSet
from tft_data_utils import preprocessing
from tft_training import train_model
from pytorch_forecasting.metrics import SMAPE


def obj(trial):
    hparams ={
        "lr": 1e-3,
        "hid_size": trial.suggest_categorical("hidden_size", [16, 32, 64]),
        "attention_head_size":1,
        "dropout": 0.2,
        "hid_cont_size":trial.suggest_categorical("hidden_continuous_size", [8, 16, 32]),
        "max_epochs": 500,
        "grad_clip_val":0.1,        
        "lim_train_batch":0.1,
        "lstm_layers":1

                }
    df_long, tft_dataset = preprocessing()
    model, trainer, path_best_model = train_model(df_long, tft_dataset, hparams)

    val_dataset = TimeSeriesDataSet.from_dataset(
            tft_dataset,
            df_long,
            predict=True,
            stop_randomization = True,
        )
    val_dataloader = val_dataset.to_dataloader(train=False, batch_size=64)
    preds = model.predict(val_dataloader)
    acts = torch.cat([y[0]for x,y in iter(val_dataloader)])

    smape = SMAPE()(preds, acts)
    return smape.item()

study = optuna.create_study(direction='minimize')
study.optimize(obj, n_trials=5)
print("Optimal params:", study.best_params)