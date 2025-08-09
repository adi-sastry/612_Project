import optuna
import torch
from pytorch_forecasting import TimeSeriesDataSet
from tft_data_utils import preprocessing
from tft_training import train_model
from pytorch_forecasting.metrics import SMAPE


def obj(trial):
    hparams ={
        "lr": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "hid_size": trial.suggest_categorical("hidden_size", [16, 32, 64]),
        "attention_head_size":trial.suggest_int("attention_head_size", 1, 4),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "hid_cont_size":trial.suggest_categorical("hidden_continuous_size", [8, 16, 32]),
        "max_epochs": 10,
        "grad_clip_val":trial.suggest_float("gradient_clip_val", 0.01, 1.0),
        "lim_train_batch":0.1

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