from pytorch_forecasting import TimeSeriesDataSet, NaNLabelEncoder,GroupNormalizer, TemporalFusionTransformer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
# import torch
import pytorch_lightning
import pytorch_forecasting
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# import plotly.express as px
from pytorch_forecasting.metrics import QuantileLoss
from tft import PollutionTFT

def train_model(df_long, tft_dataset):

    val_dataset = TimeSeriesDataSet.from_dataset(
        tft_dataset,
        df_long,
        predict=True,
        stop_randomization = True,
    )

    tft = TemporalFusionTransformer.from_dataset(
        tft_dataset,
        learning_rate =1e-3, #0.3, #changed to run efficiently on CPU
        hidden_size = 16,
        attention_head_size = 1,
        dropout = 0.1,
        hidden_continuous_size = 8,
        output_size = 7,
        loss=QuantileLoss(),
        log_interval=-1,
        reduce_on_plateau_patience=4,
    )

    #avoid overfitting and training time associated with overfitting
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=True,
        mode='min'
    )

    #saving best checkpoint for evaluation in 
    checkpoint_cb = ModelCheckpoint(
        monitor='val_loss',
        dirpath='tft_checkpoints/',
        filename="best_checkpoint",
        save_top_k=1,
        mode='min'
    )

    print(type(tft))
    print(isinstance(tft, pytorch_lightning.LightningModule))
    print(TemporalFusionTransformer.__module__)

    trainer = Trainer(
        max_epochs=50,
        accelerator="cpu",
        devices=1,
        gradient_clip_val=0.1,
        limit_train_batches=30, #changed 1 to run efficiently on CPU
        log_every_n_steps=10,
        callbacks=[early_stop,checkpoint_cb]

    )

    train_dataloader = tft_dataset.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_dataloader = val_dataset.to_dataloader(train=False, batch_size=64*10, num_workers=0)

    trainer.fit(
        tft,
        train_dataloaders =train_dataloader,
        val_dataloaders=val_dataloader,
    )

    best_model_path = checkpoint_cb.best_model_path
    return tft, trainer, best_model_path