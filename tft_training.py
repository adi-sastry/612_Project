from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting.metrics import QuantileLoss

def train_model(df_long, tft_dataset, hyparams):
    val_dataset = TimeSeriesDataSet.from_dataset(
        tft_dataset,
        df_long,
        predict=True,
        stop_randomization=True,
    )

    # ---- Loss/output size alignment ----
    ql = QuantileLoss()

    # ---- Model ----
    tft = TemporalFusionTransformer.from_dataset(
        tft_dataset,
        learning_rate=hyparams.get("lr", 3e-4),
        hidden_size=hyparams.get("hid_size", 64),
        attention_head_size=hyparams.get("attention_head_size", 4),
        dropout=hyparams.get("dropout", 0.1),
        hidden_continuous_size=hyparams.get("hid_cont_size", 32),
        output_size=len(ql.quantiles),  
        loss=ql,
        log_interval=-1,
        reduce_on_plateau_patience=4,
    )

    # ---- Callbacks ----
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=6,
        mode="min",
        verbose=True,
    )

    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        dirpath="tft_checkpoints/",
        filename="best_checkpoint",
        save_top_k=1,
        mode="min",
    )

    # ---- Trainer ----
    trainer = Trainer(
        max_epochs=hyparams.get("max_epochs", 70),
        accelerator="cpu",
        devices=1,
        gradient_clip_val=hyparams.get("grad_clip_val", 0.2),
        limit_train_batches=hyparams.get("lim_train_batch", 1.0),
        log_every_n_steps=10,
        callbacks=[early_stop, checkpoint_cb],
    )

    # ---- Dataloaders ----
    train_dataloader = tft_dataset.to_dataloader(train=True, batch_size=32, num_workers=0)
    val_dataloader = val_dataset.to_dataloader(train=False, batch_size=128, num_workers=0)

    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    best_model_path = checkpoint_cb.best_model_path
    return tft, trainer, best_model_path