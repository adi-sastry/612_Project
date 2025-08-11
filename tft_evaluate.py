import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.metrics import SMAPE
import matplotlib.pyplot as plt
from tft import PollutionTFT

def evaluate_model(df_long, tft_dataset, bmp):
    #loading best model checkpoint saved in tft_training.py
    model_from_best_check_point = PollutionTFT.load_from_checkpoint(bmp)

    # Setting up validation dataloader as done in training.py
    val_dataset = TimeSeriesDataSet.from_dataset(
        tft_dataset,
        df_long,
        predict=True,
        stop_randomization = True,
    )

    val_dataloader = val_dataset.to_dataloader(train=False, batch_size=64*10, num_workers=0)

    # Evaluate on validation 
    preds = model_from_best_check_point.predict(val_dataloader)
    true = torch.cat([y[0]for x,y in iter(val_dataloader)])
    

    # Symmetric Mean Absolute Percentage Error
    smape = SMAPE()(preds,true)

    #Plotting Predicted vs. Actual Sample

    best_loss = float("inf")
    best_batch = None
    best_pred = None
    best_act = None

    for x,y in iter(val_dataloader):
        #predict
        output = model_from_best_check_point(x)
        preds= output[0]

        #computing loss for each sample
        for i in range(preds.shape[0]):
            ipred = preds[i].detach().cpu()
            iact = y[0][i].detach().cpu()

            error = SMAPE()(ipred.unsqueeze(-1),iact.unsqueeze(-1))

            if error < best_loss:
                best_loss = error
                best_batch = x
                best_pred = ipred
                best_act = iact

    pred_to_plot = best_pred[:, 0]
    plt.figure(figsize=(10, 5))
    plt.plot(best_act, label="True Target Values from Validation")
    plt.plot(pred_to_plot, label="Predicted")
    plt.legend()
    plt.title(f"Best Prediction (SMAPE: {best_loss:.2f})")
    plt.xlabel("Time step")
    plt.ylabel("Target")
    plt.savefig("best_prediction_plot.png")
    #plt.show() 
    return smape

    

