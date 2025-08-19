# 612_Project
DATA/MSML 612 Project - Air Pollution Forecasting Using Transformers

1. Set up the environment
    From the project root, create and activate the virtual environment, then install dependencies:

        bash setup.sh
        source tft_venv/bin/activate

2. Run the models:
    (a) Temporal Fusion Transformer
        (i)     To train and evaluate on all cities:

            python tft_main.py --cities "ALL" --csv cleaned_pollution_data.csv

        (ii)    Run for specific cities:
        
            python tft_main.py --cities "Los Angeles,New York,Chicago" --csv cleaned_pollution_data.

        (iii)   View results

                Metrics and plots will be saved in the outputs/ directory.

                Example files:

                    metrics.csv – aggregated performance metrics
                    metrics_by_city.csv – metrics broken down by city
                    best_prediction_plot.png – visualization of the best prediction
                    city_<cityname>_best.png – per-city prediction plots
                    
    (b) Baseline LSTM

        (i)     Train and evaluate on all cities:

                    python baseline_lstm.py --cities "ALL" --csv cleaned_pollution_data.csv --epochs 10

        (ii)    Run for specific cities:

                    python baseline_lstm.py --cities "Los Angeles,New York,Chicago" --csv cleaned_pollution_data.csv --epochs 10

        (iii)   View results

                LSTM Outputs

                    lstm_outputs/best_lstm_model.pth – saved best model checkpoint
                    outputs/lstm_metrics.csv – aggregated performance metrics
                    outputs/lstm_metrics_by_city.csv – per-city metrics

3. Generate LSTM plots:

        python lstm_make_plots.py --csv cleaned_pollution_data.csv --cities "Los Angeles,New York,Chicago"

        This produces:

            lstm_horizon_mse.png – horizon-wise error
            lstm_pollutant_{mae,rmse,r2}.png – per-pollutant bar plots
            lstm_cities_{top5,bottom5}_rmse.png – best/worst city RMSE
            lstm_best_prediction_from_model.png – true vs. predicted sequence
            lstm_residual_hist_from_model.png – validation residual distribution