# 612_Project
DATA/MSML 612 Project - Air Pollution Forecasting Using Transformers

1. Set up the environment
    From the project root, create and activate the virtual environment, then install dependencies:

        bash setup.sh
        source tft_venv/bin/activate

2. Run the model
    (a) To train and evaluate on all cities:

        python tft_main.py --cities "ALL" --csv cleaned_pollution_data.csv

    (b) Run for specific cities:
        
        python tft_main.py --cities "Los Angeles,New York,Chicago" --csv cleaned_pollution_data.

3. View results

    Metrics and plots will be saved in the outputs/ directory.

    Example files:

        metrics.csv – aggregated performance metrics
        metrics_by_city.csv – metrics broken down by city
        best_prediction_plot.png – visualization of the best prediction
        city_<cityname>_best.png – per-city prediction plots
        