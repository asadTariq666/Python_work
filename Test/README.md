#  Revenue Modeling

This repository uses Python 3.8.

## Data Dictionary 

| Feature      | Description |
| ----------- | ----------- |
| Cancellation Rate            | The % of Policies cancelled within a 12 month period       |
| Cancellation Revenue         | The Amount Cancellation revenue within a 12 month period        |
| New Business Revenue         | The Amount of New Business Revenue in a 12 month period        |
| PY Total Revenue             | The Total Revenue in the same month  last year        |
| Rate Commission              | The difference of Total Revenue less Cancellation Revenue and New Business Revenue        |
| Rate Commission Percentage   | (Rate Commission-PY Total Revenue)/Rate Commission        |
| Renewal Rate                 | Percentage of Total # of policies renewed divided by the Total # of Expiring Policies        |
| Retention Rate               | Percentage of # of policies retained between the CY Month and PY Month        |
| Revenue Growth               | Percentage of # of policies retained between the CY Month and PY Month        |
| Sales Velocity               | New Business Revenue/ PY Total Revenue        |
| Total Revenue                | Total Revenue  (Month)        |
| Rate Increase ($)			   | CY Total Revenue - CY New Business Revenue - CY Cancellation Revenue		|
| Rate Increase (%)			   | (Rate Increase ($) - PY Total Revenue) / Rate Increase ($)

## To generate forecasts

Create the following folders at the top level of this project directory:

- inputs/data/raw
- inputs/data/interim
- outputs/results/pl
- outputs/results/cl
- outputs/reports/pl
- outputs/reports/cl

If you just want to use the optimized models to generate forecasts, put the input data in *Test/input/raw/*.
Then run **prep_data.py** followed by **build_forecast.py**. **build_forecast.py** outputs the following files in outputs/results/[LOB]/ (where LOB is "cl" or "pl"):

- forecasts_LOB.csv
- forecast_metrics_LOB.csv
- scenarios_LOB.csv

forecast_metrics_LOB.csv contains the input data used to create the yearly forecast. forecasts.csv contains the actual forecast data.
scenarios_LOB.csv contains the various scenarios generated (forecasts with +/- x% to various metrics) and shows what the forecast is for each scenario.
You can then run **scenario_reshape.py** to reshape the output into a format compatible with the Excel visualization tool.

If you would like to customize the scenarios in the final forecast output, you can open *inputs/configuration/scenarios_input.csv*. Each row
represents a different scenario used by the forecasting tool to produce a forecast scenario.

## Scripts in this repository

First, run **prep_data.py** to preprocess and transform the data set and split it into training/test sets.

**analyze_metrics.py** runs some EDA code and outputs some analysis of each feature.

**optimize_models.py** runs cross validation on a variety of models and hyperparameters to optimize each model type.

**optimize_arima.py** does the same for ARIMA modeling.

**compare_models.py** compares optimized models on the test set and outputs some figures/results.

**modeling_rev_metrics** explores some modeling of 2 operational metrics using ARIMA. This was ultimately not used.

**build_forecast.py** uses the full data set to fit optimized models and creates a forecast for the next year.

**config.py** stores string paths. Make sure to create the directories in the repository for input/output folders.

**utilities.py** contains helper/convenience functions used by multiple scripts.

**modeling.py** contains functions pertaining to building and scoring models.

**scenario_reshape.py** reshapes outputs from scenario modeling into Excel friendly matrix.
