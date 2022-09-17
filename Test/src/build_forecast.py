import pandas as pd
import sklearn.linear_model as lm
from matplotlib import pyplot as plt
import statsmodels.tsa.statespace.sarimax as sarimax
import numpy as np

from utilities import *
from modeling import build_model, model_with_params, score_validation
import config as cfg

""" 
build_forecast
given historical revenue/operational data, build 12 month revenue forecast for the next calendar year
outputs forecasts, forecast metrics (metrics used to build the forecast), and scenarios in which input metrics
are changed in future data to show how forecasts are impacted.

To change what scenarios are forecasted, edit the table found in Test/inputs/configuration/scenarios_input.csv
"""

CURRENT_YEAR = 2022

#  scenario_filepath = cfg.scenario_path + 'scenarios_input.csv'
scenario_filepath = cfg.additional_scenarios_path + "additional_scenarios.csv"

# set Line of Business for predictions
lob = "cl"


# returns model for predicting operational metrics
def get_op_metric_model(metric, lob, dataset):
    model = None
    if metric == "New Business Revenue":
        if lob == "pl":
            model = model_with_params("ridge", {"alpha": 0.408})
        elif lob == "cl":
            model = model_with_params("ridge", {"alpha": 0.408})
    elif metric == "Cancellation Revenue":
        if lob == "pl":
            model = model_with_params("ridge", {"alpha": 0.408})
        elif lob == "cl":
            model = model_with_params("svr", {"c": 1e8, "eps": 0.01})
    return model.fit(dataset.drop(dropped_vars, axis=1), dataset[f"{metric} 12"])


# generates each scenario as a separate row in the scenarios dataframe
def generate_scenarios():

    scenarios = pd.read_csv(scenario_filepath)

    # generate predictions for july-dec 2021
    dataLastYear = full[(full["Year"] == CURRENT_YEAR - 1) & (full["Month"] >= 9)]  # 7
    predThisYear = monthly_predictor.predict(dataLastYear.drop(dropped_vars, axis=1))
    dataThisYear = full[(full["Year"] == CURRENT_YEAR) & (full["Month"] < 9)]  # 7
    predNextYear = monthly_predictor.predict(dataThisYear.drop(dropped_vars, axis=1))

    # get predictions for existing data
    for i, ym in zip(
        range(predThisYear.shape[0]),
        range(int(str(CURRENT_YEAR) + "07"), int(str(CURRENT_YEAR) + "13")),
    ):
        scenarios[ym] = predThisYear[i]
    for i, ym in zip(
        range(predNextYear.shape[0]),
        range(int(str(CURRENT_YEAR + 1) + "01"), int(str(CURRENT_YEAR + 1) + "07")),
    ):
        scenarios[ym] = predNextYear[i]
    for ym in range(
        int(str(CURRENT_YEAR + 1) + "07"), int(str(CURRENT_YEAR + 1) + "13")
    ):
        scenarios[ym] = np.nan

    count_row = scenarios.shape[0]  # Gives number of rows
    count_col = scenarios.shape[1]  # Gives number of columns

    for i in range(scenarios.shape[0]):
        change_columns = list(
            filter(lambda e: e in full.columns.tolist(), scenarios.columns.tolist())
        )
        changes = scenarios.iloc[i, :][change_columns]
        rows = generate_rows(
            full[full["Year"] == CURRENT_YEAR].drop(dropped_vars, axis=1),
            forecaster,
            full,
            changes,
        )
        rows.drop(dropped_vars, axis=1, inplace=True, errors="ignore")
        pred = monthly_predictor.predict(rows)
        for j, ym in zip(range(len(pred)), range(202207, 202313)):
            scenarios[ym][i] = pred[j]

    count_row = change_columns.shape[0]  # Gives number of rows
    count_col = changes.shape[1]  # Gives number of columns

    # generate forecasts for operational metrics
    rows = generate_rows(
        full[full["Year"] == CURRENT_YEAR].drop(dropped_vars, axis=1), forecaster, full
    )
    cr_model = get_op_metric_model("Cancellation Revenue", lob, train)
    nbr_model = get_op_metric_model("New Business Revenue", lob, train)
    combined = pd.concat(
        [dataLastYear, dataThisYear, rows]
    )  # combine data2020, data2021 and rows
    nbr = nbr_model.predict(combined.drop(dropped_vars, axis=1))
    cr = cr_model.predict(combined.drop(dropped_vars, axis=1))

    # return scenarios, pd.concat([dataLastYear, dataThisYear]), nbr, cr
    return scenarios, combined, nbr, cr


# generate input data for various sets of operational metrics
def generate_rows(monthly_data, arima_model, before_, changes_=pd.Series()):

    new_business_rev = before_.loc[
        (before_["Year"] == (CURRENT_YEAR - 1)) & (before_["Month"] >= 7),
        "New Business Revenue",
    ]
    cancellation_rev = before_.loc[
        (before_["Year"] == (CURRENT_YEAR - 1)) & (before_["Month"] >= 7),
        "Cancellation Revenue",
    ]

    # generate additional rows
    # year/months hardcoded for simplicity
    generated_rows = monthly_data.copy()
    generated_rows.index = list(
        range(int(str(CURRENT_YEAR) + "07"), int(str(CURRENT_YEAR) + "13"))
    )
    # Month (year is same)
    generated_rows["Month"] = range(7, 13)
    # PY Total Revenue = just last year's revenue
    generated_rows["PY Total Revenue"] = before_["Total Revenue"].tail(6).values

    # generated_rows['arima_output'] = arima_model.forecast(steps=18).values[12:]

    # ARIMA gives a forecast of total revenue
    generated_rows["Total Revenue"] = arima_model.forecast(steps=6).values

    # These operational metrics are being incremented up/down
    generated_rows["New Business Revenue"] = new_business_rev.values
    generated_rows["Cancellation Revenue"] = cancellation_rev.values

    # These operational metrics are calculated from others
    generated_rows["Rate Commission"] = generated_rows["Total Revenue"].values - (
        generated_rows["Cancellation Revenue"].values
        + generated_rows["New Business Revenue"].values
    )
    generated_rows["Rate Commission Percentage"] = (
        generated_rows["Rate Commission"].values
        - generated_rows["PY Total Revenue"].values
    ) / generated_rows["Rate Commission"].values

    for metric in changes_.index:
        generated_rows[metric] = (
            generated_rows[metric] + (generated_rows[metric] * 0.01) * changes_[metric]
        )
        if metric == "Rate Commission" or metric == "Rate Commission Percentage":
            if metric == "Rate Commission":
                # (Rate Commission-PY Total Revenue) / Rate Commission
                generated_rows["Rate Commission Percentage"] = (
                    generated_rows["Rate Commission"]
                    - generated_rows["PY Total Revenue"]
                ) / generated_rows["Rate Commission"]
            if metric == "Rate Commission Percentage":
                generated_rows["Rate Commission"] = 1 / (
                    (1 - generated_rows["Rate Commission Percentage"])
                    / generated_rows["PY Total Revenue"]
                )
    return generated_rows


# load data
train = pd.read_csv(cfg.interim_path + f"{lob}_train12.csv", index_col="Year_Month")
full = pd.read_csv(cfg.interim_path + f"{lob}_full12.csv", index_col="Year_Month")
dropped_vars = ["Total Revenue 12", "Total Revenue 18", "Total Revenue 24"]

# fill in NAs at beginning of data
column_means = train.mean()
full = full.fillna(column_means)

# build ARIMA model
ma = 2 if lob == "pl" else 3
forecaster = sarimax.SARIMAX(
    full["Total Revenue"], order=(10, 1, ma), seasonal_order=(10, 1, ma, 12)
).fit(method="nm")
# full['arima_output'] = np.concatenate([forecaster.fittedvalues[12:].values, forecaster.forecast(steps=12).values])

# remove rows at end of dataset without future revenue
train = full.iloc[:-10, :]

# build lasso revenue models
lm_type = "lasso"
if lm_type == "ridge":
    monthly_predictor = lm.Ridge(alpha=1.02, normalize=True).fit(
        train.drop(dropped_vars, axis=1), train["Total Revenue 12"]
    )
elif lm_type == "linear":
    monthly_predictor = lm.Lasso(alpha=1e-100, normalize=False, positive=True).fit(
        train.drop(dropped_vars, axis=1), train["Total Revenue 12"]
    )
    # monthly_predictor = lm.Lasso(alpha=1e-100, normalize=False, positive=True).fit(train.drop(dropped_vars, axis=1), train['Total Revenue 12'])
else:
    monthly_predictor = lm.Lasso(alpha=100, normalize=False, positive=False).fit(
        train.drop(dropped_vars, axis=1), train["Total Revenue 12"]
    )
twoyear_predictor = lm.Lasso(alpha=200, normalize=False, positive=False).fit(
    train.drop(dropped_vars, axis=1), train["Total Revenue 24"]
)

scenarios, forecast_inputs, nbr, cr = generate_scenarios()
# get lasso 12 month model's predictions for Jan 2021-July 2022 (first row of scenarios table)
preds = scenarios.drop(full.columns, axis=1, errors="ignore").head(1).T
preds["lasso12"] = preds[0]
preds.drop(0, axis=1)
preds["arima"] = forecaster.forecast(steps=18).values
preds["lasso24"] = twoyear_predictor.predict(
    full[(full["Year"] >= CURRENT_YEAR - 1)].drop(dropped_vars, axis=1)
)[:18]
preds["Cancellation Revenue forecast"] = cr
preds["New Business Revenue forecast"] = nbr

# save forecasting data
forecast_inputs.to_csv(cfg.results_path + f"{lob}/forecast_metrics_{lob}_{lm_type}.csv")
preds.to_csv(cfg.results_path + f"{lob}/forecasts_{lob}_{lm_type}.csv")
scenarios.to_csv(cfg.results_path + f"{lob}/scenarios_{lob}_{lm_type}.csv")
