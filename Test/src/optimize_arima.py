

import pandas as pd
import statsmodels.tsa.statespace.sarimax as sarimax
from statsmodels.tsa.stattools import pacf
import statsmodels.tsa.vector_ar.var_model as var
import config as cfg
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
from matplotlib import pyplot as plt

from utilities import *


# given a model, forecast the next n months and plot them against the actual data
def forecast_plot(model, y_actual):
    y_pred = model.forecast(steps=y_actual.shape[0]+12)[12:]
    plt.plot(range(y_pred.shape[0]), y_pred)
    plt.plot(range(y_actual.shape[0]), y_actual.values)
    plt.ticklabel_format(style='plain')
    #plt.ticklabel_format(useOffset=False)
    plt.show()


def cross_validation(df_, params_):
    # Split data into k training sets + rest of the data as validation
    folds = [18, 24, 30, 36]
    rmses = []
    r2s = []
    mape_s = []
    aics = []
    for k in folds:
        train = df_.iloc[:k, :]
        val = df_.iloc[k:(k+12), :]
        model_, result = build_model(train, val, params_)
        if not np.isnan(result['AIC']):
            rmses.append(result['RMSE'])
            r2s.append(result['R^2'])
            mape_s.append(result['MAPE'])
            aics.append(result['AIC'])
            #forecast_plot(model_, val['Total Revenue 12 Mo Ahead'])
    return model_, rmses, r2s, mape_s, aics


def score_validation(model_, y):
    pred_ = model_.forecast(steps=y.shape[0])
    #pred_ = model_.forecast(model_.y, steps=y.shape[0])
    #pred_ = pred_[:, 20]
    mse_ = mean_squared_error(y.values, pred_)
    r2_ = r2_score(y, pred_)
    return mse_ ** 0.5, r2_, mape(y.values, pred_)


def build_model(train_, val_, params_):

    #model = sarimax.SARIMAX(train_[metric], order=params_['order'], seasonal_order=params_['seasonal_order']).fit(method='nm')
    model = sarimax.SARIMAX(train_[metric], order=params_['order']).fit(method='nm')
    #model = var.VAR(train).fit()
    if not np.isnan(model.aic):
        # Validation results
        results = score_validation(model, val_[metric])
        print('validation results')
        print(f'RMSE:' + results["RMSE"])
        print(f'R^2:' + results["R^2"])
        print(f'MAPE:' + results["MAPE"])
        return model, {"RMSE": results["RMSE"], "R^2": results["R^2"], "MAPE": results["MAPE"], "AIC": model.aic}
    else:
        return model, {'AIC': np.nan}
    #forecast_plot(model, val_['Total Revenue 12 Mo Ahead'])

metric = 'Cancellation Revenue'
#metric = 'New Business Revenue'
#build_model('cl')
lob = 'pl'
df = pd.read_csv(cfg.interim_path + f'{lob}_train.csv', index_col='Year_Month')

# ARIMA
print("ARIMA")

pacf_func = pacf(df[metric], nlags=20)
print(list(zip(range(20), pacf_func)))
arima_results = pd.DataFrame({'AR': range(1, 20)})
ma = pd.DataFrame({'MA': range(1, 10)})
arima_results = arima_results.merge(ma, how='cross')
arima_results['MAPE'] = pd.Series(np.zeros(arima_results.shape[0]))
for index, row in arima_results.iterrows():
    ar = row['AR']
    ma = row['MA']
    print(f"AR: {ar} MA: {ma}")
    params = {'order': (ar, 1, ma), 'seasonal_order': (1, 1, ma, 12)}
    model, rmses, r2s, mape_s, aic = cross_validation(df, params)
    arima_results.loc[(arima_results['AR'] == ar) & (arima_results['MA'] == ma), 'MAPE'] = np.mean(mape_s)
    arima_results.loc[(arima_results['AR'] == ar) & (arima_results['MA'] == ma), 'AIC'] = np.mean(aic)
arima_results.to_csv(cfg.results_path + f'{lob}/optimization/arima_optimization_{metric}.csv')
