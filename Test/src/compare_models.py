import pandas as pd
import sklearn.linear_model as lm
import sklearn.ensemble as ensemble
import sklearn.tree as tree
from sklearn.svm import SVR
from matplotlib import pyplot as plt
import statsmodels.tsa.statespace.sarimax as sarimax
import numpy as np

from utilities import *
import config as cfg
from modeling import build_model, model_with_params


# target variables- these are dropped in certain situations
dropped_vars = ['Total Revenue 12', 'Total Revenue 18', 'Total Revenue 24', 'New Business Revenue 12', 'Cancellation Revenue 12']

# set LOB, how many months ahead to predict
#metric = 'Total Revenue'
#metric = 'New Business Revenue'
metric = 'Cancellation Revenue'
lob = 'cl'
mo_ahead = 12

# load data
train = pd.read_csv(cfg.interim_path + f'{lob}_train{mo_ahead}.csv', index_col='Year_Month')
test = pd.read_csv(cfg.interim_path + f'{lob}_test{mo_ahead}.csv', index_col='Year_Month')

# each model's parameters used to create it. just for convenience
tot_rev_models_pl = [
    ('lasso', model_with_params('lasso', {'alpha': 100})),
    ('ridge', model_with_params('ridge', {'alpha': 1.02})),
    ('svr', model_with_params('svr', {'c': 500000, 'eps': 0.01})),
    ('tree', model_with_params('tree', {'max_depth': 5, 'min_imp_dec': 0.16})),
    ('rf', model_with_params('rf', {'max_depth': 6, 'min_imp_dec': 0.02, 'num_est': 20}))
]

tot_rev_models_cl = [
    ('lasso', model_with_params('lasso', {'alpha': 100})),
    ('ridge', model_with_params('ridge', {'alpha': 1.22})),
    ('svr', model_with_params('svr', {'c': 1e8, 'eps': 0.01})),
    ('tree', model_with_params('tree', {'max_depth': 4, 'min_imp_dec': 0.08})),
    ('rf', model_with_params('rf', {'max_depth': 7, 'min_imp_dec': 0.2, 'num_est': 20}))
]

nbr_models_pl = [
    ('lasso', model_with_params('lasso', {'alpha': 1000})),
    ('ridge', model_with_params('ridge', {'alpha': 0.408})),
    ('svr', model_with_params('svr', {'c': 500000, 'eps': 0.01})),
    ('tree', model_with_params('tree', {'max_depth': 9, 'min_imp_dec': 0.2})),
    ('rf', model_with_params('rf', {'max_depth': 5, 'min_imp_dec': 0.065, 'num_est': 40}))
]

nbr_models_cl = [
    ('lasso', model_with_params('lasso', {'alpha': 3000})),
    ('ridge', model_with_params('ridge', {'alpha': 0.408})),
    ('svr', model_with_params('svr', {'c': 1e8, 'eps': 0.01})),
    ('tree', model_with_params('tree', {'max_depth': 11, 'min_imp_dec': 0.06})),
    ('rf', model_with_params('rf', {'max_depth': 4, 'min_imp_dec': 0.2, 'num_est': 20}))
]

models = nbr_models_pl if lob == 'pl' else nbr_models_cl

# compare each model type's performance
results_list = []
fitted_models = []
mins = []
maxes = []
model_preds = pd.DataFrame()
for model in models:
    print(model[0])
    fitted_model_, results, predictions_ = build_model(train, test, model[1], mo_ahead, dropped_vars, metric)
    fitted_models.append(fitted_model_)
    # performance metrics
    rmse = results['val']['RMSE']
    r2 = results['val']['R^2']
    mape_ = results['val']['MAPE']
    mins.append(results['val']['min_MAPE'])
    maxes.append(results['val']['max_MAPE'])
    # save predictions
    model_preds[model[0]] = predictions_
    model_preds['actual'] = test[metric + f' {mo_ahead}'].values
    results_list.append(results['val'])
    print(f'RMSE: {rmse}')
    print(f'R^2: {r2}')
    print(f'MAPE: {mape_}')

model_preds = model_preds[['actual', 'ridge']]
model_preds.to_csv(cfg.results_path + f'{lob}/test_set_details_{metric}_{lob}.csv')
predictions = pd.DataFrame()
for model_, fm in zip(models, fitted_models):
    pred_ = fm.predict(test.drop(dropped_vars, axis=1))
    predictions[model_[0]] = pred_
predictions.index = test[f'{metric} {mo_ahead}'].index
predictions['actual'] = test[f'{metric} {mo_ahead}']

names = [model[0] for model in models]
y_pos = np.arange(len(names))

# code below commented out for speed - ARIMA takes too long when only interested in comparing other models
'''
# ARIMA
print("ARIMA")
names.append('ARIMA')
y_pos = np.arange(len(names))
ar = 10
ma = 2 if lob == 'pl' else 3
model = sarimax.SARIMAX(train[metric], order=(ar, 1, ma), seasonal_order=(ar, 1, ma, 12)).fit(method='nm')
pred = model.forecast(test.shape[0]+mo_ahead)[mo_ahead:]
actual = test[f'{metric} {mo_ahead}']
rmse = mean_squared_error(pred, actual.values) ** 0.5
r2 = r2_score(pred, actual.values)
mape_ = mape(actual.values, pred)
results_list.append({"RMSE": rmse, "R^2": r2, "MAPE": mape_})
predictions["ARIMA"] = pred.values
'''
#predictions.to_csv(cfg.results_path + f'{lob}_test_set_predictions_{mo_ahead}.csv')

# plot performance metrics
performance_metrics = ['RMSE', 'R^2', 'MAPE']
for pm in performance_metrics:
    scores = [res[pm] for res in results_list]
    plt.bar(y_pos, scores, align='center', alpha=0.5)
    plt.xticks(y_pos, names)
    plt.ylabel(pm)
    plt.title(f'Test Set {pm} by Model Type {mo_ahead} Mo Ahead: {metric}')
    #plt.savefig(cfg.results_path + f'{lob}/model_comparison_{mo_ahead}_{pm}.png')
    plt.show()
    plt.close()

# uncomment to get model feature importances output if desired
'''
# outputs ceofficients of each feature in the model along with that feature's mean
lm = build_model(train, test, model_with_params('lasso', {'alpha': 100}), 12, metric)
column_names = train.drop(dropped_vars, axis=1).columns
importance = pd.DataFrame({'feature': column_names, 'coefficient': lm[0].coef_, 'mean': train.drop(dropped_vars, axis=1).mean()})
#importance.to_csv(cfg.reports_path + f'{lob}_coefficients.csv')
print(importance)
'''