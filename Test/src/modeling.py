import pandas as pd
import sklearn.linear_model as lm
import sklearn.ensemble as ensemble
import sklearn.tree as tree
from sklearn.svm import SVR
from matplotlib import pyplot as plt
import statsmodels.tsa.statespace.sarimax as sarimax
import numpy as np

from utilities import *


# get sklearn model object with parameters
def model_with_params(model_name_, params):
    if model_name_ == 'ridge':
        return lm.Ridge(alpha=params['alpha'], normalize=True)
    elif model_name_ == 'lasso':
        return lm.Lasso(alpha=params['alpha'], normalize=False)
    elif model_name_ == 'tree':
        return tree.DecisionTreeRegressor(max_depth=params['max_depth'], min_impurity_decrease=params['min_imp_dec'])
    elif model_name_ == 'rf':
        return ensemble.RandomForestRegressor(max_depth=params['max_depth'], min_impurity_decrease=params['min_imp_dec'], n_estimators=int(params['num_est']))
    elif model_name_ == 'svr':
        return SVR(C=params['c'], gamma='scale', epsilon=params['eps'], kernel='poly')
    else:
        raise NotImplementedError()


# fits a model to training data and tests on val
def build_model(train, val, model_, mo_ahead, dropped_vars_, metric='Total Revenue'):

    column_means = train.mean()
    train = train.fillna(column_means)

    model_ = model_.fit(train.drop(dropped_vars_, axis=1), train[f'{metric} {mo_ahead}'])
    results = {}
    results['training'] = score_validation(model_, train.drop(dropped_vars_, axis=1), train[f'{metric} {mo_ahead}'])
    results['val'], predictions_ = score_validation(model_, val.drop(dropped_vars_, axis=1), val[f'{metric} {mo_ahead}'])
    return model_, results, predictions_


# Return performance metrics for a given model on inputs x and actual values y
def score_validation(model_, x, y):
    pred_ = model_.predict(x)
    mse_ = mean_squared_error(y, pred_)
    r2_ = r2_score(y, pred_)
    errors = percent_error(y, pred_)
    return {"RMSE": mse_ ** 0.5, "R^2": r2_, "MAPE": mape(y, pred_), "min_MAPE": min(errors), "max_MAPE": max(errors)}, pred_