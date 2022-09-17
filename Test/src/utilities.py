import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import sklearn.linear_model as lm
import sklearn.ensemble as ensemble
import sklearn.tree as tree
from sklearn.svm import SVR


# MAPE (Mean Absolute Percentage Error)
def mape(y_true, y_pred):
    return 100 * np.mean(np.abs((y_true - y_pred) / y_true))


# Absolute Percentage Error, sample-wise
def percent_error(y_true, y_pred):
    return 100 * np.abs((y_true - y_pred) / y_true)


# eliminate outliers (outside of a feature's median by 3 standard deviations or more)
def eliminate_outliers(df_):
    for column in df_.columns:
        median = df_[column].median()
        std = df_[column].std()
        outliers = (df_[column] - median).abs() > (std * 3)
        df_.loc[outliers, column] = median
        #df_[column].fillna(median, inplace=True)
    return df_
