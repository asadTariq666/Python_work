import pandas as pd
from sklearn.preprocessing import StandardScaler

import config as cfg
import sklearn.linear_model as lm
import sklearn.ensemble as ensemble
import sklearn.tree as tree
from sklearn.svm import SVR
from matplotlib import pyplot as plt
import numpy as np

from utilities import *

from modeling import build_model, model_with_params

dropped_vars = ['Total Revenue 12', 'Total Revenue 18', 'Total Revenue 24', 'New Business Revenue 12', 'Cancellation Revenue 12']

# optimize parameters such that MAPE is minimized on validation sets


# For each fold, analyze performance on unseen data
def cross_validation(df, model_, mo_ahead_, metric='Total Revenue'):
    # Split data into k training sets + rest of the data as validation
    folds = [24, 30, 36]
    rmses = []
    r2s = []
    mape_s = []
    for k in folds:
        train = df.iloc[:k, :]
        val = df.iloc[k:(k+12), :]
        model_, result, _ = build_model(train, val, model_, mo_ahead_, dropped_vars, metric)
        rmses.append(result['val']['RMSE'])
        r2s.append(result['val']['R^2'])
        mape_s.append(result['val']['MAPE'])
    return model_, rmses, r2s, mape_s


metric = 'Total Revenue'
lob = 'pl'
mo_ahead = 12
df = pd.read_csv(cfg.interim_path + f'{lob}_train{mo_ahead}.csv', index_col='Year_Month')

# Ridge
print("Ridge")
# build results data frame - each row is a combination of hyperparams
ridge_results = pd.DataFrame({'Alpha': np.linspace(0, 10, 50)})
lasso_results = pd.DataFrame({'Alpha': np.linspace(20, 120, 100)})
ridge_results['MAPE'] = pd.Series(np.zeros(ridge_results.shape[0]))
for alpha in ridge_results['Alpha']:
    model = model_with_params('ridge', {'alpha': alpha})
    model, rmses, r2s, mape_s = cross_validation(df, model, mo_ahead)
    ridge_results.loc[ridge_results['Alpha'] == alpha, 'MAPE'] = np.mean(mape_s)
ridge_results.to_csv(cfg.results_path + f'{lob}/optimization/ridge_optimization_{mo_ahead}_{metric}.csv')

# LASSO
print("LASSO")
alphas = np.linspace(1, 3000, 100) if lob == 'cl' else np.linspace(40, 200, 140)
lasso_results = pd.DataFrame({'Alpha': alphas})
lasso_results['MAPE'] = pd.Series(np.zeros(lasso_results.shape[0]))
for alpha in lasso_results['Alpha']:
    model = model_with_params('lasso', {'alpha': alpha})
    model, rmses, r2s, mape_s = cross_validation(df, model, mo_ahead, metric)
    lasso_results.loc[lasso_results['Alpha'] == alpha, 'MAPE'] = np.mean(mape_s)
lasso_results.to_csv(cfg.results_path + f'{lob}/optimization/lasso_optimization_{mo_ahead}_{metric}.csv')

# Tree
print("Tree")
tree_results = pd.DataFrame({'Max Depth': range(3, 13)})
impurity_decrease = pd.Series(np.linspace(0.02, 0.2, 10), name='Min Impurity Decrease')
tree_results = tree_results.merge(impurity_decrease, how='cross')
tree_results['MAPE'] = pd.Series(np.zeros(tree_results.shape[0]))
for index, row in tree_results.iterrows():
    max_depth = row['Max Depth']
    min_impurity = row['Min Impurity Decrease']
    model = model_with_params('tree', {'max_depth': max_depth, 'min_imp_dec': min_impurity})
    model, rmses, r2s, mape_s = cross_validation(df, model, mo_ahead, metric)
    tree_results.loc[(tree_results['Max Depth'] == max_depth) & (tree_results['Min Impurity Decrease'] == min_impurity), 'MAPE'] = np.mean(mape_s)
tree_results.to_csv(cfg.results_path + f'{lob}/optimization/tree_optimization_{mo_ahead}_{metric}.csv')

# Forest
print("Random Forest")
rf_results = pd.DataFrame({'Max Depth': range(3, 8)})
impurity_decrease = pd.Series(np.linspace(0.02, 0.2, 5), name='Min Impurity Decrease')
estimators = pd.Series(range(20, 120, 20), name='Num Estimators')
rf_results = rf_results.merge(impurity_decrease, how='cross')
rf_results = rf_results.merge(estimators, how='cross')
rf_results['MAPE'] = pd.Series(np.zeros(rf_results.shape[0]))
for index, row in rf_results.iterrows():
    max_depth = row['Max Depth']
    min_impurity = row['Min Impurity Decrease']
    num_estimators = row['Num Estimators']
    model = model_with_params('rf', {'max_depth': max_depth, 'min_imp_dec': min_impurity, 'num_est':num_estimators})
    model, rmses, r2s, mape_s = cross_validation(df, model, mo_ahead, metric)
    rf_results.loc[(rf_results['Max Depth'] == max_depth) & (rf_results['Min Impurity Decrease'] == min_impurity) & (rf_results['Num Estimators'] == num_estimators), 'MAPE'] = np.mean(mape_s)
rf_results.to_csv(cfg.results_path + f'{lob}/optimization/random_forest_optimization_{mo_ahead}_{metric}.csv')

# SVR
print("SVR")
cvals = [1e6, 1e7, 1e8, 1e9, 1e10] if lob == 'cl' else [1e5, 500000, 1e6, 5000000, 1e7]
svr_results = pd.DataFrame({'C': cvals})
epsilon = pd.Series([1e-2, 1e-3, 1e-4, 1e-5], name='Epsilon')
svr_results = svr_results.merge(epsilon, how='cross')
svr_results['MAPE'] = pd.Series(np.zeros(svr_results.shape[0]))
for index, row in svr_results.iterrows():
    c = row['C']
    eps = row['Epsilon']
    model = model_with_params('svr', {'c': c, 'eps': eps})
    model, rmses, r2s, mape_s = cross_validation(df, model, mo_ahead, metric)
    svr_results.loc[(svr_results['C'] == c) & (svr_results['Epsilon'] == eps), 'MAPE'] = np.mean(mape_s)
svr_results.to_csv(cfg.results_path + f'{lob}/optimization/svr_optimization_{mo_ahead}_{metric}.csv')
