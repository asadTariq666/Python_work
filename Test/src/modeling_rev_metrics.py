import statsmodels.tsa.statespace.sarimax as sarimax
import statsmodels.tsa.arima_model as arima
import pandas as pd

from utilities import *
import config as cfg

# Build ARIMA models of operational metrics important to revenue growth
# New Business Revenue
# Cancellation Revenue

# parameters to ARIMA
#CL
#NBR
# ar 13 ma 4
#Canc Rev
# ar 1 ma 5
#PL
#NBR
# ar 11 ma 2
#Canc Rev
# ar 18 ma 7


# load data
mo_ahead = 12
lob = 'cl'
full = pd.read_csv(cfg.interim_path + f'{lob}_full.csv', index_col='Year_Month')
train = pd.read_csv(cfg.interim_path + f'{lob}_train12.csv', index_col='Year_Month')
test = pd.read_csv(cfg.interim_path + f'{lob}_test12.csv', index_col='Year_Month')

# build ARIMA model
ma = 2 if lob == 'pl' else 3
model = sarimax.SARIMAX(train['New Business Revenue'], order=(10, 1, 2), seasonal_order=(10,1,2,12)).fit(method='nm')
pred = model.forecast(steps=test.shape[0] + 12)[12:].values
actual = test['New Business Revenue 12']
errors_nbr = percent_error(actual, pred)
score = mape(actual, pred)
print(score)
print(min(errors_nbr), max(errors_nbr), score)

model = sarimax.SARIMAX(train['Cancellation Revenue'], order=(10, 1, 2), seasonal_order=(10,1,2,12)).fit(method='nm')
pred = model.forecast(steps=test.shape[0]+12)[12:].values
actual = test['Cancellation Revenue 12']
errors_cr = percent_error(actual, pred)
score = mape(actual, pred)
print(score)
print(min(errors_cr), max(errors_cr), score)
