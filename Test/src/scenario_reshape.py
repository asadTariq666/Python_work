'''
Function to reshape data generated from scenario forecasts into an Excel friendly matrix.

'''

# %% Import Libraries
import pandas as pd
import os
import config as cfg
import numpy as np

# %% Set Parameters and Load Data
line = 'CL'
#model = 'ridge'
model = 'lasso'
#model = 'linear'
csvfile = cfg.results_path + f'{line.lower()}/scenarios_{line.lower()}_{model}.csv'

df = pd.read_csv(csvfile, low_memory=False)


# %% Reshape Data
df = df.T.copy()
df.reset_index(inplace=True, drop=False)

# Old method for renaming year+month
# for i in [5,6,7]:
#     df.loc[i,'index'] = '20220' + str(df.loc[i,'index'])
# for i in [8,9,10]:
#     df.loc[i,'index'] = '2022' + str(df.loc[i,'index'])

for i in [1, 2, 3, 4, 5, 6, 7]: # increase this if there are more features
    for j in list(range(0, 56)):
        if df.loc[i, j] != 0:
            df.loc[i, j] = df.loc[i, 'index'] + '_' + str(df.loc[i,j])

newdf = pd.DataFrame()

for i in range(0, 56):
    tdf = df.loc[:, ['index', i]]
    tdf.columns = ['Year_Month', 'Value']
    tdf['LOB'] = line
    tdf['CY_Value'] = 'NULL'
    tdf['PY_Value'] = 'NULL'

    if all(tdf['Value'].iloc[1:7] == 0):
        tdf['Scenario'] = 'Base Prediction'
    else:
        tdf['Scenario'] = tdf.loc[tdf.index[1:7][tdf['Value'].iloc[1:7] != 0], 'Value'].iloc[0]

    newdf = newdf.append(tdf.iloc[8::, :]) # increase this number if there are more features

newdf = newdf[['Year_Month', 'LOB', 'Value', 'CY_Value', 'PY_Value', 'Scenario']]


# %% Save Output into CSV
newdf.to_csv(cfg.results_path + f'{line.lower()}/scenarios_{line.lower()}_{model}_reshape.csv')

