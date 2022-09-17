import pandas as pd
import numpy as np
import config as cfg

'''
prep_data
This script takes the raw input data file (assumed to be located in Test/inputs/raw/)
and transforms it for use in modeling and forecasting. This script places the resulting files in the folder
Test/inputs/interim/, separated by line of business.
'''

# input file - change this if your raw input has a different name
#input_filename = 'revforecastdata-20210709.csv'
input_filename = 'Key_Measure_Product__202208.csv'

df = pd.read_csv(cfg.raw_path + f'{input_filename}')
#df = pd.read_excel(cfg.raw_path + f'{input_filename}')
df.set_index('Year_Month', inplace=True)
metrics = df['Measure_Name'].unique()

# separate data by LOB
cl = df[df['LOB'] == 'CL']
pl = df[df['LOB'] == 'PL']


# From the raw data, transform data so that each column represents one of the metrics
def create_table(df_, lob):

    df_table = pd.DataFrame(index=df_.index)
    for metric in metrics:  # gather metrics
        df_table[metric] = df_.loc[cl['Measure_Name'] == metric, 'Value']

    # sort chronologically
    df_table_dupsremoved = df_table[~df_table.index.duplicated(keep='first')].sort_index()

    # remove rows without revenue data
    df_table_dupsremoved.dropna(subset=['Total Revenue'], inplace=True)

    # Create target variables by shifting metrics backward in time
    df_table_dupsremoved['Total Revenue 12'] = df_table_dupsremoved['Total Revenue'].shift(periods=-12)
    df_table_dupsremoved['Total Revenue 18'] = df_table_dupsremoved['Total Revenue'].shift(periods=-18)
    df_table_dupsremoved['Total Revenue 24'] = df_table_dupsremoved['Total Revenue'].shift(periods=-24)
    df_table_dupsremoved['New Business Revenue 12'] = df_table_dupsremoved['New Business Revenue'].shift(periods=-12)
    df_table_dupsremoved['Cancellation Revenue 12'] = df_table_dupsremoved['Cancellation Revenue'].shift(periods=-12)

    df_table_dupsremoved['Year'] = pd.Series(df_table_dupsremoved.index).apply(lambda e: int(str(e)[:4])).values
    df_table_dupsremoved['Month'] = pd.Series(df_table_dupsremoved.index).apply(lambda e: int(str(e)[4:])).values

    if lob == 'cl':
        # Adjusts outliers for the month of January 2021 and March/April 2020
        means = df_table_dupsremoved.mean()
        mean_cancel = means['Cancellation Rate']
        mean_retention = means['Retention Rate']
        mean_renewal = means['Renewal Rate']
        
        # January 2021
        # time periods hardcoded for simplicity
        df_table_dupsremoved.loc[(df_table_dupsremoved['Year'] == 2021) & (df_table_dupsremoved['Month'] == 1), 'Cancellation Rate'] = mean_cancel
        df_table_dupsremoved.loc[(df_table_dupsremoved['Year'] == 2021) & (df_table_dupsremoved['Month'] == 1), 'Retention Rate'] = mean_retention
        df_table_dupsremoved.loc[(df_table_dupsremoved['Year'] == 2021) & (df_table_dupsremoved['Month'] == 1), 'Renewal Rate'] = mean_renewal

        # March 2020
        # time periods hardcoded for simplicity
        mean_rev = df_table_dupsremoved.loc[df_table_dupsremoved['Year'] == 2020, "Total Revenue"].mean()
        df_table_dupsremoved.loc[(df_table_dupsremoved['Year'] == 2020) & (
                    df_table_dupsremoved['Month'] == 3), 'Cancellation Rate'] = mean_cancel
        df_table_dupsremoved.loc[(df_table_dupsremoved['Year'] == 2020) & (
                    df_table_dupsremoved['Month'] == 3), 'Retention Rate'] = mean_retention
        df_table_dupsremoved.loc[(df_table_dupsremoved['Year'] == 2020) & (
                    df_table_dupsremoved['Month'] == 3), 'Renewal Rate'] = mean_renewal
        df_table_dupsremoved.loc[(df_table_dupsremoved['Year'] == 2020) & (
                df_table_dupsremoved['Month'] == 3), 'Total Revenue'] = mean_rev
        
        # April 2020
        # time periods hardcoded for simplicity
        df_table_dupsremoved.loc[(df_table_dupsremoved['Year'] == 2020) & (
                df_table_dupsremoved['Month'] == 4), 'Cancellation Rate'] = mean_cancel
        df_table_dupsremoved.loc[(df_table_dupsremoved['Year'] == 2020) & (
                df_table_dupsremoved['Month'] == 4), 'Retention Rate'] = mean_retention
        df_table_dupsremoved.loc[(df_table_dupsremoved['Year'] == 2020) & (
                df_table_dupsremoved['Month'] == 4), 'Renewal Rate'] = mean_renewal
        df_table_dupsremoved.loc[(df_table_dupsremoved['Year'] == 2020) & (
                df_table_dupsremoved['Month'] == 4), 'Total Revenue'] = mean_rev

    # split into training and test data
    months_ahead = [12, 18, 24]
    for mo in months_ahead:
        test_size = 12  # months
        df12 = df_table_dupsremoved.dropna(subset=[f'Total Revenue {mo}'])  # these have no future rev data
        df12_test = df12.tail(test_size)
        df12_train = df12.iloc[:-test_size, :]
        df12_train.to_csv(cfg.interim_path + f'{lob}_train{mo}.csv')
        df12_test.to_csv(cfg.interim_path + f'{lob}_test{mo}.csv')
        df_table_dupsremoved.to_csv(cfg.interim_path + f'{lob}_full{mo}.csv')


# run the script
create_table(cl, 'cl')
create_table(pl, 'pl')
