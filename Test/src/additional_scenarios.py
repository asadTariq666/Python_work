import pandas as pd
import config as cfg

'''
additinal_scenarios
Once the dataset is prepared, you can run this code to generate some additional scenarios based on various assumptions
about the operational metrics. For example, assume you want to know what will happen given: retention rate 88%,
cancellation rate 12%, new business growth 15%. This script finds the differences between the actual annualized average
of those metrics in the last year and the assumed percentages. Then, those differences are applied to the monthly data
to produce a forecasted set of operational metrics that will then be used for modeling in build_forecast.py.

outputs: additional rows to add to scenarios_input.csv
'''

lob = 'cl'

original_scenarios = pd.read_csv(cfg.scenario_path + 'scenarios_input.csv')

additional_scenarios = [{
    'Cancellation Rate': 0.12,
    'Retention Rate': 0.88,
    'New Business Revenue': 0.15
}]

full = pd.read_csv(cfg.interim_path + f'{lob}_full.csv', index_col='Year_Month')
actual2021 = full[(full['Year'] == 2021) & (full['Month'] < 7)]
projected2021 = full[(full['Year'] == 2020) & (full['Month'] >= 7)].copy()
projected2021['Year'] = 2021
concatenated = pd.concat([actual2021, projected2021])

additional_scenarios_changes = []
for scenario in additional_scenarios:
    changes = {}
    for metric in scenario.keys():
        print(metric)
        if metric.endswith('Rate'):
            actual_mean = concatenated[metric].mean()
            print(f'actual mean: {str(actual_mean)}')
            target_mean = scenario[metric]
            diff = target_mean - actual_mean
            adjusted_metric = concatenated[metric] + diff
            print(f'new mean: + {str(adjusted_metric.mean())}')
            adjustment = diff * 100
            print(adjustment)
            changes[metric] = adjustment
            if metric == 'Cancellation Rate':
                print('adjust cancellation revenue')
                pct_change_rev = adjustment / actual_mean
                changes['Cancellation Revenue'] = pct_change_rev
        else:  # Just New Business Rev for now
            cy_nbr = concatenated[metric].sum()
            py_nbr = full[full['Year'] == 2020][metric].sum()
            growth_annual = cy_nbr - py_nbr
            growth_annual_rate = (growth_annual / py_nbr) * 100
            print(f'actual growth: {growth_annual_rate}')
            target_growth_cad = scenario[metric] * py_nbr
            diff = target_growth_cad - growth_annual
            adjusted_metric = concatenated[metric] + (diff / 12)
            adjusted_growth_rate = ((adjusted_metric.sum() - py_nbr) / py_nbr) * 100
            print(f'new growth rate: {adjusted_growth_rate}')
            adjustment = (diff / py_nbr) * 100
            print(adjustment)
            changes[metric] = adjustment

    # ensure that there is a value for every row in scnearios csv
    for metric in original_scenarios.columns:
        if metric not in changes:
            changes[metric] = 0
    additional_scenarios_changes.append(changes)

for scenario in additional_scenarios_changes:
    original_scenarios = original_scenarios.append(scenario, ignore_index=True)

original_scenarios.fillna(0, inplace=True)
original_scenarios.to_csv(cfg.additional_scenarios_path + 'additional_scenarios.csv', index=False)
