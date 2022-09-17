import pandas as pd
from scipy.stats import pearsonr
import statsmodels.tsa.stattools as smt
from matplotlib import pyplot as plt
import config as cfg

'''
analyze_metrics

This script produces some reports and figures pertaining to each metric.

'''


cl = pd.read_csv(cfg.interim_path + 'cl_train12.csv', index_col='Year_Month')
pl = pd.read_csv(cfg.interim_path + 'pl_train12.csv', index_col='Year_Month')

# generate correlation and covariance matrices
cl.corr().to_csv(cfg.reports_path + 'cl_corr_matrix.csv')
cl.cov().to_csv(cfg.reports_path + 'cl_cov_matrix.csv')
pl.corr().to_csv(cfg.reports_path + 'pl_corr_matrix.csv')
pl.cov().to_csv(cfg.reports_path + 'pl_cov_matrix.csv')

for line in [(cl, 'cl'), (pl, 'pl')]:

    df = line[0]
    metrics = df.columns
    metrics = metrics.drop(['Total Revenue 12'])
    results = pd.DataFrame({'Metric': pd.Series(metrics)})

    for metric in metrics:
        # correlation with future rev
        results.loc[results['Metric'] == metric, 'Mean'] = df[metric].mean()
        results.loc[results['Metric'] == metric, 'Stdev'] = df[metric].std()
        dropped_na = df[[metric, "Total Revenue 12"]].dropna()
        corr = pearsonr(dropped_na[metric], dropped_na['Total Revenue 12'])
        results.loc[results['Metric'] == metric, 'Correlation with Total Rev 12 Mo Ahead'] = corr[0]
        results.loc[results['Metric'] == metric, 'p'] = corr[1]

        # metric distribution
        plt.hist(dropped_na[metric])
        plt.title(line[1] + ' ' + metric + ' distribution')
        plt.savefig(cfg.reports_path + 'distributions/' + line[1] + '/' + metric + '.png')
        plt.close()

        # CCF
        conf95 = 2 / (dropped_na.shape[0] ** 0.5)
        ccf = smt.ccf(dropped_na['Total Revenue 12'], dropped_na[metric], adjusted=False)
        #fig, ax = plt.subplots()
        #ax.xcorr(dropped_na['Total Revenue'], dropped_na[metric])
        ##plt.show()
        plt.stem(ccf[:36], basefmt='C2')
        plt.grid(True)
        plt.title(line[1] + ' CCF: ' + metric + ' and Revenue 12 Mo Ahead')
        #baseline.color = 'blue'
        plt.hlines([(-1 * conf95), conf95], xmin=0, xmax=36, colors='red')
        plt.savefig(cfg.reports_path + 'ccf_plots/' + line[1] + '/' + metric + '.png')
        plt.close()

    results.to_csv(cfg.reports_path + line[1] + '_feature_stats.csv', index=False)
