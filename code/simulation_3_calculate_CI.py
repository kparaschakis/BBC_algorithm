# Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from compute_CI_test import percentile_uniformity
from matplotlib.backends.backend_pdf import PdfPages

# Read simulation configurations
sim_configurations = sorted(os.listdir('../simulated_data_results'))
sim_configurations = [r.split('_theoretical')[0] for r in sim_configurations if '_theoretical' in r]
# Create summaries table
summaries = pd.DataFrame(columns=['configuration', 'average_theoretical',
                                  'average_pooled_lower', 'pooled_uniformity', 'pooled_0.95_CI_inclusion',
                                  'average_fold_lower', 'fold_uniformity', 'fold_0.95_CI_inclusion'])
# Loop over simulation results
plot_pdf_saver = PdfPages('../percentile_plots.pdf')
new_plot = plt.figure()
for r in range(len(sim_configurations)):
    summaries.at[r, 'configuration'] = sim_configurations[r]
    theoretical_result =\
        pd.read_csv('../simulated_data_results/' + sim_configurations[r] + '_theoretical.csv')['0'].values
    summaries.at[r, 'average_theoretical'] = np.mean(theoretical_result)
    pooled_result = pd.read_csv('../simulated_data_results/' + sim_configurations[r] + '_pooled.csv').values
    if np.isnan(pooled_result).sum() == 0:
        summaries.at[r, 'pooled_uniformity'] = percentile_uniformity(pooled_result, theoretical_result, alpha_=0.05,
                                                                     plot_baselines=True, plot_result=True)
        percentiles_pooled = [np.mean(pooled_result[p, :] <= theoretical_result[p])
                              for p in range(len(theoretical_result))]
        summaries.at[r, 'pooled_0.95_CI_inclusion'] = np.mean(np.array(percentiles_pooled) >= 0.05)
        lower_pooled = [sorted(pooled_result[p, :])[int(0.05*pooled_result.shape[1])]
                        for p in range(len(pooled_result))]
        summaries.at[r, 'average_pooled_lower'] = np.mean(lower_pooled)
    fold_result = pd.read_csv('../simulated_data_results/' + sim_configurations[r] + '_fold.csv').values
    if np.isnan(fold_result).sum() == 0:
        summaries.at[r, 'fold_uniformity'] = percentile_uniformity(fold_result, theoretical_result, alpha_=0.05,
                                                                   plot_baselines=False, plot_result=True)
        plt.title(sim_configurations[r])
        plt.legend(['Uniform', 'CI line', 'CI line', 'Pooled', 'Fold'])
        plot_pdf_saver.savefig(new_plot)
        plt.clf()
        percentiles_fold = [np.mean(fold_result[p, :] <= theoretical_result[p])
                            for p in range(len(theoretical_result))]
        summaries.at[r, 'fold_0.95_CI_inclusion'] = np.mean(np.array(percentiles_fold) >= 0.05)
        lower_fold = [sorted(fold_result[p, :])[int(0.05 * fold_result.shape[1])]
                      for p in range(len(fold_result))]
        summaries.at[r, 'average_fold_lower'] = np.mean(lower_fold)
    print('Simulation ' + str(r + 1) + ' of ' + str(len(sim_configurations)))
plot_pdf_saver.close()
plt.close()
# Save summaries
# noinspection PyTypeChecker
summaries.to_csv('../summaries.csv', index=False)

# Missing cases for mabt and co.
missing = pd.DataFrame(columns=['dataset', 'DeLong', 'Hanley_McNeil', 'bt', 'DeLong_10p', 'Hanley_McNeil_10p', 'bt_10p', 'mabt'])
for s in range(len(sim_configurations)):
    results = pd.read_csv('../multiplicity-adjusted_bootstrap_tilting/simulated_data_results/' + sim_configurations[s])
    missing.at[s, 'dataset'] = sim_configurations[s]
    missing.at[s, missing.columns[1:]] = results.isna().sum()[missing.columns[1:]]
# noinspection PyTypeChecker
missing.to_csv('../multiplicity-adjusted_bootstrap_tilting/simulated_data_missing_cases.csv')
