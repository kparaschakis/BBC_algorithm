# Libraries
import os
import pandas as pd
import numpy as np
import arff
import matplotlib.pyplot as plt
from compute_CI_test import percentile_uniformity
from matplotlib.backends.backend_pdf import PdfPages

# Read simulation configurations
real_datasets = os.listdir('../real_datasets_results/')
real_datasets = sorted([d.split('_fold')[0] for d in real_datasets if 'fold' in d])
# Create summaries table
summaries = pd.DataFrame(columns=['dataset', 'train_n', 'classes', 'balance',
                                  'average_holdOut_BBC',
                                  'average_pooled_lower', 'pooled_uniformity', 'pooled_0.95_CI_inclusion',
                                  'average_fold_lower', 'fold_uniformity', 'fold_0.95_CI_inclusion',
                                  'average_holdOut_validation', 'average_holdOut_evaluation',
                                  'average_DeLong_lower', 'DeLong_0.95_CI_inclusion',
                                  'average_HanleyMcNeil_lower', 'HanleyMcNeil_0.95_CI_inclusion',
                                  'average_bt_lower', 'bt_0.95_CI_inclusion',
                                  'average_DeLong10p_lower', 'DeLong10p_0.95_CI_inclusion',
                                  'average_HanleyMcNeil10p_lower', 'HanleyMcNeil10p_0.95_CI_inclusion',
                                  'average_bt10p_lower', 'bt10p_0.95_CI_inclusion',
                                  'average_mabt_lower', 'mabt_0.95_CI_inclusion'])
# Loop over simulation results
plot_pdf_saver = PdfPages('../real_datasets_percentile_plots.pdf')
new_plot = plt.figure()
for d in range(len(real_datasets)):
    splits = pd.read_csv('../real_datasets/train_test_split_indices/' + real_datasets[d] + '.csv')
    dataset_ = arff.load(open('../real_datasets/' + real_datasets[d] + '.arff'))
    data = pd.DataFrame(dataset_['data'])
    data.columns = [dataset_['attributes'][a][0].lower() for a in range(len(dataset_['attributes']))]
    #
    summaries.at[d, 'dataset'] = real_datasets[d]
    summaries.at[d, 'train_n'] = splits.shape[0]
    summaries.at[d, 'classes'] = len(data['class'].unique())
    summaries.at[d, 'balance'] = (data['class'].value_counts()/len(data)).min()
    #
    holdOut_results = pd.read_csv('../real_datasets_holdOut_performances/' + real_datasets[d] + '.csv', index_col=0)
    summaries.at[d, 'average_holdOut_BBC'] = holdOut_results.BBC_winner.mean()
    if holdOut_results.shape[1] > 1:
        summaries.at[d, 'average_holdOut_validation'] = holdOut_results.winners_valid.mean()
        summaries.at[d, 'average_holdOut_evaluation'] = holdOut_results.winners_eval.mean()
    #
    results_BBC_pooled = pd.read_csv('../real_datasets_results/' + real_datasets[d] + '_pooled.csv').values
    results_BBC_fold = pd.read_csv('../real_datasets_results/' + real_datasets[d] + '_fold.csv').values
    lower_pooled = [sorted(results_BBC_pooled[p, :])[int(0.05 * results_BBC_pooled.shape[1])]
                    for p in range(len(results_BBC_pooled))]
    lower_fold = [sorted(results_BBC_fold[p, :])[int(0.05 * results_BBC_fold.shape[1])]
                  for p in range(len(results_BBC_fold))]
    summaries.at[d, 'average_pooled_lower'] = np.mean(lower_pooled)
    summaries.at[d, 'average_fold_lower'] = np.mean(lower_fold)
    #
    percentiles_pooled = [np.mean(results_BBC_pooled[p, :] <= holdOut_results.loc[p, 'BBC_winner'])
                          for p in range(len(holdOut_results))]
    percentiles_fold = [np.mean(results_BBC_fold[p, :] <= holdOut_results.loc[p, 'BBC_winner'])
                        for p in range(len(holdOut_results))]
    summaries.at[d, 'pooled_0.95_CI_inclusion'] = np.mean(np.array(percentiles_pooled) >= 0.05)
    summaries.at[d, 'fold_0.95_CI_inclusion'] = np.mean(np.array(percentiles_fold) >= 0.05)
    #
    if holdOut_results.shape[1] > 1:
        results_mabt = pd.read_csv('../multiplicity-adjusted_bootstrap_tilting/real_datasets_results/' +\
                                   real_datasets[d] + '.csv')
        summaries.at[d, 'average_DeLong_lower'] = results_mabt['DeLong'].mean()
        summaries.at[d, 'average_HanleyMcNeil_lower'] = results_mabt['Hanley_McNeil'].mean()
        summaries.at[d, 'average_bt_lower'] = results_mabt['bt'].mean()
        summaries.at[d, 'average_DeLong10p_lower'] = results_mabt['DeLong_10p'].mean()
        summaries.at[d, 'average_HanleyMcNeil10p_lower'] = results_mabt['Hanley_McNeil_10p'].mean()
        summaries.at[d, 'average_bt10p_lower'] = results_mabt['bt_10p'].mean()
        summaries.at[d, 'average_mabt_lower'] = results_mabt['mabt'].mean()
        #
        summaries.at[d, 'DeLong_0.95_CI_inclusion'] =\
            np.sum(results_mabt['DeLong'] <= holdOut_results['winners_valid']) / results_mabt.notna()['DeLong'].sum()
        summaries.at[d, 'HanleyMcNeil_0.95_CI_inclusion'] =\
            np.sum(results_mabt['Hanley_McNeil'] <= holdOut_results['winners_valid']) /\
            results_mabt.notna()['Hanley_McNeil'].sum()
        summaries.at[d, 'bt_0.95_CI_inclusion'] =\
            np.sum(results_mabt['bt'] <= holdOut_results['winners_valid']) / results_mabt.notna()['bt'].sum()
        summaries.at[d, 'DeLong10p_0.95_CI_inclusion'] =\
            np.sum(results_mabt['DeLong_10p'] <= holdOut_results['winners_eval']) /\
            results_mabt.notna()['DeLong_10p'].sum()
        summaries.at[d, 'HanleyMcNeil10p_0.95_CI_inclusion'] =\
            np.sum(results_mabt['Hanley_McNeil_10p'] <= holdOut_results['winners_eval']) /\
            results_mabt.notna()['Hanley_McNeil_10p'].sum()
        summaries.at[d, 'bt10p_0.95_CI_inclusion'] =\
            np.sum(results_mabt['bt_10p'] <= holdOut_results['winners_valid']) / results_mabt.notna()['bt_10p'].sum()
        summaries.at[d, 'mabt_0.95_CI_inclusion'] =\
            np.sum(results_mabt['mabt'] <= holdOut_results['winners_valid']) / results_mabt.notna()['mabt'].sum()
    #
    if np.isnan(results_BBC_pooled).sum() == 0:
        summaries.at[d, 'pooled_uniformity'] = percentile_uniformity(results_BBC_pooled, holdOut_results['BBC_winner'],
                                                                     alpha_=0.05, plot_baselines=True, plot_result=True)
    if np.isnan(results_BBC_fold).sum() == 0:
        summaries.at[d, 'fold_uniformity'] = percentile_uniformity(results_BBC_fold, holdOut_results['BBC_winner'],
                                                                   alpha_=0.05, plot_baselines=False, plot_result=True)
        plt.title(real_datasets[d])
        plt.legend(['Uniform', 'CI line', 'CI line', 'Pooled', 'Fold'])
        plot_pdf_saver.savefig(new_plot)
        plt.clf()
    print('Dataset ' + str(d + 1) + ' of ' + str(len(real_datasets)))
plot_pdf_saver.close()
plt.close()
# Save summaries
# noinspection PyTypeChecker
summaries.to_csv('../real_datasets_summaries.csv', index=False)

# Missing cases for mabt and co.
missing_summaries = pd.DataFrame(columns=['dataset', 'DeLong', 'HanleyMcNeil', 'bt', 'DeLong10p', 'HanleyMcNeil10p',
                                          'bt10p', 'mabt'])
for d in range(len(real_datasets)):
    missing_summaries.at[d, 'dataset'] = real_datasets[d]
    result_columns = ['DeLong', 'HanleyMcNeil', 'bt', 'DeLong10p', 'HanleyMcNeil10p', 'bt10p', 'mabt']
    if os.path.exists('../multiplicity-adjusted_bootstrap_tilting/real_datasets_results/' + real_datasets[d] + '.csv'):
        results = pd.read_csv('../multiplicity-adjusted_bootstrap_tilting/real_datasets_results/' +
                              real_datasets[d] + '.csv')
        missing_cases = results[['DeLong', 'Hanley_McNeil', 'bt', 'DeLong_10p',
                                 'Hanley_McNeil_10p', 'bt_10p', 'mabt']].isna().sum()
        missing_summaries.at[d, result_columns] = [int(m) for m in missing_cases]
    else:
        missing_summaries.at[d, result_columns] = np.nan

# noinspection PyTypeChecker
missing_summaries.to_csv('../multiplicity-adjusted_bootstrap_tilting/real_datasets_missing cases.csv', index=False)
