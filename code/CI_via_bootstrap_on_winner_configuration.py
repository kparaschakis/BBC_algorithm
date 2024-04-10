import os
import pandas as pd
import numpy as np
from core.BBC_parallel import auc_multiclass
from sklearn.metrics import roc_auc_score

# Simulation
runs = 200
simulation_results_table = pd.DataFrame(columns=['setting', 'average_performance', 'inclusion_norm', 'tightness_norm',
                                                 'inclusion_boot', 'tightness_boot'])
settings = os.listdir('../simulated_data_for_comparison/')
np.random.seed(148)
for s in range(len(settings)):
    setting = settings[s]
    simulation_results_table.at[s, 'setting'] = setting
    main_path = '../simulated_data_for_comparison/' + setting + '/'
    alpha, beta, samples, configurations, balance, classes = [setting.split('_')[p] for p in [1, 3, 5, 7, 9, 11]]
    samples = int(samples)
    configurations = int(configurations)
    classes = int(classes)
    lower_bounds_norm = []
    lower_bounds_boot = []
    winner_performances = []
    for r in range(runs):
        folds = pd.read_csv('../simulated_data_for_comparison/' + setting + '/folds_' + str(r) + '.csv')['0'].values
        outcome = pd.read_csv('../simulated_data_for_comparison/' + setting + '/outcome_' + str(r) + '.csv')['0'].values
        if classes > 2:
            performances = np.zeros((configurations, classes, classes))
            for c in range(classes):
                performances[:, c, :] =\
                    pd.read_csv(main_path + 'performances_' + str(r) + '_' + str(c) + '.csv').values
            predictions_table = np.zeros((samples, configurations, classes))
            for c in range(classes):
                predictions_table[:, :, c] =\
                    pd.read_csv(main_path + 'predictions_' + str(r) + '_' + str(c) + '.csv').values
        else:
            performances = pd.read_csv(main_path + 'performances_' + str(r) + '.csv')['0'].values
            predictions_table = pd.read_csv(main_path + 'predictions_' + str(r) + '.csv').values
        #
        metric_func = roc_auc_score if classes == 2 else auc_multiclass
        N = len(outcome)  # number of samples
        C = predictions_table.shape[1]
        F = len(np.unique(folds))
        performance_matrix = np.zeros((F, C))
        for f in range(F):
            for c in range(C):
                performance_matrix[f, c] = metric_func(outcome[folds == f], predictions_table[folds == f, c])
        winner_configuration = np.argmax(np.mean(performance_matrix, axis=0))
        winner_perf = performances[winner_configuration]
        if classes > 2:
            S = (np.sum(winner_perf) - np.sum(np.diagonal(winner_perf))) /\
                winner_perf.shape[0] / (winner_perf.shape[0] - 1)
        else:
            S = winner_perf
        winner_performances.append(S)
        winner_values = performance_matrix[:, winner_configuration]
        lower_bounds_norm.append(np.mean(winner_values) - 1.644854*np.std(winner_values)/np.sqrt(len(winner_values)))
        bootstrap_dist = []
        for i in range(500):
            bootstrap_dist.append(np.mean(np.random.choice(winner_values, len(winner_values), replace=True)))
        lower_bounds_boot.append(np.array(sorted(bootstrap_dist))[int(0.05*len(bootstrap_dist))])
        print('Setting', s+1, 'of', len(settings), ', run', r+1, 'of', runs)
    winner_performances = np.array(winner_performances)
    lower_bounds_norm = np.array(lower_bounds_norm)
    lower_bounds_boot = np.array(lower_bounds_boot)
    simulation_results_table.at[s, 'average_performance'] = np.mean(winner_performances)
    simulation_results_table.at[s, 'inclusion_norm'] = np.mean(winner_performances > lower_bounds_norm)
    simulation_results_table.at[s, 'tightness_norm'] = np.mean(winner_performances) - np.mean(lower_bounds_norm)
    simulation_results_table.at[s, 'inclusion_boot'] = np.mean(winner_performances > lower_bounds_boot)
    simulation_results_table.at[s, 'tightness_boot'] = np.mean(winner_performances) - np.mean(lower_bounds_boot)
# Save results
# noinspection PyTypeChecker
simulation_results_table.to_csv('../simulation_summaries_winners_curse.csv')

# Benchmark datasets
n_runs = 100
benchmark_results_table = pd.DataFrame(columns=['dataset', 'average_performance', 'inclusion_norm', 'tightness_norm',
                                                'inclusion_boot', 'tightness_boot'])
dataset_list = os.listdir('../real_datasets/JAD_results/')
dataset_list = sorted([d.split('_run_0')[0] for d in dataset_list if '_run_0_outcome' in d])
np.random.seed(148)
for d in range(len(dataset_list)):
    dataset = dataset_list[d]
    benchmark_results_table.at[d, 'dataset'] = dataset
    main_path = '../real_datasets/JAD_results/' + dataset
    lower_bounds_norm = []
    lower_bounds_boot = []
    winner_performances = pd.read_csv('../real_datasets_holdOut_performances/' + dataset + '.csv')['BBC_winner'].values
    for r in range(n_runs):
        outcome = pd.read_csv(main_path + '_run_' + str(r) + '_outcome.csv')['0'].values
        split_indices = pd.read_csv(main_path + '_run_' + str(r) + '_splitIndices.csv')
        folds = np.array([split_indices.index[(split_indices == i).sum(axis=1) == 1][0] for i in range(len(outcome))])
        configuration_descriptions = pd.read_csv(main_path + '_run_' + str(r) + '_configurations.csv')
        configurations = len(configuration_descriptions)
        classes = len(np.unique(outcome))
        samples = len(outcome)
        if classes > 2:
            predictions_table = np.zeros((samples, configurations, classes))
            for c in range(classes):
                predictions_table[:, :, c] =\
                    pd.read_csv(main_path + '_run_' + str(r) + '_predictions_' + str(c) + '.csv').values
        else:
            predictions_table = pd.read_csv(main_path + '_run_' + str(r) + '_predictions.csv').values
        #
        metric_func = roc_auc_score if classes == 2 else auc_multiclass
        N = len(outcome)  # number of samples
        C = predictions_table.shape[1]
        F = len(np.unique(folds))
        performance_matrix = np.zeros((F, C))
        for f in range(F):
            for c in range(C):
                performance_matrix[f, c] = metric_func(outcome[folds == f], predictions_table[folds == f, c])
        winner_configuration = np.argmax(np.mean(performance_matrix, axis=0))
        winner_values = performance_matrix[:, winner_configuration]
        lower_bounds_norm.append(np.mean(winner_values) - 1.644854*np.std(winner_values)/np.sqrt(len(winner_values)))
        bootstrap_dist = []
        for i in range(500):
            bootstrap_dist.append(np.mean(np.random.choice(winner_values, len(winner_values), replace=True)))
        lower_bounds_boot.append(np.array(sorted(bootstrap_dist))[int(0.05*len(bootstrap_dist))])
        print('Dataset', d+1, 'of', len(dataset_list), ', run', r+1, 'of', n_runs)
    lower_bounds_norm = np.array(lower_bounds_norm)
    lower_bounds_boot = np.array(lower_bounds_boot)
    benchmark_results_table.at[d, 'average_performance'] = np.mean(winner_performances)
    benchmark_results_table.at[d, 'inclusion_norm'] = np.mean(winner_performances > lower_bounds_norm)
    benchmark_results_table.at[d, 'tightness_norm'] = np.mean(winner_performances) - np.mean(lower_bounds_norm)
    benchmark_results_table.at[d, 'inclusion_boot'] = np.mean(winner_performances > lower_bounds_boot)
    benchmark_results_table.at[d, 'tightness_boot'] = np.mean(winner_performances) - np.mean(lower_bounds_boot)
# Save results
# noinspection PyTypeChecker
benchmark_results_table.to_csv('../real_datasets_summaries_winners_curse.csv')
