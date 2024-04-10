# Libraries and functions
import pandas as pd
from core.BBC_parallel import *

# Define BBC types
bbc_types = ['pooled', 'fold']
# Number of BBC iterations
bbc_iter = 500
# Read simulation configurations
dataset_list = os.listdir('../real_datasets/JAD_results/')
dataset_list = sorted([d.split('_run_0')[0] for d in dataset_list if '_run_0_outcome' in d])
# Loop over various simulation configurations
for d in range(len(dataset_list)):
    # Read simulation parameters
    dataset = dataset_list[d]
    main_path = '../real_datasets/JAD_results/' + dataset
    n_runs = os.listdir('../real_datasets/JAD_results')
    n_runs = len([r for r in n_runs if dataset in r and 'outcome' in r])
    # Loop over runs
    bb = {}
    winners = []
    for bbc_type in bbc_types:
        bb[bbc_type] = []
    for r in range(n_runs):
        # Import simulated data
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
        # Apply BBC
        data_type = 'classification' if classes == 2 else 'multiclass'
        for bbc_type in bbc_types:
            bbc_dist, winner_configuration = bbc(predictions_table, outcome, data_type, folds, bbc_type=bbc_type,
                                                 iterations=bbc_iter)
            bb[bbc_type].append(bbc_dist)
            if bbc_type == bbc_types[0]:
                winners.append(winner_configuration)
        print('Dataset', d + 1, 'of', len(dataset_list), '/ Run', r + 1, 'of', n_runs)
    for bbc_type in bbc_types:
        # noinspection PyTypeChecker
        pd.DataFrame(bb[bbc_type]).to_csv('../real_datasets_results/' + dataset + '_' + bbc_type + '.csv', index=False)
    # noinspection PyTypeChecker
    pd.DataFrame(winners).to_csv('../real_datasets_results/' + dataset + '_winner_configurations.csv', index=False)
