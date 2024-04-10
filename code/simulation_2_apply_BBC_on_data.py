# Libraries and functions
import pandas as pd
from core.BBC_parallel import *

# Define BBC types
bbc_types = ['pooled', 'fold']
# Number of BBC iterations
bbc_iter = 500
# Read simulation configurations
sim_configurations = sorted(os.listdir('../simulated_data_for_comparison/'))
# Loop over various simulation configurations
for sc in range(len(sim_configurations)):
    # Read simulation parameters
    sim_conf = sim_configurations[sc]
    main_path = '../simulated_data_for_comparison/' + sim_conf + '/'
    alpha, beta, samples, configurations, balance, classes = [sim_conf.split('_')[p] for p in [1, 3, 5, 7, 9, 11]]
    samples = int(samples)
    configurations = int(configurations)
    classes = int(classes)
    n_runs = os.listdir('../simulated_data_for_comparison/' + sim_conf)
    n_runs = len([r for r in n_runs if 'outcome' in r])
    # Loop over runs
    bb = {}
    theoretical = []
    for bbc_type in bbc_types:
        bb[bbc_type] = []
    for r in range(n_runs):  # QUESTION: CAN WE PARALLELIZE THIS???
        # Import simulated data
        folds = pd.read_csv(main_path + 'folds_' + str(r) + '.csv')['0'].values
        outcome = pd.read_csv(main_path + 'outcome_' + str(r) + '.csv')['0'].values
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
        # Apply BBC
        data_type = 'classification' if classes == 2 else 'multiclass'
        for bbc_type in bbc_types:
            bbc_dist, winner_configuration = bbc(predictions_table, outcome, data_type, folds, bbc_type=bbc_type,
                                                 iterations=bbc_iter)
            bb[bbc_type].append(bbc_dist)
            if bbc_type == bbc_types[0]:
                if data_type == 'multiclass':
                    outcome_unique = np.unique(outcome)
                    theoretical.append((np.sum(performances[winner_configuration]) -
                                        np.sum(np.diagonal(performances[winner_configuration]))) /\
                                       len(outcome_unique) / (len(outcome_unique)-1))
                else:
                    theoretical.append(performances[winner_configuration])
        print('Setting', sc + 1, 'of', len(sim_configurations), '/ Run', r + 1, 'of', n_runs)
    for bbc_type in bbc_types:
        # noinspection PyTypeChecker
        pd.DataFrame(bb[bbc_type]).to_csv('../simulated_data_results/' + sim_conf + '_' + bbc_type + '.csv',
                                          index=False)
    # noinspection PyTypeChecker
    pd.DataFrame(theoretical).to_csv('../simulated_data_results/' + sim_conf + '_theoretical.csv', index=False)
