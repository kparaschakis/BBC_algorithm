# Libraries and functions
from generate_data import *
import pandas as pd
import os

# Simulation parameters
alpha_beta_list = [[9, 6], [24, 6]]
samples_list = [50, 500]
config_list = [100, 500]
balance_list = ['equal', 'imbalanced']
classes_list = [2, 10]
CI_iter = 200
# Set seed
np.random.seed(4321)
# Loop
for alpha_beta in alpha_beta_list:
    for samples in samples_list:
        for config in config_list:
            for balance_ in balance_list:
                for classes in classes_list:
                    alpha, beta = alpha_beta
                    folder_name = 'alpha_' + str(alpha) + '_beta_' + str(beta) + '_samples_' + str(samples) +\
                                  '_config_' + str(config) + '_balance_' + balance_ + '_classes_' + str(classes)
                    os.makedirs('../simulated_data_for_comparison/' + folder_name)
                    for i in range(CI_iter):
                        balance = np.repeat(1/classes, classes) if balance_ == 'equal' else\
                            [0.05] + list(np.repeat((1 - 0.05)/(classes - 1), classes - 1))
                        n_folds = min(10, int(samples * min(balance)))
                        predictions_table, outcome, folds, performances =\
                            get_data(alpha, beta, balance, config, samples, folds_=n_folds, type_='multiclass')
                        if classes == 2:
                            predictions_table = predictions_table[:, :, 1]
                            performances = performances[:, 1, 0]
                        # noinspection PyTypeChecker
                        pd.DataFrame(outcome).to_csv('../simulated_data_for_comparison/' + folder_name +
                                                     '/outcome_' + str(i) + '.csv', index=False)
                        # noinspection PyTypeChecker
                        pd.DataFrame(folds).to_csv('../simulated_data_for_comparison/' + folder_name +
                                                   '/folds_' + str(i) + '.csv', index=False)
                        if classes == 2:
                            # noinspection PyTypeChecker
                            pd.DataFrame(predictions_table).to_csv('../simulated_data_for_comparison/' + folder_name +
                                                                   '/predictions_' + str(i) + '.csv', index=False)
                            # noinspection PyTypeChecker
                            pd.DataFrame(performances).to_csv('../simulated_data_for_comparison/' + folder_name +
                                                              '/performances_' + str(i) + '.csv', index=False)
                        else:
                            for k in range(predictions_table.shape[2]):
                                # noinspection PyTypeChecker
                                pd.DataFrame(predictions_table[:, :, k]).to_csv('../simulated_data_for_comparison/' +
                                                                                folder_name + '/predictions_' + str(i) +
                                                                                '_' + str(k) + '.csv', index=False)
                                # noinspection PyTypeChecker
                                pd.DataFrame(performances[:, k, :]).to_csv('../simulated_data_for_comparison/' +
                                                                           folder_name + '/performances_' + str(i) +
                                                                           '_' + str(k) + '.csv', index=False)
