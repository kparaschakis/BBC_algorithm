import os
import itertools
import json

from time import time

import numpy as np

from BBC_parallel import bbc
from generate_data import get_data

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

#plt.style.use('seaborn-paper')

plt.rc('axes', axisbelow=True)
#plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('font',**{'family':'serif','serif':['Times'], 'size': 11})
plt.rc('text', usetex=True)

def single_run(config_dict):
    result_dict = {}
    # Get data
    predictions_table, outcome, folds, performances = get_data(24, 6, 0.5, config_dict['cfgs'],
                                                               config_dict['samples'], folds_=config_dict['n_folds'],
                                                               type_='classification')
    for bbc_type in ('pooled', 'fold'):
        time_list = []
        for i in range(config_dict['tot_iter']):
            start_time = time()
            bbc_dist, winner_configuration = bbc(predictions_table, outcome, 'classification', folds, bbc_type=bbc_type,
                                                 iterations=config_dict['bbc_iter'], n_jobs=config_dict['n_jobs'])
            end_time = time() - start_time
            time_list.append(end_time)
        result_dict[bbc_type] = time_list
        print(f"BBC: {bbc_type} - time avg: {np.mean(time_list):.4f} +/- {np.std(time_list):.4f} ({config_dict['tot_iter']} iters. Total time: {np.sum(time_list)})")

    speedup = np.mean(result_dict['pooled']) / np.mean(result_dict['fold'])
    print(f"Avg. Speedup FBBC: {speedup:.3f} x")
    return result_dict

if __name__ == "__main__":
    np.random.seed(42)

    # HIGH LEVELS
    N = (50, 100, 250, 500, 1000, 5000, 10000) # Sample size
    C = (5, 10, 20, 50, 100, 250) # configs
    F = (3, 5, 10, 20, 30, 50) # folds

    mapper = {"Data Sample": 'samples',
              "Configurations": 'cfgs',
              "Folds": 'n_folds'}

    # Configuration
    config_dict = {'samples': 500,
                   'cfgs': 5,
                   'n_folds': 3,
                   'bbc_iter': 200,
                   'tot_iter': 100,
                   'n_jobs': 1}
    print('Default cfg:', config_dict)

    for exp_type, exp_values in zip(("Data Sample", "Configurations", "Folds"), (N, C, F)):
    #for exp_type, exp_values in zip(("Data Sample",), (N, )):

        print(f" {exp_type} experiment starting...")
        pool, fold = [],[]
        exp_results = {}

        for i, x in enumerate(exp_values):
            print(f"\t{exp_type}: {x}")
            exp_results[i] = config_dict.copy()
            exp_results[i][mapper[exp_type]] = x
            results = single_run(exp_results[i])
            exp_results[i]['results'] = results
            pool.append(results['pooled'])
            fold.append(results['fold'])
            print('\n')

        # Plot
        pool_val = np.median(pool, axis=1)
        pool_err = np.vstack([pool_val-np.percentile(pool, 25, axis=1), np.percentile(pool, 75, axis=1) - pool_val])
        fold_val = np.median(fold, axis=1)
        fold_err = np.vstack([fold_val - np.percentile(fold, 25, axis=1), np.percentile(fold, 75, axis=1) - fold_val])

        fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))
        ax.errorbar(exp_values, pool_val, yerr=pool_err, capsize=5,
                    linestyle='--', marker='o', label='BBC')
        ax.errorbar(exp_values, fold_val, yerr=fold_err, capsize=5,
                    linestyle='-.', marker='x', label='FBBC')
        ax.set_xlabel(f'{exp_type}')
        ax.set_ylabel('Run time (s)')
        ax.grid(True)
        ax.set_yscale("log")
        plt.legend()
        fig.tight_layout()
        plt.show()
        fig.savefig(f'../results/{exp_type}_LOG.pdf', dpi=250, bbox_inches='tight')

        with open(f'../results/time_{exp_type}.json', 'w') as fp:
            json.dump(exp_results, fp, indent=0)
        print(f"json saved in results folder")

    print('done!')

    plt.close('all')

