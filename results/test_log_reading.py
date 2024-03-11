import json
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

plt.switch_backend("Qt5Agg")

for fname, exp in zip(('Configurations', 'Data Sample', 'Folds'), ('cfgs', 'samples', 'n_folds')):

    with open(f'time_{fname}.json') as f:
        d = json.load(f)

    exp_values = [x[exp] for x in d.values()]
    pool = [x['results']['pooled'] for x in d.values()]
    fold = [x['results']['fold'] for x in d.values()]

    # Plot
    pool_val = np.median(pool, axis=1)
    pool_err = np.vstack([pool_val - np.percentile(pool, 25, axis=1), np.percentile(pool, 75, axis=1) - pool_val])
    fold_val = np.median(fold, axis=1)
    fold_err = np.vstack([fold_val - np.percentile(fold, 25, axis=1), np.percentile(fold, 75, axis=1) - fold_val])

    fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))
    ax.errorbar(exp_values, pool_val, yerr=pool_err, capsize=5,
                linestyle='--', marker='o', label='BBC')
    ax.errorbar(exp_values, fold_val, yerr=fold_err, capsize=5,
                linestyle='-.', marker='x', label='FBBC')
    ax.set_ylabel('Run time (s)')
    ax.set_xlabel(f'{fname}')
    ax.grid(True)
    ax.set_yscale("log")
    plt.legend()
    fig.tight_layout()
    fig.savefig(f'{fname}_LOG.pdf', dpi=250, bbox_inches='tight')

    # fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))
    # ax.errorbar(exp_values, pool_val, yerr=pool_err, capsize=5,
    #             linestyle='--', marker='o', label='BBC')
    # ax.errorbar(exp_values, fold_val, yerr=fold_err, capsize=5,
    #             linestyle='-.', marker='x', label='FBBC')
    # ax.set_ylabel('Run time (s)')
    # ax.set_xlabel(f'{fname}')
    # ax.grid(True)
    # plt.legend()
    # fig.tight_layout()
plt.show(block=True)