import numpy as np
import pandas as pd
import time
from BBC_parallel import bbc
from generate_data import get_data
from scipy import stats
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score
import matplotlib.pyplot as plt


def percentile_uniformity(bootstrap_distributions, theoretical_values, alpha_=0.05):
    """
    :param alpha_:
    :param bootstrap_distributions: output of bcc. #n runs x bcc iter
    :param theoretical_values: #performance
    :return:
    """
    # Calculate percentiles
    n_runs = len(theoretical_values)
    percentiles = [np.mean(bootstrap_distributions[p, :] <= theoretical_values[p]) for p in range(n_runs)]
    percentiles = sorted(percentiles)

    # True uniforms for confidence intervals
    uniforms = np.zeros((10000, n_runs))
    for u in range(uniforms.shape[0]):
        uniforms[u, :] = sorted(np.random.uniform(0, 1, n_runs))
    uniforms_upper = np.repeat(0.0, n_runs)
    uniforms_lower = np.repeat(0.0, n_runs)
    for j in range(uniforms.shape[1]):
        uniforms_upper[j] = sorted(uniforms[:, j])[int(alpha_/2 * uniforms.shape[0])]
        uniforms_lower[j] = sorted(uniforms[:, j])[int(1-alpha_/2 * uniforms.shape[0])]

    # Plot
    plt.plot([0, 1], [0, 1], c='grey')
    plt.plot(uniforms_upper, (np.arange(len(uniforms_upper)) + 1)/len(uniforms_upper), ':', c='grey')
    plt.plot(uniforms_lower, (np.arange(len(uniforms_upper)) + 1)/len(uniforms_upper), ':', c='grey')
    plt.plot(percentiles, (np.arange(n_runs) + 1)/n_runs)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('percentiles')
    plt.ylabel('CDF')

    # Statistical test p-value (H0: Underlying distribution is uniform)
    return stats.kstest(percentiles, stats.uniform(loc=0.0, scale=1).cdf)[1]


if __name__ == "__main__":

    alpha = 54
    beta = 6
    samples = 500
    config = 20
    balance = 0.5
    type_ = 'classification'
    bbc_type = 'fold'
    bbc_iter = 100
    CI_iter = 1000

    bb = []
    theoretical = []
    for i in tqdm(range(CI_iter)):
        predictions_table, outcome, folds, performances = get_data(alpha, beta, config, samples, folds_=5,
                                                                   balance_=balance, type_=type_)
        bbc_dist, winner_configuration = bbc(predictions_table, outcome, type_, folds, bbc_type=bbc_type,
                                             iterations=bbc_iter)
        theoretical.append(performances[winner_configuration])
        bb.append(bbc_dist)
    bb_ = np.array(bb)

    p_score = percentile_uniformity(bb_, theoretical, 0.005)
    print(p_score)
    plt.show()

    # alphas = (9, 14, 24, 54)
    # for a in alphas:
    #     b = np.random.beta(a, 6, 1000)
    #     print(a, b.mean(), b.std()**2)
