# Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# Checking uniformity of percentiles of true values on BBC distributions
def percentile_uniformity(bootstrap_distributions, theoretical_values):
    # Calculate percentiles
    n_runs = len(theoretical_values)
    percentiles = [np.mean(bootstrap_distributions[i, :] <= theoretical_values[i]) for i in range(n_runs)]
    percentiles = sorted(percentiles)
    # True uniforms for confidence intervals
    uniforms = np.zeros((10000, n_runs))
    for i in range(uniforms.shape[0]):
        uniforms[i, :] = sorted(np.random.uniform(0, 1, n_runs))
    uniforms_upper = np.repeat(0.0, n_runs)
    uniforms_lower = np.repeat(0.0, n_runs)
    for j in range(uniforms.shape[1]):
        uniforms_upper[j] = sorted(uniforms[:, j])[int(0.005 * uniforms.shape[0])]
        uniforms_lower[j] = sorted(uniforms[:, j])[int(0.995 * uniforms.shape[0])]
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
