from BBC_parallel import *
from generate_data import *
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt


def percentile_uniformity(bootstrap_distributions, theoretical_values, alpha_=0.05, plot_baselines=True,
                          plot_result=True):
    """
    :param plot_baselines:
    :param plot_result:
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
    if plot_baselines:
        plt.plot([0, 1], [0, 1], c='grey')
        plt.plot(uniforms_upper, (np.arange(len(uniforms_upper)) + 1)/len(uniforms_upper), ':', c='grey')
        plt.plot(uniforms_lower, (np.arange(len(uniforms_upper)) + 1)/len(uniforms_upper), ':', c='grey')
    if plot_result:
        plt.plot(percentiles, (np.arange(n_runs) + 1)/n_runs)
    if plot_baselines:
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
        predictions_table, outcome, folds, performances = get_data(alpha, beta, balance, config, samples, folds_=5,
                                                                   type_=type_)
        bbc_dist, winner_configuration = bbc(predictions_table, outcome, type_, folds, bbc_type=bbc_type,
                                             iterations=bbc_iter)
        if type_ == 'multiclass':
            outcome_unique = np.unique(outcome)
            theoretical.append((np.sum(performances[winner_configuration]) -
                                np.sum(np.diagonal(performances[winner_configuration]))) /\
                               len(outcome_unique) / (len(outcome_unique)-1))
        else:
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
