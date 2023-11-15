import numpy as np
from scipy.stats import norm


def get_data(beta_a_, beta_b_, configurations_=100, samples_=1000, folds_=5, balance_=0.5, type_='classification'):
    # samples_: sample size
    # configurations_: number of configurations
    # beta_a_: parameter 1 of performance distribution (beta) of the configurations
    # beta_b_: parameter 2 of performance distribution (beta) of the configurations
    # folds_: number of folds
    # balance_: outcome balance

    assert type_ in ['classification', 'regression']
    configuration_performances = np.random.beta(beta_a_, beta_b_, configurations_)

    if type_ == 'classification':
        # Classification: It generates the true performances (auc) for each configuration from a beta distribution and
        # then the prediction (scores, i.e. no probabilities) that correspond to each configuration.
        # Only binary

        configuration_means = np.sqrt(2) * norm.ppf(configuration_performances)
        # Simulate predictions (scores) table
        samples_0 = int(round(balance_ * samples_))
        samples_1 = samples_ - samples_0
        outcome = np.concatenate([np.repeat(0, samples_0), np.repeat(1, samples_1)])
        predictions_table_1 = np.zeros((samples_1, configurations_))
        predictions_table_0 = np.zeros((samples_0, configurations_))
        for c in range(configurations_):
            predictions_table_1[:, c] = np.random.normal(configuration_means[c], 1, samples_1)
            predictions_table_0[:, c] = np.random.normal(0, 1, samples_0)
        predictions_table = np.concatenate([predictions_table_0, predictions_table_1], axis=0)

    elif type_ == 'regression':
        # Regression: It generates the true performances (R squared) for each configuration from a beta distribution
        # and then the predictions* that correspond to each configuration.
        #
        # * Note that the simulated "predictions" are no actual predictions, but rather predictor arrays with a certain
        # correlation with the outcome/label, but that is equivalent to a prediction: If we fit a simple linear
        # regression model outcome ~ predictor we can end up with the actual predictions. But there is no need to do so
        # (for the sake of saving time); we can instead use the data correlation between predictor and outcome, which
        # would be equivalent to the R squared of the above linear model.

        configuration_noise_sigma = np.sqrt(1 / configuration_performances - 1)
        # Simulate predictions table
        outcome = np.random.normal(0, 1, samples_)
        predictions_table = np.zeros((samples_, configurations_))
        for c in range(configurations_):
            predictions_table[:, c] = outcome + np.random.normal(0, configuration_noise_sigma[c], samples_)

    else:
        raise NotImplementedError

    # Define folds
    folds = np.tile(range(folds_), int(np.ceil(samples_ / folds_)))
    folds = folds[:len(outcome)]

    return predictions_table, outcome, folds, configuration_performances
