from joblib import Parallel, delayed
from fastauc.fast_auc import *


def corrcoef2(V1, V2):
    return np.corrcoef(V1, V2)[0, 1] ** 2


def auc_multiclass(outcome, predictions, averaged=True):
    auc = CppAuc()
    outcome_unique = np.unique(outcome)
    performance_vector = 0.5 * np.ones((len(outcome_unique), len(outcome_unique)))
    for out_1 in range(len(outcome_unique)):
        for out_2 in range(len(outcome_unique)):
            if out_1 != out_2:
                performance_vector[out_1, out_2] =\
                    auc.roc_auc_score(outcome[np.in1d(outcome, [out_1, out_2])] == out_1,
                                      predictions[np.in1d(outcome, [out_1, out_2]), out_1].astype(np.float32))
    if averaged:
        performance_vector = (np.sum(performance_vector) - np.sum(np.diagonal(performance_vector))) /\
                             len(outcome_unique) / (len(outcome_unique)-1)
    return performance_vector


def bbc_pooled(args):
    labels, oos_matrix, N, C, metric_func, analysis_type = args
    in_bag_indices = sorted(np.random.choice(N, N, replace=True))
    out_of_bag_indices = list(set(list(range(N))) - set(in_bag_indices))
    if analysis_type in ['classification', 'multiclass']:
        while (len(np.unique(labels)) > len(np.unique(labels[in_bag_indices]))) |\
                (len(np.unique(labels)) > len(np.unique(labels[out_of_bag_indices]))):
            in_bag_indices = sorted(np.random.choice(N, N, replace=True))
            out_of_bag_indices = list(set(list(range(N))) - set(in_bag_indices))
    if analysis_type in ['multiclass', 'regression']:
        in_bag_performances = [metric_func(labels[in_bag_indices], oos_matrix[in_bag_indices, j]) for j in range(C)]
    else:
        in_bag_performances = [metric_func(labels[in_bag_indices].astype(bool),
                                           oos_matrix[in_bag_indices, j].astype(np.float32)) for j in range(C)]
    winner_configuration = np.argmax(in_bag_performances)
    if analysis_type in ['multiclass', 'regression']:
        out_of_bag_performance = metric_func(labels[out_of_bag_indices],
                                             oos_matrix[out_of_bag_indices, winner_configuration])
    else:
        out_of_bag_performance = metric_func(labels[out_of_bag_indices].astype(bool),
                                             oos_matrix[out_of_bag_indices, winner_configuration].astype(np.float32))
    return out_of_bag_performance


def bbc_fold(args):
    performance_matrix, F = args
    in_bag_indices = sorted(np.random.choice(F, F, replace=True))
    while len(in_bag_indices) == len(set(in_bag_indices)):
        in_bag_indices = sorted(np.random.choice(F, F, replace=True))
    out_of_bag_indices = list(set(list(range(F))) - set(in_bag_indices))
    winner_configuration = np.argmax(np.mean(performance_matrix[in_bag_indices, :], axis=0))
    out_of_bag_performances = np.mean(performance_matrix[out_of_bag_indices, winner_configuration])
    return out_of_bag_performances


def bbc(oos_matrix, labels, analysis_type, folds, bbc_type='pooled', iterations=1000, n_jobs=-1):
    auc = CppAuc()
    assert bbc_type in ('pooled', 'fold')
    metric_func = auc.roc_auc_score if analysis_type == 'classification'\
        else corrcoef2 if analysis_type == 'regression' else auc_multiclass

    N = len(labels)  # number of samples
    C = oos_matrix.shape[1]

    F = len(np.unique(folds))
    performance_matrix = np.zeros((F, C))
    for f in range(F):
        for c in range(C):
            if analysis_type in ['multiclass', 'regression']:
                performance_matrix[f, c] = metric_func(labels[folds == f], oos_matrix[folds == f, c])
            else:
                performance_matrix[f, c] = metric_func(labels[folds == f].astype(bool),
                                                       oos_matrix[folds == f, c].astype(np.float32))
    winner_configuration = np.argmax(np.mean(performance_matrix, axis=0))

    bbc_distribution = None
    if bbc_type == 'pooled':
        bbc_distribution = Parallel(prefer="threads", n_jobs=n_jobs)(
            delayed(bbc_pooled)(
                (labels, oos_matrix, N, C, metric_func, analysis_type)
            ) for _ in range(iterations)
        )

    elif bbc_type == 'fold':
        bbc_distribution = Parallel(n_jobs=n_jobs)(
            delayed(bbc_fold)(
                (performance_matrix, F)
            ) for _ in range(iterations)
        )

    return np.array(bbc_distribution), winner_configuration
