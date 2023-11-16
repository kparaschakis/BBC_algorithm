import numpy as np
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed


def corrcoef2(V1, V2):
    return np.corrcoef(V1, V2)[0, 1] ** 2


def bbc_pooled(args):
    labels, oos_matrix, N, C, metric_func = args
    # metric_func = roc_auc_score if metric_func_name == 'roc_auc_score' else r2_score
    in_bag_indices = sorted(np.random.choice(N, N, replace=True))
    out_of_bag_indices = list(set(list(range(N))) - set(in_bag_indices))
    in_bag_performances = [metric_func(labels[in_bag_indices], oos_matrix[in_bag_indices, j]) for j in range(C)]
    winner_configuration = np.argmax(in_bag_performances)
    out_of_bag_performance = metric_func(labels[out_of_bag_indices],
                                         oos_matrix[out_of_bag_indices, winner_configuration])
    return out_of_bag_performance


def bbc_averaged(args):
    labels, oos_matrix, N, C, metric_func, analysis_type, fold_ids, folds = args
    in_bag_indices = sorted(np.random.choice(N, N, replace=True))
    out_of_bag_indices = list(set(list(range(N))) - set(in_bag_indices))
    in_bag_performances = []
    for j in range(C):
        in_bag_fold_performances = []
        for f in fold_ids:
            index_selection = [ib for ib in in_bag_indices if folds[ib] == f]
            if ((analysis_type == 'regression') |
                ((analysis_type == 'classification') & (len(np.unique(labels[index_selection])) > 1))) & \
                    (len(index_selection) > 1):
                in_bag_fold_performances.append(metric_func(labels[index_selection], oos_matrix[index_selection, j]))
        in_bag_performances.append(np.mean(in_bag_fold_performances))
    winner_configuration = np.argmax(in_bag_performances)
    out_of_bag_fold_performances = []
    for f in fold_ids:
        index_selection = [ib for ib in out_of_bag_indices if folds[ib] == f]
        if ((analysis_type == 'regression') |
            ((analysis_type == 'classification') & (len(np.unique(labels[index_selection])) > 1))) & \
                (len(index_selection) > 1):
            out_of_bag_fold_performances.append(metric_func(labels[index_selection],
                                                            oos_matrix[index_selection, winner_configuration]))
    out_of_bag_performances = np.mean(out_of_bag_fold_performances)
    return out_of_bag_performances


def bbc_fold(args):
    performance_matrix, F = args
    in_bag_indices = sorted(np.random.choice(F, F, replace=True))
    while len(in_bag_indices) == len(set(in_bag_indices)):
        in_bag_indices = sorted(np.random.choice(F, F, replace=True))
    out_of_bag_indices = list(set(list(range(F))) - set(in_bag_indices))
    winner_configuration = np.argmax(np.mean(performance_matrix[in_bag_indices, :], axis=0))
    out_of_bag_performances = np.mean(performance_matrix[out_of_bag_indices, winner_configuration])
    return out_of_bag_performances


def bbc(oos_matrix, labels, analysis_type, folds, bbc_type='pooled', iterations=1000):
    assert bbc_type in ('pooled', 'averaged', 'fold')
    metric_func = roc_auc_score if analysis_type == 'classification' else corrcoef2

    N = len(labels)  # number of samples
    C = oos_matrix.shape[1]

    F = len(np.unique(folds))
    performance_matrix = np.zeros((F, C))
    for f in range(F):
        for c in range(C):
            performance_matrix[f, c] = metric_func(labels[folds == f], oos_matrix[folds == f, c])
    winner_configuration = np.argmax(np.mean(performance_matrix, axis=0))

    bbc_distribution = None
    if bbc_type == 'pooled':
        bbc_distribution = Parallel(n_jobs=-1)(
            delayed(bbc_pooled)(
                (labels, oos_matrix, N, C, metric_func)
            ) for _ in range(iterations)
        )

    elif bbc_type == 'averaged':
        fold_ids = np.unique(folds)
        bbc_distribution = Parallel(n_jobs=-1)(
            delayed(bbc_averaged)(
                (labels, oos_matrix, N, C, metric_func, analysis_type, fold_ids, folds)
            ) for _ in range(iterations)
        )

    elif bbc_type == 'fold':
        bbc_distribution = Parallel(n_jobs=-1)(
            delayed(bbc_fold)(
                (performance_matrix, F)
            ) for _ in range(iterations)
        )
        # out_of_bag_performances = []
        # F = len(np.unique(folds))
        # performance_matrix = np.zeros((F, C))
        # for f in range(F):
        #     for c in range(C):
        #         performance_matrix[f, c] = metric_func(labels[folds == f], oos_matrix[folds == f, c])
        # for i in range(iterations):
        #     in_bag_indices = sorted(np.random.choice(F, F, replace=True))
        #     while len(in_bag_indices) == len(set(in_bag_indices)):
        #         in_bag_indices = sorted(np.random.choice(F, F, replace=True))
        #     out_of_bag_indices = list(set(list(range(F))) - set(in_bag_indices))
        #     winner_configuration = np.argmax(np.mean(performance_matrix[in_bag_indices, :], axis=0))
        #     out_of_bag_performances.append(np.mean(performance_matrix[out_of_bag_indices, winner_configuration]))
        # bbc_distribution = out_of_bag_performances

    return np.array(bbc_distribution), winner_configuration
