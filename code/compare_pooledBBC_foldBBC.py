# Libraries
from BBC_parallel import *
from generate_data import *
from compute_CI_test import percentile_uniformity
from tqdm import tqdm
import matplotlib.pyplot as plt

alpha = 14
beta = 6
samples = 200
config = 50
balance = [1/3, 1/3, 1/3]
type_ = 'multiclass'
bbc_types = ['pooled', 'fold']
bbc_iter = 200
CI_iter = 200

bb = {}
theoretical = []
for bbc_type in bbc_types:
    bb[bbc_type] = []
for i in tqdm(range(CI_iter)):
    predictions_table, outcome, folds, performances = get_data(alpha, beta, balance, config, samples, folds_=5,
                                                               type_=type_)
    for bbc_type in bbc_types:
        bbc_dist, winner_configuration = bbc(predictions_table, outcome, type_, folds, bbc_type=bbc_type,
                                             iterations=bbc_iter)
        bb[bbc_type].append(bbc_dist)
        if bbc_type == bbc_types[0]:
            if type_ == 'multiclass':
                outcome_unique = np.unique(outcome)
                theoretical.append((np.sum(performances[winner_configuration]) -
                                    np.sum(np.diagonal(performances[winner_configuration]))) /\
                                   len(outcome_unique) / (len(outcome_unique)-1))
            else:
                theoretical.append(performances[winner_configuration])
bb_ = {}
for bbc_type in bbc_types:
    bb_[bbc_type] = np.array(bb[bbc_type])
for bbc_type in bbc_types:
    if bbc_type == bbc_types[0]:
        p_score = percentile_uniformity(bb_[bbc_type], theoretical, 0.01)
    else:
        p_score = percentile_uniformity(bb_[bbc_type], theoretical, 0.01, False)
    print(bbc_type, ' ', p_score)
plt.legend(['Uniform', 'Upper', 'Lower'] + bbc_types)

print('Pooled BBC distribution avg. std:', np.mean(np.std(bb_['pooled'], axis=1)))
print('Fold BBC distribution avg. std:', np.mean(np.std(bb_['fold'], axis=1)))
