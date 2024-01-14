# Libraries
import arff
import os
import numpy as np
import pandas as pd

# Pandas dataset display options
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 10)
# Parameters
train_n_list = [50, 500]
max_dataset_length = 50000
n_splits = 100
# Read dataset names
dataset_list = os.listdir('../real_datasets/')
# Table of datasets with info
splits_table = pd.DataFrame(columns=['dataset', 'classes', 'n', 'minority', 'train_n'])
splits_table['dataset'] = dataset_list
for d in range(len(dataset_list)):
    dataset = arff.load(open('../real_datasets/' + dataset_list[d]))
    data = pd.DataFrame(dataset['data'])
    data.columns = [dataset['attributes'][a][0].lower() for a in range(len(dataset['attributes']))]
    splits_table.at[d, 'classes'] = len(data['class'].unique())
    splits_table.at[d, 'n'] = len(data)
    splits_table.at[d, 'minority'] = data['class'].value_counts().min()
    print('Dataset', d+1, 'of', len(dataset_list))
# Get rid of too large datasets
splits_table.drop(splits_table.index[splits_table['n'] > max_dataset_length], inplace=True)
# Sort by n
splits_table.sort_values('n', inplace=True)
# Decide training data size
train_n_repeats = int(np.ceil(len(splits_table)/len(train_n_list)))
train_n_vector = np.repeat(train_n_list, repeats=train_n_repeats)[:len(splits_table)]
splits_table['train_n'] = train_n_vector
splits_table['minority_in_train'] = splits_table['minority'] * splits_table['train_n'] / splits_table['n']
# Get rid of datasets with less than 3 minority in training
splits_table.drop(splits_table.index[splits_table['minority_in_train'] < 3], inplace=True)
# Set seed
np.random.seed(5678)
# Split indices per dataset
for d in splits_table.index:
    dataset_name = splits_table.loc[d, 'dataset']
    dataset = arff.load(open('../real_datasets/' + dataset_name))
    data = pd.DataFrame(dataset['data'])
    data.columns = [dataset['attributes'][a][0].lower() for a in range(len(dataset['attributes']))]
    # Needs to be stratified!
    for s in range(n_splits):
        ...
