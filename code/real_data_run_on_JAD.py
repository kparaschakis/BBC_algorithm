# Libraries
import os
import pandas as pd
import arff
from real_data_read_JAD_output import *
from jadbio_internal.ml import analysis_form
from jadbio_internal.api_client import ApiClient
from jadbio_internal.ml.tuning import tuning_params
from jadbio_internal.ml.tuning import cv_params

# Dataset splits
n_splits = 100
# JAD repeats (needs to cover max_folds below)
n_repeats = 1
# Find split indices files
split_files_path = '../real_datasets/train_test_split_indices/'
split_files = os.listdir(split_files_path)
# Initialise client
jad = ApiClient('http://139.91.207.48:9999', 'admin@jadbio.com', 'admin')
project = jad.project.find_project('bbc_research')
# Number of JAD cores
cores = 3
# Define analysis
cv_params = cv_params.standard_cv(None, n_repeats).with_best_model_rule().with_disabled_progress_update(). \
    with_disabled_correction()
tuning_grid = tuning_params.auto()
form_grid = analysis_form.classification('class', 'TYPICAL', cores).with_tuning_strategy(tuning_grid). \
    with_cv_params(cv_params).with_plots([])
# Loop over datasets
data_files_path = '../real_datasets/'
for d in range(len(split_files)):
    # Import dataset
    dataset_name = split_files[d].split('.csv')[0]
    dataset = arff.load(open('../real_datasets/' + dataset_name + '.arff'))
    data = pd.DataFrame(dataset['data'])
    data.columns = [dataset['attributes'][a][0].lower() for a in range(len(dataset['attributes']))]
    # Import split indices file
    splits_file = pd.read_csv(split_files_path + split_files[d], index_col=0)
    # Upload train datasets for all splits
    for s in range(n_splits):
        indices = splits_file[str(s)]
        indices_int = [int(indices[i]) for i in range(len(indices))]
        data_to_upload = data.loc[indices_int]
        data_to_upload['class'] = ['c' + data_to_upload['class'][i] for i in data_to_upload.index]
        data_to_upload.to_csv('../temp.csv')
        jad.upload_dataset(project, dataset_name + '_run_' + str(s), '../temp.csv')
        print('Training split', s+1, 'of', n_splits, 'uploaded')
    dataset_list = jad.project.find_project_datasets(project)
    evaluation_grid_list = []
    for dataset_i in range(len(dataset_list)):
        # Read dataset id
        dataset_ID = dataset_list[dataset_i]['id']
        # Stack analyses
        evaluation_grid_list.append(jad.ml.submit_analysis(dataset_id=dataset_ID, form=form_grid))
    # Wait until analyses finish and load results
    evaluation_grid = jad.ml.wait_multiple_testing(evaluation_grid_list)
    # Delete datasets from JAD
    for dataset_i in range(len(dataset_list)):
        dataset_ID = dataset_list[dataset_i]['id']
        jad.delete_dataset(dataset_ID)
    # Save predictions
    for analysis_i in range(len(evaluation_grid_list)):
        analysis_id = evaluation_grid_list[analysis_i]
        dataset_run = evaluation_grid[analysis_id]['datasetInfo']['datasetName']
        outcome, split_indices, descriptions, predictions = read_JAD_output_classification(evaluation_grid[analysis_id])
        pd.DataFrame(outcome).to_csv('../real_datasets/JAD_results/' + dataset_run + '_outcome.csv', index=False)
        pd.DataFrame(split_indices).to_csv('../real_datasets/JAD_results/' + dataset_run + '_splitIndices.csv',
                                           index=False)
        pd.DataFrame(descriptions).to_csv('../real_datasets/JAD_results/' + dataset_run + '_configurations.csv',
                                          index=False)
        if len(predictions.shape) == 2:
            pd.DataFrame(predictions).to_csv('../real_datasets/JAD_results/' + dataset_run + '_predictions.csv',
                                             index=False)
        else:
            for o in range(len(np.unique(outcome))):
                pd.DataFrame(predictions[:, :, o]).to_csv('../real_datasets/JAD_results/' + dataset_run +
                                                          '_predictions_' + str(o) + '.csv', index=False)
