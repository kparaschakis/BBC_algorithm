# Libraries
import pandas as pd
import arff
from io import StringIO
from fastauc.fast_auc import *
from make_API_configuration import *
#
from jadbio_internal.ml import analysis_form
from jadbio_internal.api_client import ApiClient
from jadbio_internal.ml.tuning import tuning_params
from jadbio_internal.ml.tuning import cv_params

# Fast auc
auc = CppAuc()
# JAD client
jad = ApiClient('http://139.91.207.48:9999', 'admin@jadbio.com', 'admin')
project = jad.project.find_project('bbc_research')
# Parameters
n_repeats = 1
cores = 16
# Define analysis CV
cv_args = cv_params.standard_cv(None, n_repeats).with_best_model_rule().with_disabled_progress_update()
# Files list
datasets_list = os.listdir('../real_datasets_results/')
datasets_list = [f.split('_winner')[0] for f in datasets_list if 'winner' in f]
# Loop over files
for f in range(len(datasets_list)):
    dataset = datasets_list[f]
    file_name = dataset + '_winner_configurations.csv'
    winners_file = pd.read_csv('../real_datasets_results/' + file_name)
    winners_file.columns = ['BBC_winner']
    # noinspection PyTypeChecker
    mabt_file_name = '../multiplicity-adjusted_bootstrap_tilting/real_datasets_results/' + dataset + '.csv'
    if os.path.isfile(mabt_file_name):
        winners_file = pd.concat([winners_file, pd.read_csv(mabt_file_name)[['winners_valid', 'winners_eval']]], axis=1)
        winners_file['winners_valid'] = winners_file['winners_valid'] - 1  # R starts counting from 1
        winners_file['winners_eval'] = winners_file['winners_eval'] - 1  # R starts counting from 1
    winners_performances = winners_file.copy()[0:0]
    splits_file = pd.read_csv('../real_datasets/train_test_split_indices/' + dataset + '.csv', index_col=0)
    for r in range(len(winners_file)):
        dataset_ = arff.load(open('../real_datasets/' + dataset + '.arff'))
        data = pd.DataFrame(dataset_['data'])
        data.columns = [dataset_['attributes'][a][0].lower() for a in range(len(dataset_['attributes']))]
        # Upload train and test datasets
        indices = splits_file[str(r)]
        indices_int = [int(indices[i]) for i in range(len(indices))]
        data_train = data.loc[indices_int]
        data_test = data.drop(indices_int)
        data_train['class'] = ['c' + data_train['class'][i] for i in data_train.index]
        data_test['class'] = ['c' + data_test['class'][i] for i in data_test.index]
        data_train.to_csv('../temp.csv')
        data_test.to_csv('../temp_test.csv')
        train_id = jad.upload_dataset(project, dataset + '_train_run_' + str(r), '../temp.csv')
        test_id = jad.upload_dataset(project, dataset + '_test_run_' + str(r), '../temp_test.csv')
        configuration_descriptions = pd.read_csv('../real_datasets/JAD_results/' + dataset + '_run_' + str(r) +
                                                 '_configurations.csv').values
        configuration_descriptions = [cd[0] for cd in configuration_descriptions]
        winners = list(np.array(configuration_descriptions)[winners_file.loc[r]])
        evaluations = []
        # run for winner configurations
        for c in range(len(winners)):
            if winners[c] in winners[:c]:
                evaluations.append(evaluations[winners[:c].index(winners[c])])
            else:
                winner_description = winners[c]
                best_configuration = [make_configuration(winner_description)]
                tuning_best = tuning_params.custom(best_configuration)
                form_best = analysis_form.testing('class', 'QUICK', cores, tuning_best).with_cv_params(cv_args).\
                    with_plots(['Roc', 'Metrics'])
                jad_evaluation_best = jad.ml.analyze_testing(dataset_id=train_id, form=form_best)
                validation_id = jad.ml.validate(jad_evaluation_best['analysis_id'], test_id)['validation_id']
                validation_results = jad.ml.download_validation_predictions(validation_id)
                jad.ml.delete_analysis(jad_evaluation_best['analysis_id'])
                csvStringIO = StringIO(validation_results.decode("utf-8"))
                validation_df = pd.read_csv(csvStringIO, sep=",")
                if data_train.nunique()['class'] == 2:
                    first_target_value = validation_df.loc[0, 'Label']
                    test_predictions = validation_df['Prob ( class = ' + str(first_target_value) + ' )'].\
                        astype(np.float32)
                    test_predictions = test_predictions.values
                    outcome = validation_df['Label'].values == first_target_value
                    test_set_performance = auc.roc_auc_score(outcome, test_predictions)
                else:
                    head_to_head_auc = []
                    target_values = np.unique(validation_df['Label'])
                    for c_1 in range(len(target_values)):
                        class_1 = target_values[c_1]
                        for c_2 in range(len(target_values)):
                            if c_1 != c_2:
                                class_2 = target_values[c_2]
                                classes = [class_1, class_2]
                                outcome_part = \
                                    validation_df.loc[np.in1d(validation_df['Label'], classes), 'Label'].values == \
                                    class_2
                                test_predictions = validation_df['Prob ( class = ' + str(class_2) + ' )']. \
                                    astype(np.float32)
                                test_predictions = test_predictions.values[np.in1d(validation_df['Label'], classes)]
                                head_to_head_auc.append(auc.roc_auc_score(outcome_part, test_predictions))
                    test_set_performance = np.mean(head_to_head_auc)
                evaluations.append(test_set_performance)
        winners_performances.at[r] = evaluations
        jad.delete_dataset(train_id)
        jad.delete_dataset(test_id)
