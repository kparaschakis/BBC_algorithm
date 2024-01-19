# Libraries
import numpy as np


# Function to read the JAD output
def read_JAD_output_classification(JAD_analysis):
    split_indices = JAD_analysis['splitIndices']
    configuration_descriptions = []
    outcome = JAD_analysis['targetData']['data']
    if len(np.unique(outcome)) == 2:
        oos_predictions = []
    else:
        oos_predictions = np.zeros((len(outcome),
                                    len(JAD_analysis['oosPredictions']['predictions'].keys()),
                                    len(np.unique(outcome))))
    for c in range(len(JAD_analysis['oosPredictions']['predictions'].keys())):
        configuration_description = list(JAD_analysis['oosPredictions']['predictions'].keys())[c]
        configuration_descriptions.append(configuration_description)
        # gather predictions
        if len(np.unique(outcome)) == 2:
            oos_list = JAD_analysis['oosPredictions']['predictions'][configuration_description]
            if c == 0:
                oos_predictions = np.array(oos_list)
            else:
                oos_predictions = np.append(oos_predictions, np.array(oos_list), axis=1)
        else:
            oos_predictions[:, c, :] =\
                np.array([JAD_analysis['oosPredictions']['predictions'][configuration_description][i][0]
                          for i in range(len(outcome))])
    return outcome, split_indices, configuration_descriptions, oos_predictions
