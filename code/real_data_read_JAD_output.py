# Libraries
import numpy as np


# Function to read the JAD output
def read_JAD_output_2(JAD_analysis):
    configuration_descriptions = []
    oos_predictions = []
    for i in range(len(JAD_analysis['oosPredictions']['predictions'].keys())):
        configuration_description = list(JAD_analysis['oosPredictions']['predictions'].keys())[i]
        configuration_descriptions.append(configuration_description)
        # Calculate evaluation
        oos_list = JAD_analysis['oosPredictions']['predictions'][configuration_description]
        if i == 0:
            oos_predictions = np.array(oos_list)
        else:
            oos_predictions = np.append(oos_predictions, np.array(oos_list), axis=1)
    return configuration_descriptions, oos_predictions
