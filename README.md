# Condifence Interval Estimation of Predictive Performance in the Context of AutoML

This repository contains the code for the paper ["Condifence Interval Estimation of Predictive Performance in the Context of AutoML"](https://arxiv.org/abs/2406.08099) by Konstantinos Paraschakis, Andrea Castellani, Giorgos Borboudakis and Ioannis Tsamardinos (accepted at AutoML 2024).

-----
## Requirements
Our codebase is written in the following languages:
* python3: main code, BBC/BBC-F implementation, main analysis and plots.
* C++: fast AUC-ROC implementation (adapted from: https://github.com/diditforlulz273/fastauc). It requires Linux as OS.
* R: comparative methods, minor adaptation from original code of the authors of ["Post-Selection Confidence Bounds for Prediction Performance"](https://arxiv.org/abs/2210.13206) (https://github.com/pascalrink/mabt-experiments?tab=readme-ov-file).

### Python Dependencies 
* numpy~=1.24.3
* pandas~=1.5.3
* matplotlib~=3.7.1
* scipy~=1.10.1
* scikit-learn~=1.2.2
* joblib~=1.2.0
* seaborn~=0.12.2
* numba~=0.57.0
* tqdm~=4.65.0

Installing dependencies:
```bash
# using pip
pip install -r requirements.txt
# OR
# using Conda
conda create --name <env_name> --file requirements.txt
```

-----
## Repository files description
```bash
.
├── code/                                 # code folder
│   ├── fastauc/                          # contains C++ code for a faster implementation (compared to the standard python sklearn implementation) of the AUC metric
│   │   └── ...
│   ├── core/                             # contains the BBC/BBC-F implementations, the simulated data generator, and other core functions
│   │   ├── BBC_parallel.py               # contains the function to implement BBC and BBC-F. It makes use of multithreading and parallelization tools from joblib library
│   │   ├── generate_data.py              # function to generate the simulation data for the experiments
│   │   └── ...
│   ├── plot_results/
│   │   ├── bbc_time_analysis.py          # script to reproduce the time complexity analysis and generate the plots
│   │   └── plot_results.py               # script to generate the plots included in the paper
│   ├── misc/                             # miscellaneous functions 
│   │   └── ... 
│   ├── real_data_1_splits.py             # imports the datasets (downloaded from OpenML) from a folder in arff. format, creates 100 different stratified train/holdout splits and saves the split indices in a dedicated folder
│   ├── real_data_2_run_on_JAD.py         # uploads for each dataset the various training subsets from the previous step and runs the analysis on JADBio. Then it stores the resulting outcome, out-of-sample predictions, fold indices, and descriptions of the applied configurations. This step cannot be run without special API access to JADBio. Instead, we have included in the accompanying meterial the output (folder "JAD_results") needed to run further steps
│   ├── real_data_3_apply_BBC_on_data.py  # applies the two BBC methods on the results from the previous step and stores the best configuration for each JADBio analysis and the corresponding botstrap distributions
│   ├── real_data_4_evaluation_on_JAD.py  # uploads for each dataset/split the holdout dataset and runs the winner configuration to get the holdout performances. This step cannot be run without special API access to JADBio. Instead, we have included in the accompanying meterial the output (folder "real_datasets_holdOut_performances") needed to run further steps
│   ├── real_data_5_calculate_CI.py       # takes the results of the previous steps, along with the corresponding results from the R-code run on the same out-of-sample predictions for the comparison methods, and calculates the inclusion percentages and the average tightnesses and stores the results
│   ├── CI_via_bootstrap_on_winner_configuration.py # Naive method NB
│   ├── simulation_1_data.py                    # simulates the configuration performances, outcomes, out-of-sample predictions, and fold indices for the various simulation settings presented in the paper and stores them as csv files in a dedicated folder
│   ├── simulation_2_apply_BBC_on_data.py       # applies the two BBC methods (original and BBC-F) on the simulated predictions from the previous step and stores the resulting bootstrap distributions and the theoreticl performqnces of the selected configurations in a dedicated folder
│   ├── simulation_3_calculate_CI.py            # takes the output from the previous step and calculates the one-sided 95% CI for all cases and the inclusion percentages and the average tightness in each case and stores all results
│   ├── simulation_4_combine_with_R_results.py  # combines the results on the two BBC methods and the results from the R-code (discussed later) on the rest of the methods
│   └── R_code/                           # contains the R code for the comparative algorithms
│       ├── MabtCi-function.R             # contains the function for applying the MABT method
│       ├── TiltCi-function.R             # contains the function for applying the BT method
│       ├── simulated_data.R              # loads the data simulated by "simulation_1_data.py" above and applies all comparison methods to our BBC methods. It calculates the 95% one-sided CI's, along with the corresponding "winner configurations" in each case and stores them. The results are then loaded by "simulation_4_combine_with_R_results.py" as mentioned above
│       └── real_datasets.R               # loads the JADBio output on the real datasets stored by "real_data_2_run_on_JAD.py" above and calculates the 95% intervals for all datasets and all splits, along with the corresponding "winner configurations" and stores the results
├── output/                               # output results folder. It contains final and intermediate results
│   ├── time_analysis/          # folder with logs and plots of the BBC/BBC-F time complexity analysis
│   │   └── ...
│   ├── final_results.xlsx      # excel sheet with the summary of the final results reported in the paper
│   └── ...                     # the other files are intermediate results output by the library. They are stored here for simplicity
├── real_datasets/              # create this folder to store JAD results
│   └── ...                     # <- put here the JAD results
└── ...
```

-----
## Download JAD results
To access and download the raw results on the real-world benchmakrs dataset (1.12 GB), obtained with JADBio , click the button:
[![Download oos results](https://img.shields.io/badge/Download-Dataset-blue.svg)](https://figshare.com/s/b8f72d61476be2fc04fc)

In order to corectly run the code and reproduce the results, create and unpack the files in the folder: ```./real_datasets/``` . 

The structure of the provided results is the following:
```bash
└── out-of-sample results with JADBio/                                
    ├── JAD_results/        # output of script 'real_data_2_run_on_JAD.py'
    │   ├── <dataset_name>_run_<N>_configurations.csv      
    │   ├── <dataset_name>_run_<N>_outcome.csv                 
    │   ├── <dataset_name>_run_<N>_splitIndices.csv                  
    │   └── ...
    └── real_datasets_holdOut_performances/    # output of script 'real_data_4_evaluation_on_JAD.py'    
        ├── <dataset_name>.csv                 
        └── ...
```


-----
## Running instructions

We conduct experiments with real-world data, and simulated data.
The runtime of our experiments, just the python part i.e., analysis of the data and computation of BBC and BBC-F, depends on the system specification, but it can go from 2h to 10h.

To train the models and evaluate their performance on the real world data, we use the commercial suite for AutoML [JADBio](https://jadbio.com/).
With access to their API, to reproduce the results, just execute in order the files ```./code/real_data_<#>.py```, from 1 to 5.
Without access to the JADBio API, we provide the evaluation output at the aforementioned link, and the intermediate analysis in the folder ./output/ and also the final results in the final_results.xlsx file.


To reproduce the results with the simulated data, here it is an example.
To change the simulated data hyperparameter, modify the lines 7-12 in the file ```./code/simulation_1_data.py```.

### Example
Turn on the created python environment and run the following code from the ```./code/``` folder:
```bash
python simulation_1_data.py
python simulation_2_apply_BBC_on_data.py
python simulation_3_calculate_CI.py
```
At this point, the R code from ```./code/R_code/``` folder needs to run.
In case there are some problems with R, we provided the expected output in the ```/output/``` folder.
Then:
```bash
python simulation_4_combine_with_R_results.py
```
Now, you should have a folder with all the results. 
For simplicty, we summarize them in an excel file (```/output/final_results.xlsx```), where we compute the p-score and ranking.
Finally, to reproduce the plots, just run:
```bash
python plot_results/plot_results.py
```

-----
To reproduce the time-complexity analysis of BBC and BBC-F. Execute the following command:
```bash
python plot_results/bbc_time_analysis.py
```
Then, the produced plots and logs should be found in the folder ```/output/time_analsys/``` .

Anyway, we provide all our intermediate and final results of our experiments in ```/output/``` folder.

-----
## References
If this work has been useful, please cite us with:
```Bibtex
@inproceedings{
Paraschakis2024confidence,
title={Confidence Interval Estimation of Predictive Performance in the Context of Auto{ML}},
author={Konstantinos Paraschakis, Andrea Castellani, Giorgos Borboudakis and Ioannis Tsamardinos},
booktitle={AutoML 2024 Methods Track},
year={2024},
url={https://arxiv.org/abs/2406.08099}
}
```

