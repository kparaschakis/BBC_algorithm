# Condifence Interval Estimation of Predictive Performance in the Context of AutoML

Scripts "BBC_parallel.py", "compare_pooledBBC_foldBBC.py", "compute_CI_test.py", "generate_data.py", "make_API_configuration.py", and "read_JAD_output.py" contain code that is either loaded by the main scripts we will discuss below, or code that is not used at all for this paper.

-----

The folder "fastAUC" contains a faster implementation (compared to the standard python sklearn implementation) of the AUC metric.

-----

The main scripts are grouped into two categories: "simulation" and "real_data". Each category contains the scripts numbered in the order they need to be applied.

Simulation:
- "simulation_1_data.py" simulates the configuration performances, outcomes, out-of-sample predictions, and fold indices for the various simulation settings presented in the paper and stores them as csv files in a dedicated folder.
- "simulation_2_apply_BBC_on_data.py" applies the two BBC methods (original and BBC-F) on the simulated predictions from the previous step and stores the resulting bootstrap distributions and the theoreticl performqnces of the selected configurations in a dedicated folder.
- "simulation_3_calculate_CI" takes the output from the previous step and calculates the one-sided 95% CI for all cases and the inclusion percentages and the average tightness in each case and stores all results.
- "simulation_4_combine_with_R_results.py" combines the results on the two BBC methods and the results from the R-code (discussed later) on the rest of the methods.

Real data:
- "real_data_1_splits" imports the datasets (downloaded from OpenML) from a folder in arff. format, creates 100 different stratified train/holdout splits and saves the split indices in a dedicated folder.
- "real_data_2_run_on_JAD.py" uploads for each dataset the various training subsets from the previous step and runs the analysis on JADBio. Then it stores the resulting outcome, out-of-sample predictions, fold indices, and descriptions of the applied configurations. This step cannot be run without special API access to JADBio. Instead, we have included in the accompanying meterial the output (folder "JAD_results") needed to run further steps.
- "real_data_3_apply_BBC_on_data.py" applies the two BBC methods on the results from the previous step and stores the best configuration for each JADBio analysis and the corresponding botstrap distributions.
- "real_data_4_evaluation_on_JAD" uploads for each dataset/split the holdout dataset and runs the winner configuration to get the holdout performances. This step cannot be run without special API access to JADBio. Instead, we have included in the accompanying meterial the output (folder "real_datasets_holdOut_performances") needed to run further steps.
- "real_data_5_calculate_CI" takes the resuls of the previous steps, along with the corresponding results from the R-code run on the same out-of-sample predictions for the comparison methods, and calculates the inclusion percentages and the average tightnesses and stores the results.

-----

Below are some notes on the R-code provided by the authors of "Post-Selection Confidence Bounds for Prediction Performance. 422 2023. arXiv: 2210.13206 [stat.ML]", used to apply the comparison methods in our paper. (https://github.com/pascalrink/mabt-experiments?tab=readme-ov-file). This code can also be found in this repository.

The R scripts "MabtCi-function.R" and "TiltCi-function.R" contain the functions for applying the MABT and BT methods and are loaded by the other two scripts "simulated_data.R" and "real_datasets.R". 

"simulated_data.R" loads the data simulated by "simulation_1_data.py" above and applies all comparison methods to our BBC methods. It calculates the 95% one-sided CI's, along with the corresponding "winner configurations" in each case and stores them. The results are then loaded by "simulation_4_combine_with_R_results.py" as mentioned above.

"real_datasets.R" loads the JADBio output on the real datasets stored by "real_data_2_run_on_JAD.py" above and calculates the 95% intervals for all datasets and all splits, along with the corresponding "winner configurations" and stores the results.

-----

The script "plot_results.py" generates the plots found in the paper. As input it requires the excel file "BBC vs MABT.xlsx" that has a summary of the experimental results.

The script "bbc_time_analysis.py" computes the runtime analysis between the pooled and the fold-BBC.



