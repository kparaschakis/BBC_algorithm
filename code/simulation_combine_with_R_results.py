# Libraries
import pandas as pd

# Load results
bbc_results = pd.read_csv('../summaries_BBC.csv')
bbc_results.index = bbc_results['configuration']
bbc_results.drop('configuration', axis=1, inplace=True)
r_results = pd.read_csv('../multiplicity-adjusted_bootstrap_tilting/summaries_MABT.csv')
r_results.index = r_results['configuration']
r_results.drop('configuration', axis=1, inplace=True)
results = pd.concat([bbc_results, r_results], axis=1)
results = results.loc[results.isna().sum(axis=1) == 0]
results.to_csv('../summaries_all.csv')
