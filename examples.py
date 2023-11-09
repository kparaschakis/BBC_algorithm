# Imports
from functions import *

# Create some example arrays
distributions = np.random.normal(0, 1, 100000).reshape(100, 1000)
theoretical = np.random.normal(0, 0.5, 100)
# Apply percentile uniformity checking function
percentile_uniformity(distributions, theoretical)
