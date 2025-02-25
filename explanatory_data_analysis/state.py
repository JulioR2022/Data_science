import pandas as pd
import numpy as np
from scipy.stats import trim_mean
import wquantiles as wq

#read the file
state = pd.read_csv('../dataset/state.csv')
print(state.head(15))

population_mean = state['Population'].mean()
population_median = state['Population'].median()

#drops 10% of each end
population_trim_mean = trim_mean(state['Population'], 0.1)

print(f'\nMean of Population: {population_mean}')
print(f'\nMedian of Population: {population_median}')
print(f'\nTrim Mean of Population: {population_trim_mean}')

# Compute weighted mean using Numpy
weighted_mean = np.average(state['Murder.Rate'], weights=state['Population'])
# Compute weighted median using wquantiles
weighted_median = wq.median(state['Murder.Rate'], weights=state['Population'])

print(f'\nWeighted Mean of Murder.Rate: {weighted_mean}')
print(f'\nWeighted Median of Murder.Rate: {weighted_median}')

