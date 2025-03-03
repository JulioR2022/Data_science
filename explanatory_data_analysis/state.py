import pandas as pd
import numpy as np
from scipy.stats import trim_mean
import wquantiles as wq
from statsmodels import robust

#read the file
state = pd.read_csv('../dataset/state.csv')
print(state.head(15))

population_mean = state['Population'].mean()
population_median = state['Population'].median()

#drops 10% of each end
population_trim_mean = trim_mean(state['Population'], 0.1)

#Compute the standard deviation of Population
population_std = state['Population'].std()

#Compute the IQR of Population
population_iqr = (state['Population'].quantile(0.75) 
                    - state['Population'].quantile(0.25))

#Compute the MAD of Population
population_mad = robust.scale.mad(state['Population'])

print(f'\nMean of Population: {population_mean}')
print(f'\nMedian of Population: {population_median}')
print(f'\nTrim Mean of Population: {population_trim_mean}')
print(f'\nStandand Deviation of Population: {population_std}')
print(f'\nIQR of Population: {population_iqr}')
print(f'\nMAD of Population: {population_mad}')


# Compute weighted mean of Murder.Rate using Numpy
weighted_mean = np.average(state['Murder.Rate'], weights=state['Population'])
# Compute weighted median Murder.Rate using wquantiles 
weighted_median = wq.median(state['Murder.Rate'], weights=state['Population'])


print(f'\nWeighted Mean of Murder.Rate: {weighted_mean}')
print(f'\nWeighted Median of Murder.Rate: {weighted_median}')

#Quantiles to be calculated
percentages = [0.05, 0.15, 0.25, 0.50, 0.75]

#Compute the percentales
percentales = state['Murder.Rate'].quantile(percentages)
#Create the DataFrame
table_quantiles = pd.DataFrame(percentales)
#Edit the Name of the atributes
table_quantiles.index = [f'{p * 100}%' for p in percentages]

print("\n\t\tTable of Quantiles\n")
print(table_quantiles.transpose())

