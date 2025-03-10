import pandas as pd
import numpy as np
from scipy.stats import trim_mean
import wquantiles as wq
from statsmodels import robust
import matplotlib.pylab as plt


lc_loans = pd.read_csv('../dataset/lc_loans.csv')
tab = lc_loans.pivot_table(index='grade', columns='status',
                                aggfunc=lambda x: len(x), margins=True) 

data_frame = tab.loc['A':'G',:].copy()
data_frame.loc[:,'Charged Off':'Late'] = data_frame.loc[:,'Charged Off':'Late'].div(data_frame['All'], axis=0)

data_frame['All'] = data_frame['All'] / sum(data_frame['All'])
print(data_frame)