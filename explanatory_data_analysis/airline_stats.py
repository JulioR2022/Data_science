import pandas as pd
import numpy as np
from scipy.stats import trim_mean
import wquantiles as wq
from statsmodels import robust
import matplotlib.pylab as plt

airline_stats = pd.read_csv('../dataset/airline_stats.csv')
print(airline_stats.head(8))

ax = airline_stats.boxplot('pct_carrier_delay', by='airline', figsize=(8, 8))
ax.set_xlabel('')
ax.set_ylabel('Daily % of Delayed Flights')
plt.suptitle('')
plt.title('')
plt.show()