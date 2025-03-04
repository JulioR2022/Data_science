import pandas as pd
import numpy as np
from scipy.stats import trim_mean
import wquantiles as wq
from statsmodels import robust
import matplotlib.pylab as plt

#read the file
dfw = pd.read_csv('../dataset/dfw.csv')
print(100 * dfw / dfw.values.sum())

#Create a bar charts
ax = dfw.transpose().plot.bar(figsize=(4, 4), legend=False)
ax.set_xlabel('Cause of delay')
ax.set_ylabel('Count')

plt.tight_layout()
plt.show()

