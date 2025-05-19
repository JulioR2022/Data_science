import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from sklearn.utils import resample
"""
This code generates a QQ plot using a sample of 1000 random numbers 
drawn from a normal distribution.
"""

fig, ax = plt.subplots(figsize=(8, 8))
# Generate a sample of 1000 random numbers from a normal distribution
normal_sample = stats.norm.rvs(size=1000)
stats.probplot(normal_sample, plot=ax)
plt.tight_layout()
plt.show()

# Generate a QQ plot using a standarded sample of 1000 random numbers drawn from loan_income
fig2, ax2 = plt.subplots(figsize=(8, 8))
loan_income = pd.read_csv('../dataset/loan_income.csv').squeeze('columns')
loan_income = resample(loan_income, n_samples=1000, replace=False)
loan_income_final = []
loan_mean = loan_income.mean()
loan_std = loan_income.std()
for loan in loan_income:
    loan_income_final.append((loan - loan_mean) / loan_std)

stats.probplot(loan_income_final, plot=ax2)
plt.ylabel('Standardized Loan Income')
plt.xlabel('Quantile of Normal Distribution')
plt.tight_layout()
plt.show()
