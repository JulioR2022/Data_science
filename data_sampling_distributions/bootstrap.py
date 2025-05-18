from sklearn.utils import resample
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   

loan_income = pd.read_csv('../dataset/loan_income.csv').squeeze('columns')

# Create a bootstrap sample
n_samples = 10000
results = []
for _ in range(n_samples):
    sample = resample(loan_income)
    results.append(sample.mean())

results = pd.Series(results)

print(f'Original mean: {loan_income.mean():.2f}')
print(f"Mean of the bootstrap sample: {results.mean():.2f}")
print(f'Original Median: {loan_income.median():.2f}')
print(f"Median of the bootstrap sample: {results.median():.2f}")


# Confidence intervals
lower_bound = results.quantile(0.05)
upper_bound = results.quantile(0.95)
print(f'95% confidence interval: {lower_bound:.2f} - {upper_bound:.2f}')

# Plot the bootstrap distribution  
sns.histplot(results, bins=30, kde=True)
plt.axvline(lower_bound, color='red', linestyle='--', label='Lower Bound')
plt.axvline(upper_bound, color='green', linestyle='--', label='Upper Bound')
plt.axvline(loan_income.mean(), color='blue', linestyle='--', label='Original Mean')
plt.axvline(loan_income.median(), color='orange', linestyle='--', label='Original Median')
plt.title('Bootstrap Distribution of Loan Income Means')
plt.xlabel('Mean Loan Income')
plt.ylabel('Frequency')
plt.legend()
plt.show()