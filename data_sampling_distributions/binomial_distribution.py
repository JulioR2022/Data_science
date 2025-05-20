from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
# Fip a coin n times, the probability of getting heads is 0.5
# What to know the chance of getting x heads
n_trials = 10
p = 0.02
x = 3

print(stats.binom.pmf(x,n_trials,p))

result = (stats.poisson.rvs(5, size=1000))
sns.histplot(
    result,
    bins=range(min(result), max(result) + 2),  # Bins discretos para Poisson
    kde=False,
    alpha=0.7,
    color="blue",
    label="Simulação",
    stat="probability"
)

x = range(min(result), max(result) + 1)
pmf = stats.poisson.pmf(x, mu=5)  # Função de massa teórica
plt.scatter(
    x,
    pmf,
    color="#e74c3c",
    marker="o",
    s=100,
    zorder=3,
    label="PMF Teórica (λ=5)"
)

plt.title("Simulação vs. Teórica - Distribuição de Poisson (λ=5)")
plt.xlabel("Valores (k)")
plt.ylabel("Probabilidade")
plt.xticks(x)
plt.legend()
plt.show()