import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

filename = '../dataset/crash_data.csv'
df = pd.read_csv(filename, low_memory=False)

# Verifica os anos existentes no DataFrame
counts = df['Year'].value_counts().sort_index()

# Filtro o periodo de interesse
counts = counts.loc[1989:2021]
# Plotar o gráfico de barras
ax = counts.plot(kind='bar', figsize=(14, 7), edgecolor='black', color='blue')
plt.title("Acidents per Year(1989-2021)", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=45, ha='right')  # Rotacionar rótulos do eixo X

# Adicionar os valores nas barras
for p in ax.patches:
    ax.annotate(
        f"{int(p.get_height())}",  # Texto a ser exibido (valor da barra)
        (p.get_x() + p.get_width() / 2, p.get_height()),  # Posição do texto
        ha='center',  
        va='bottom',
        fontsize=10,
        color='black',
        xytext=(0, 3),  # Offset do texto em pontos
        textcoords='offset points'
    )

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()  # Ajustar layout
plt.show()