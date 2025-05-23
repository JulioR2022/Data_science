import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings("ignore")

filename = '../../dataset/crash_data.csv'
images_path = 'data_visualization/'
df = pd.read_csv(filename, low_memory=False)

numerical_columns = df.select_dtypes(include=['int', 'float']).columns.tolist()
categoric_columns = df.select_dtypes(include=['object']).columns.tolist()

df_numerical = df[numerical_columns]
df_categoric = df[categoric_columns]

df_numerical_descrition = df_numerical.describe()

fig, ax = plt.subplots(figsize=(18,7))
ax.axis('off')
table = ax.table(
    cellText=df_numerical_descrition.values.round(4),
    colLabels=df_numerical_descrition.columns,
    rowLabels=df_numerical_descrition.index,
    cellLoc='center',
    loc='center',
    colColours=['#f0f0f0'] * len(df_numerical_descrition.columns),
    rowColours=['#f0f0f0'] * len(df_numerical_descrition.index)
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
plt.title("Descriptive Statistics of Numerical Columns", fontsize=16)
plt.savefig(images_path + 'numerical_columns_description.png')
plt.close()


# Verifica os anos existentes no DataFrame
counts = df['Year'].value_counts().sort_index()

# Filtro o periodo de interesse
counts = counts.loc[1989:2021]
# Plotar o gráfico de barras
ax = counts.plot(kind='bar', figsize=(14, 7), edgecolor='black', color='blue')
plt.title("Accidents per Year(1989-2021)", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Number of Accidents", fontsize=12)
plt.xticks(rotation=45, ha='right')  # Rotacionar rótulos do eixo X
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()  # Ajustar layout
plt.savefig(images_path +'accidents_per_year.png')
plt.close()


male = df[df['Gender'] == 'Male']
female = df[df['Gender'] == 'Female']
# Verificar relação entre acidentes e sexo
acc_male = male['Year'].value_counts().sort_index().rename('Male')
acc_female = female['Year'].value_counts().sort_index().rename('Female')
acc_by_gender = pd.concat([acc_male, acc_female], axis=1).fillna(0)

plt.figure(figsize=(14, 7))

# Definir posições das barras
years = acc_by_gender.index
bar_width = 0.4

# Barras para homens
plt.bar(
    years - bar_width/2, 
    acc_by_gender['Male'], 
    width=bar_width, 
    color='blue', 
    alpha=0.7, 
    label='Male',
    edgecolor='black'
)

# Barras para mulheres
plt.bar(
    years + bar_width/2, 
    acc_by_gender['Female'], 
    width=bar_width, 
    color='pink', 
    alpha=0.7, 
    label='Female',
    edgecolor='black'
)

plt.title("Acidentes por Ano e Gênero (1989-2021)", fontsize=14)
plt.xlabel("Ano", fontsize=12)
plt.ylabel("Número de Acidentes", fontsize=12)
plt.xticks(years, rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(images_path +'accidents_year_gender.png')
plt.close()

# Garante que o gênero seja apenas masculino ou feminino
filtered_df = df[df['Gender'].isin(['Male', 'Female'])]
# Gera um dataframe relacionando o tipo de acidente e o gênero
cross_table = pd.crosstab(df['Dayweek'], filtered_df['Gender'])
plt.figure(figsize=(12, 8))
# annot -> exibe os valores na tabela
# fmt -> formata os valores exibidos
sns.heatmap(cross_table, annot=True, fmt='d', cmap='Blues')
plt.title("Accidents per Week and Gender")
plt.xlabel("Gender")
plt.ylabel("Day of the Week")
plt.savefig(images_path +'heatmap_acc_weekdays_gender.png')
plt.close()

chi2, p_value, dof, expected = chi2_contingency(cross_table)
print(f"Estatística Qui-Quadrado: {chi2:.2f}")
print(f"Valor-p: {p_value:.4f}")