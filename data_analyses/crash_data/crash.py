import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import random
import warnings
warnings.filterwarnings("ignore")

def fill_with_mode(column = None):
    """
    Função para escolher a moda de uma série de dados.
    """
    fill_column = column.copy()
    modas = column.mode()
    if len(modas) == 1:
        fill_column = fill_column.fillna(modas[0])
        return fill_column
    
    fill_column = fill_column.fillna(random.choice(modas))
    return fill_column

def calculate_outliers(df = None):
    """
    Função para calcular os outliers de um dataframe.
    """
    if df is None:
        raise ValueError("O dataframe não pode ser None.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("O argumento df deve ser um dataframe do pandas.")
    
    numerical_columns = df.select_dtypes(include=['int','float']).columns.tolist()
    outliers = []

    for column in numerical_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
        mask = (df[column] < lower_limit) | (df[column] > upper_limit)
        qtd_outliers = mask.sum()
        percentage_outliers = (qtd_outliers / df[column].size) * 100
        outliers.append({
            'Column': column,
            'Qtd_outliers': qtd_outliers,
            'Percentage_outliers': percentage_outliers,
        })

    return pd.DataFrame(outliers)

filename = '../../dataset/crash_data.csv'
images_path = 'data_visualization/'
df = pd.read_csv(filename, low_memory=False)

df['Speed Limit'] = df['Speed Limit'].replace(['-9', -9, '', ' '], np.nan)
#Forçando erro para NaN
df['Speed Limit'] = pd.to_numeric(df['Speed Limit'], errors='coerce')

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

null_summary = pd.DataFrame({
    'Coluna': df.columns,
    'Quantidade': df.isnull().sum(axis=0).values,
    'Porcentagem': (df.isnull().mean() * 100).round(2).values
})
null_summary.to_markdown(buf= images_path + 'sumario_nulos.md',index=False)

#Lidando com os valores nulos
df = df.dropna(subset = ['Time','Bus Involvement','Articulated Truck Involvement','Gender','Age Group'])
df['Speed Limit'] = df['Speed Limit'].fillna(int(df['Speed Limit'].mean()))
#preenchendo os valores nulos com a moda
df['Heavy Rigid Truck Involvement'] = fill_with_mode(df['Heavy Rigid Truck Involvement'])
df['National Remoteness Areas'] = fill_with_mode(df['National Remoteness Areas'])
df['SA4 Name 2016'] = fill_with_mode(df['SA4 Name 2016'])
df['National LGA Name 2017'] = fill_with_mode(df['National LGA Name 2017'])
df['National Road Type'] = fill_with_mode(df['National Road Type'])


#Usando boxplot para verificar outliers
fig, ax = plt.subplots(1,2,figsize=(14, 7))

ax[0].set_title('Boxplot of Age')
ax[0].set_ylabel('Age')
ax[0].boxplot(df['Age'])

ax[1].set_title('Boxplot of Speed Limit')
ax[1].set_ylabel('Speed Limit')
ax[1].boxplot(df['Speed Limit'])

plt.tight_layout()
plt.savefig(images_path + 'boxplot_age_speed.png')
plt.close()

# Contagem de outliers
# Boxplot deixa claro que não existem outliers, mas para questões de prática irei fazer a contagem
df_outliers = df[['Age', 'Speed Limit']]
df_outliers = calculate_outliers(df_outliers)

figure, ax = plt.subplots(figsize=(14, 7))
ax.axis('off')
table = plt.table(
    cellText=df_outliers.values,
    colLabels=df_outliers.columns,
    cellLoc='center',
    loc='center',
    colColours=['#f0f0f0'] * len(df_outliers.columns),
    
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
plt.title("Outliers in Numerical Variables", fontsize=16,pad = 5)
plt.savefig(images_path + 'outliers_NumericalVariables.png')
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
acc_by_gender = pd.concat([acc_male, acc_female], axis=1)

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

sns.heatmap(df[['Age','Month','Speed Limit']].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig(images_path +'correlation_heatmap.png')
plt.close()

chi2, p_value, dof, expected = chi2_contingency(cross_table)
print(f"Estatística Qui-Quadrado: {chi2:.2f}")
print(f"Valor-p: {p_value:.4f}")