import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

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

images_path = 'data_visualization/'
train = pd.read_csv('../../dataset/train.csv')

numeric_columns = train.select_dtypes(include=['int', 'float']).columns.tolist()
categorical_columns = train.select_dtypes(include=['object']).columns.tolist()

train_numeric = train[numeric_columns]
train_categorical = train[categorical_columns]

train_numerical_descrition = train[['Age','SibSp','Fare','Parch']].describe()

fig, ax = plt.subplots(figsize=(18,7))
ax.axis('off')
table = ax.table(
    cellText=train_numerical_descrition.values.round(4),
    colLabels=train_numerical_descrition.columns,
    rowLabels=train_numerical_descrition.index,
    cellLoc='center',
    loc='center',
    colColours=['#f0f0f0'] * len(train_numerical_descrition.columns),
    rowColours=['#f0f0f0'] * len(train_numerical_descrition.index)
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
plt.title("Descriptive Statistics of Numerical Columns", fontsize=16)
plt.savefig(images_path + 'numerical_columns_description.png')
plt.close()

null_summary = pd.DataFrame({
    'Coluna': train.columns,
    'Quantidade': train.isnull().sum(axis=0).values,
    'Porcentagem': (train.isnull().mean() * 100).round(2).values
})
null_summary.to_markdown(buf= images_path + 'sumario_nulos.md',index=False)

# Lidando com valores nulos em 'Age', 'Cabin', 'Embarked'
train = train.dropna(subset=['Embarked'])
train['Age'] = train['Age'].fillna(int(train['Age'].mean()))
train['Cabin'] = fill_with_mode(train['Cabin'])

# Usando Boxplot para identificar outliers

fig, ax = plt.subplots(2,2,figsize=(20, 14))
ax[0,0].set_title('Boxplot of Age')
ax[0,0].boxplot(train['Age'])

ax[0,1].set_title('Boxplot of Fare')
ax[0,1].boxplot(x=train['Fare'])

ax[1,0].set_title('Boxplot of SibSp')
ax[1,0].boxplot(x=train['SibSp'])

ax[1,1].set_title('Boxplot of Parch')
ax[1,1].boxplot(x=train['Parch'])
plt.savefig(images_path + 'boxplot_plt.png')
plt.close()

# Usando Seaborn para visualizar os outliers
fig, ax = plt.subplots(2,2,figsize=(20, 14))
sns.boxplot(x=train['Age'], ax=ax[0,0])
sns.boxplot(x=train['Fare'], ax=ax[0,1])
sns.boxplot(x=train['SibSp'], ax=ax[1,0])
sns.boxplot(x=train['Parch'], ax=ax[1,1])
plt.savefig(images_path + 'boxplot_sns.png')
plt.close()

# Usando Seaborn para visualizar a distribuição dos dados
fig, ax = plt.subplots(2,2,figsize=(20, 14))
sns.histplot(train['Age'], ax=ax[0,0], kde=True)
sns.histplot(train['Fare'], ax=ax[0,1], kde=True)
sns.histplot(train['SibSp'], ax=ax[1,0], kde=True)
sns.histplot(train['Parch'], ax=ax[1,1], kde=True)

for axis in ax.flat:
    axis.set_ylabel('Frequency', fontsize=12)

plt.tight_layout()
plt.savefig(images_path + 'histogram_sns.png')
plt.close()

train_outliers = train[['Age','SibSp','Fare','Parch']]
train_outliers = calculate_outliers(train_outliers)

figure, ax = plt.subplots(figsize=(14, 7))
ax.axis('off')
table = plt.table(
    cellText=train_outliers.values,
    colLabels=train_outliers.columns,
    cellLoc='center',
    loc='center',
    colColours=['#f0f0f0'] * len(train_outliers.columns),
    
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
plt.title("Outliers in Numerical Variables", fontsize=16,pad = 5)
plt.savefig(images_path + 'outliers_NumericalVariables.png')
plt.close()

