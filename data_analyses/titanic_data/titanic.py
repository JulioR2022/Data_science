import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.cluster import KMeans
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
from sklearn.metrics import f1_score, recall_score, precision_score

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
train['Age'] = train['Age'].fillna(int(train['Age'].median()))

# Devido à alta quantidade de valores ausentes em 'Cabin', vamos criar uma nova categoria 'Unknown' para esses casos. Pode ser interessante para análises futuras.
train['Cabin'] = train['Cabin'].fillna('U')

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


#Tratando variaveis categoricas

train_dummies = pd.get_dummies(train, columns=['Sex'], drop_first=True) 
train['Sex'] = train_dummies['Sex_male']


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

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
sns.countplot(x='Sex', data=train, ax=axes[0])
axes[0].set_title('Distribuição de Sexo')
sns.countplot(x='Embarked', data=train, ax=axes[1])
axes[1].set_title('Distribuição de Porto de Embarque')
sns.countplot(x='Pclass', data=train, ax=axes[2])
axes[2].set_title('Distribuição de Classe do Passageiro')
plt.tight_layout()
plt.savefig(images_path + 'categorical_distributions.png')
plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(train[numeric_columns].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação de Variáveis Numéricas')
plt.tight_layout()
plt.savefig(images_path + 'correlation_matrix.png')
plt.close()

sns.pairplot(train[['Age', 'Fare', 'SibSp', 'Parch', 'Survived']], hue='Survived', palette='viridis')
plt.suptitle('Pair Plot de Variáveis Numéricas por Sobrevivência', y=1.02)
plt.tight_layout()
plt.savefig(images_path + 'pairplot_numerical_variables.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.histplot(data=train, x='Age', hue='Survived', kde=True, palette='viridis',
             binwidth=5)
plt.title('Distribuição de Idade por Status de Sobrevivência')
plt.xlabel('Idade')
plt.ylabel('Contagem')
plt.legend(title='Sobreviveu', labels=['Não', 'Sim'])
plt.savefig(images_path + 'age_survival_distribution.png')
plt.close()

#Survival Rate by Age Bins
# Define age bins
bins = [0, 12, 18, 35, 60, 80]
labels = ['Criança', 'Adolescente', 'Jovem Adulto', 'Adulto', 'Idoso']
train['AgeGroup'] = pd.cut(train['Age'], bins=bins, labels=labels, right=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='AgeGroup', y='Survived', data=train, palette='viridis')
plt.title('Taxa de Sobrevivência por Faixa Etária')
plt.xlabel('Faixa Etária')
plt.ylabel('Taxa de Sobrevivência')
plt.savefig(images_path + 'survival_rate_by_age_bins.png')
plt.tight_layout()
plt.close()

# Clean up the temporary 'AgeGroup' column if you don't need it further
train = train.drop(columns=['AgeGroup'])

# Preparando dados para análise de Deck
train['Deck'] = train['Cabin'].apply(lambda x: x[0])
deck_order = sorted(train['Deck'].unique())

deck_survival_rate = train.groupby('Deck')['Survived'].mean().reset_index()
deck_survival_rate.columns = ['Deck', 'Taxa']

deck_counts = train['Deck'].value_counts().reindex(deck_order)
deck_df = pd.DataFrame({'Deck': deck_counts.index, 'Count': deck_counts.values})

# Distribuição de Passageiros por Deck
plt.figure(figsize=(10, 6))
sns.barplot(data=deck_df,x='Deck', y='Count',hue='Deck',legend=False) 
plt.title('Distribuição de Passageiros por Deck')
plt.xlabel('Deck')
plt.ylabel('Número de Passageiros')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(images_path + 'passenger_distribution_by_deck.png')
plt.close()

# Taxa de Sobrevivência por Deck
plt.figure(figsize=(10, 6))
sns.barplot(data= deck_survival_rate,x='Deck', y='Taxa', hue='Deck',legend=False)
plt.title('Taxa de Sobrevivência por Deck')
plt.xlabel('Deck')
plt.ylabel('Taxa de Sobrevivência')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(images_path + 'survival_rate_by_deck.png')
plt.close()

# Regressão Linear
encoder = LabelEncoder()

df_regression = train.drop(['PassengerId','Name','Ticket','Cabin', 'Embarked'],axis=1)
df_regression['Deck'] = encoder.fit_transform(df_regression['Deck'])
df_regression['Pclass'] = encoder.fit_transform(df_regression['Pclass'])

x = df_regression.drop(['Survived'], axis=1)
y = df_regression['Survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

sd = StandardScaler()
x_train = sd.fit_transform(x_train)
x_test = sd.fit_transform(x_test)

classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

accuracy_score(y_test, y_pred)
conf_m = confusion_matrix(y_test, y_pred)


sns.heatmap(conf_m, annot=True)
plt.savefig(images_path + 'confusion_matrix')
plt.close()


# Plotar roc curv
prob = classifier.predict_proba(x_test)
prob = prob[:,1]
fpr, tpr, trash = roc_curve(y_test, prob)

plt.plot(fpr, tpr)
plt.title('Curva ROC')
plt.savefig(images_path + 'roc_curve.png')
plt.close()

print(precision_score(y_test,y_pred))
print(recall_score(y_test,y_pred))
print(f1_score(y_test,y_pred))

#Klusterização

wcss = []
for i in range(1,15):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(df_regression)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 15), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig(images_path + 'elbow.png')
plt.close()

kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(df_regression)
df_regression['cluster'] = y_kmeans
sns.scatterplot(data = df_regression, x = 'Fare', y =  'Deck', hue = 'cluster')
plt.savefig(images_path + 'Klusters.png')
plt.close()

#Decision Tree
modelo = DecisionTreeClassifier(max_depth=10, max_leaf_nodes=20, criterion='gini')

modelo.fit(x_train, y_train)
y_pred = modelo.predict(x_test)

print(accuracy_score(y_test, y_pred))

conf_m = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_m, annot=True)
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.savefig(images_path + 'decision_tree_confMatrix')
plt.close()