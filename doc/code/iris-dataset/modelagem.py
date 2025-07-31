# iris_analysis.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar dados
iris = pd.read_csv('iris.csv')

# Análise exploratória
print("Primeiras linhas do dataset:")
print(iris.head())

print("\nInformações do dataset:")
print(iris.info())

print("\nEstatísticas descritivas:")
print(iris.describe())

print("\nContagem por espécie:")
print(iris['species'].value_counts())

# Visualização
sns.pairplot(iris, hue='species')
plt.suptitle('Pairplot do Dataset Iris', y=1.02)
plt.show()

# Pré-processamento
X = iris.drop('species', axis=1)
y = iris['species']

# Codificar rótulos
le = LabelEncoder()
y = le.fit_transform(y)

# Normalização
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Divisão treino-teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelos
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Naive Bayes": GaussianNB(),
    "k-NN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

# Treinar e avaliar modelos
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1-Score': f1_score(y_test, y_pred, average='weighted')
    })

# Resultados
results_df = pd.DataFrame(results)
print("\nResultados dos Modelos:")
print(results_df)

# Visualização dos resultados
plt.figure(figsize=(12, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 2, i)
    sns.barplot(x='Model', y=metric, data=results_df)
    plt.title(metric)
    plt.xticks(rotation=45)
plt.tight_layout()
plt.suptitle('Métricas de Desempenho - Iris Dataset', y=1.02)
plt.show()