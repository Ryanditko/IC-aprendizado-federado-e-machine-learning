# penguin_analysis.py

import pandas as pd
import numpy as np
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
import warnings

# Suprimir warnings desnecessários
warnings.filterwarnings('ignore')

def main():
    """Função principal para análise do dataset Penguin"""
    try:
        # Carregar dados
        penguin = pd.read_csv('penguins.csv')
        
        # Análise exploratória
        print("Primeiras linhas do dataset:")
        print(penguin.head())
        
        print("\nInformações do dataset:")
        print(penguin.info())
        
        print("\nEstatísticas descritivas:")
        print(penguin.describe(include='all'))
        
        print("\nContagem por espécie:")
        print(penguin['species'].value_counts())
        
        print("\nContagem por ilha:")
        print(penguin['island'].value_counts())
        
        print("\nContagem por sexo:")
        print(penguin['sex'].value_counts())
        
        print(f"\nValores faltantes por coluna:")
        print(penguin.isnull().sum())
        
        # Visualização
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=penguin, x='bill_length_mm', y='bill_depth_mm', 
                       hue='species', style='island')
        plt.title('Relação entre Comprimento e Profundidade do Bico por Espécie e Ilha')
        plt.tight_layout()
        plt.show()
        
        # Pré-processamento
        # Remover linhas com valores faltantes
        print(f"\nTamanho original do dataset: {penguin.shape}")
        penguin_clean = penguin.dropna()
        print(f"Tamanho após remoção de valores faltantes: {penguin_clean.shape}")
        
        # Verificar se ainda temos dados suficientes
        if penguin_clean.shape[0] < 50:
            raise ValueError("Muitos dados foram removidos. Dataset muito pequeno para análise.")
        
        # Separar features e target
        X = penguin_clean.drop('species', axis=1)
        y = penguin_clean['species']
        
        # Verificar se todas as colunas categóricas existem
        categorical_columns = []
        if 'sex' in X.columns:
            categorical_columns.append('sex')
        if 'island' in X.columns:
            categorical_columns.append('island')
        
        # Codificar variáveis categóricas
        if categorical_columns:
            X = pd.get_dummies(X, columns=categorical_columns, drop_first=False)
        
        print(f"\nFeatures após encoding: {X.columns.tolist()}")
        print(f"Shape das features: {X.shape}")
        
        # Codificar rótulos
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        print(f"\nClasses encontradas: {le.classes_}")
        print(f"Distribuição das classes: {np.bincount(y_encoded)}")
        
        # Normalização
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Divisão treino-teste
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )
        
        print(f"\nTamanho do conjunto de treino: {X_train.shape}")
        print(f"Tamanho do conjunto de teste: {X_test.shape}")
        
        # Modelos
        models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
            "SVM": SVC(random_state=42),
            "Naive Bayes": GaussianNB(),
            "k-NN": KNeighborsClassifier(n_neighbors=5),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
        }
        
        # Treinar e avaliar modelos
        results = []
        print("\nTreinando modelos...")
        
        for name, model in models.items():
            try:
                print(f"Treinando {name}...")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Calcular métricas
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                results.append({
                    'Model': name,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1
                })
                
                print(f"{name} - Accuracy: {accuracy:.4f}")
                
            except Exception as e:
                print(f"Erro ao treinar {name}: {str(e)}")
                continue
        
        # Verificar se temos resultados
        if not results:
            raise ValueError("Nenhum modelo foi treinado com sucesso.")
        
        # Resultados
        results_df = pd.DataFrame(results)
        print("\n" + "="*50)
        print("RESULTADOS DOS MODELOS:")
        print("="*50)
        print(results_df.round(4))
        
        # Verificar se alcançou a meta de 70%
        print("\n" + "="*30)
        print("ANÁLISE DA META (70%):")
        print("="*30)
        for _, row in results_df.iterrows():
            model_name = row['Model']
            metrics_above_70 = sum([
                row['Accuracy'] >= 0.70,
                row['Precision'] >= 0.70,
                row['Recall'] >= 0.70,
                row['F1-Score'] >= 0.70
            ])
            print(f"{model_name}: {metrics_above_70}/4 métricas acima de 70%")
        
        # Visualização dos resultados
        plt.figure(figsize=(15, 10))
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for i, metric in enumerate(metrics, 1):
            plt.subplot(2, 2, i)
            bars = plt.bar(results_df['Model'], results_df[metric])
            plt.title(f'{metric} - Penguin Dataset', fontsize=12, fontweight='bold')
            plt.ylabel(metric)
            plt.xticks(rotation=45, ha='right')
            
            # Adicionar linha de referência em 70%
            plt.axhline(y=0.70, color='red', linestyle='--', alpha=0.7, label='Meta 70%')
            
            # Adicionar valores nas barras
            for bar, value in zip(bars, results_df[metric]):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.legend()
            plt.ylim(0, 1.1)
        
        plt.tight_layout()
        plt.suptitle('Métricas de Desempenho - Penguin Dataset', y=1.02, fontsize=16, fontweight='bold')
        plt.show()
        
        # Salvar resultados
        results_df.to_csv('penguin_results.csv', index=False)
        print(f"\nResultados salvos em 'penguin_results.csv'")
        
        return results_df
        
    except FileNotFoundError:
        print("Erro: Arquivo 'penguins.csv' não encontrado!")
        print("Certifique-se de que o arquivo está no diretório atual.")
        return None
    except Exception as e:
        print(f"Erro durante a execução: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()