# iris_analysis.py

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
import os
from sklearn.datasets import load_iris

# Suprimir warnings desnecessários
warnings.filterwarnings('ignore')

def get_iris_path():
    """Retorna o caminho completo para o arquivo iris.csv"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "iris.csv")

def load_iris_dataset():
    """Carrega o dataset iris de diferentes fontes"""
    print("🔍 Procurando dataset iris...")
    
    iris_path = get_iris_path()
    
    # Tentar carregar de arquivo local primeiro
    if os.path.exists(iris_path):
        print("✅ Arquivo local encontrado!")
        return pd.read_csv(iris_path)
    
    # Tentar via sklearn (método mais confiável)
    try:
        print("📥 Carregando via sklearn...")
        iris = load_iris()
        iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
        iris_df['species'] = iris.target_names[iris.target]
        # Salvar para uso futuro
        iris_df.to_csv(iris_path, index=False)
        print("✅ Dataset carregado via sklearn e salvo localmente!")
        return iris_df
    except Exception as e:
        print(f"❌ Erro no sklearn: {e}")
    
    # Tentar via seaborn
    try:
        print("📥 Tentando baixar via seaborn...")
        iris_df = sns.load_dataset("iris")
        if iris_df is not None and not iris_df.empty:
            # Salvar para uso futuro
            iris_df.to_csv(iris_path, index=False)
            print("✅ Dataset baixado via seaborn e salvo localmente!")
            return iris_df
    except Exception as e:
        print(f"❌ Erro no seaborn: {e}")
    
    # Tentar via URL direta
    try:
        print("📥 Tentando baixar via URL...")
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        iris_df = pd.read_csv(url)
        # Salvar para uso futuro
        iris_df.to_csv('iris.csv', index=False)
        print("✅ Dataset baixado via URL e salvo localmente!")
        return iris_df
    except Exception as e:
        print(f"❌ Erro no download via URL: {e}")
    
    # Se tudo falhar, usar dados sintéticos para demonstração
    print("⚠️ Criando dataset sintético para demonstração...")
    np.random.seed(42)
    n_samples = 150
    
    # Criar dados sintéticos baseados no dataset real
    species = ['setosa'] * 50 + ['versicolor'] * 50 + ['virginica'] * 50
    
    # Features numéricas com distribuições similares ao dataset real
    sepal_length = np.concatenate([
        np.random.normal(5.0, 0.35, 50),  # setosa
        np.random.normal(5.9, 0.51, 50),  # versicolor
        np.random.normal(6.6, 0.64, 50)   # virginica
    ])
    
    sepal_width = np.concatenate([
        np.random.normal(3.4, 0.38, 50),  # setosa
        np.random.normal(2.8, 0.31, 50),  # versicolor
        np.random.normal(3.0, 0.32, 50)   # virginica
    ])
    
    petal_length = np.concatenate([
        np.random.normal(1.5, 0.17, 50),  # setosa
        np.random.normal(4.3, 0.47, 50),  # versicolor
        np.random.normal(5.6, 0.55, 50)   # virginica
    ])
    
    petal_width = np.concatenate([
        np.random.normal(0.25, 0.11, 50), # setosa
        np.random.normal(1.3, 0.20, 50),  # versicolor
        np.random.normal(2.0, 0.27, 50)   # virginica
    ])
    
    iris_synthetic = pd.DataFrame({
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width,
        'species': species
    })
    
    # Salvar dataset sintético
    iris_synthetic.to_csv('iris.csv', index=False)
    print("✅ Dataset sintético criado e salvo!")
    
    return iris_synthetic

def main():
    """Função principal para análise do dataset Iris"""
    try:
        # Carregar dados usando a função robusta
        iris = load_iris_dataset()
        
        # Análise exploratória
        print("\n" + "="*50)
        print("ANÁLISE EXPLORATÓRIA DO DATASET IRIS")
        print("="*50)
        
        print("Primeiras linhas do dataset:")
        print(iris.head())
        
        print("\nInformações do dataset:")
        print(iris.info())
        
        print("\nEstatísticas descritivas:")
        print(iris.describe())
        
        print("\nContagem por espécie:")
        print(iris['species'].value_counts())
        
        print(f"\nValores faltantes por coluna:")
        print(iris.isnull().sum())
        
        # Verificar se há valores faltantes
        total_missing = iris.isnull().sum().sum()
        if total_missing == 0:
            print("✅ Não há valores faltantes no dataset!")
        else:
            print(f"⚠️ Encontrados {total_missing} valores faltantes")
        
        # Visualização
        plt.figure(figsize=(12, 8))
        sns.pairplot(iris, hue='species', diag_kind='hist')
        plt.suptitle('Pairplot do Dataset Iris', y=1.02)
        plt.tight_layout()
        plt.show()
        
        # Pré-processamento
        print("\n" + "="*50)
        print("PRÉ-PROCESSAMENTO DOS DADOS")
        print("="*50)
        
        print(f"Tamanho do dataset: {iris.shape}")
        
        # Remover valores faltantes se existirem
        iris_clean = iris.dropna()
        print(f"Tamanho após limpeza: {iris_clean.shape}")
        
        # Separar features e target
        X = iris_clean.drop('species', axis=1)
        y = iris_clean['species']
        
        print(f"Features: {X.columns.tolist()}")
        print(f"Shape das features: {X.shape}")
        print(f"Número de classes: {y.nunique()}")
        
        # Codificar rótulos
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        print(f"\nClasses encontradas: {le.classes_}")
        print(f"Distribuição das classes: {np.bincount(y_encoded)}")
        
        # Normalização
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"Features normalizadas: ✅")
        
        # Divisão treino-teste
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )
        
        print(f"\nTamanho do conjunto de treino: {X_train.shape}")
        print(f"Tamanho do conjunto de teste: {X_test.shape}")
        
        # Modelos
        print("\n" + "="*50)
        print("TREINAMENTO DOS MODELOS")
        print("="*50)
        
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
        print("Treinando modelos...")
        
        for name, model in models.items():
            try:
                print(f"🤖 Treinando {name}...")
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
                
                print(f"   ✅ {name} - Accuracy: {accuracy:.4f}")
                
            except Exception as e:
                print(f"   ❌ Erro ao treinar {name}: {str(e)}")
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
        print("\n" + "="*50)
        print("ANÁLISE DA META (70%):")
        print("="*50)
        total_approved = 0
        for _, row in results_df.iterrows():
            model_name = row['Model']
            metrics_above_70 = sum([
                row['Accuracy'] >= 0.70,
                row['Precision'] >= 0.70,
                row['Recall'] >= 0.70,
                row['F1-Score'] >= 0.70
            ])
            status = "✅ APROVADO" if metrics_above_70 == 4 else f"⚠️ {metrics_above_70}/4"
            print(f"{model_name}: {status}")
            if metrics_above_70 == 4:
                total_approved += 1
        
        print(f"\n🎯 RESUMO: {total_approved}/{len(results)} modelos atingiram a meta completa!")
        
        # Análise da performance geral
        avg_accuracy = results_df['Accuracy'].mean()
        if avg_accuracy >= 0.90:
            print("🏆 EXCELENTE: Performance média acima de 90%!")
        elif avg_accuracy >= 0.80:
            print("👍 BOA: Performance média acima de 80%!")
        elif avg_accuracy >= 0.70:
            print("✅ SATISFATÓRIA: Performance média acima de 70%!")
        else:
            print("⚠️ ABAIXO DO ESPERADO: Performance média abaixo de 70%")
        
        # Visualização dos resultados
        plt.figure(figsize=(15, 10))
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for i, metric in enumerate(metrics, 1):
            plt.subplot(2, 2, i)
            bars = plt.bar(results_df['Model'], results_df[metric])
            plt.title(f'{metric} - Iris Dataset', fontsize=12, fontweight='bold')
            plt.ylabel(metric)
            plt.xticks(rotation=45, ha='right')
            
            # Adicionar linha de referência em 70%
            plt.axhline(y=0.70, color='red', linestyle='--', alpha=0.7, label='Meta 70%')
            
            # Colorir barras baseado na performance
            for bar, value in zip(bars, results_df[metric]):
                if value >= 0.70:
                    bar.set_color('green')
                else:
                    bar.set_color('orange')
                
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            plt.legend()
            plt.ylim(0, 1.1)
        
        plt.tight_layout()
        plt.suptitle('Métricas de Desempenho - Iris Dataset', y=1.02, fontsize=16, fontweight='bold')
        plt.show()
        
        # Salvar resultados
        results_df.to_csv('iris_results.csv', index=False)
        print(f"\n💾 Resultados salvos em 'iris_results.csv'")
        
        # Relatório final
        print("\n" + "="*50)
        print("RELATÓRIO FINAL")
        print("="*50)
        print(f"📊 Dataset: Iris ({iris.shape[0]} amostras, {iris.shape[1]} features)")
        print(f"🎯 Modelos testados: {len(results)}")
        print(f"✅ Modelos aprovados (4/4 métricas ≥ 70%): {total_approved}")
        print(f"📈 Accuracy média: {avg_accuracy:.3f}")
        print(f"🏆 Melhor modelo: {results_df.loc[results_df['Accuracy'].idxmax(), 'Model']} ({results_df['Accuracy'].max():.3f})")
        
        return results_df
        
    except Exception as e:
        print(f"❌ Erro durante a execução: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🌸 ANÁLISE DO DATASET IRIS")
    print("=" * 50)
    main()