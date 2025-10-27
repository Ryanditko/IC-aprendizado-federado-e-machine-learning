# aprendizado-supervisionado.py - Weight-Height Dataset
# Análise supervisionada para predição de gênero baseado em altura e peso

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
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

warnings.filterwarnings('ignore')

def get_data_path():
    """Retorna o caminho para o arquivo de dados"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(base_dir))
    return os.path.join(project_root, "data", "weight_height_data.csv")

def load_data():
    """Carrega o dataset Weight-Height"""
    print("Carregando dataset Weight-Height...")
    data_path = get_data_path()
    
    if os.path.exists(data_path):
        print(f"[OK] Dataset encontrado em: {data_path}")
        return pd.read_csv(data_path)
    else:
        print(f"[INFO] Gerando dataset sintético...")
        return generate_synthetic_data()

def generate_synthetic_data():
    """Gera dados sintéticos caso o dataset não exista"""
    np.random.seed(42)
    n_samples = 10000
    
    # Gerar dados para homens (Male)
    n_male = n_samples // 2
    male_height = np.random.normal(178, 7, n_male)  # cm
    male_weight = np.random.normal(85, 12, n_male)  # kg
    
    # Gerar dados para mulheres (Female)
    n_female = n_samples - n_male
    female_height = np.random.normal(163, 6, n_female)  # cm
    female_weight = np.random.normal(65, 10, n_female)  # kg
    
    # Criar DataFrame
    data = pd.DataFrame({
        'Gender': ['Male'] * n_male + ['Female'] * n_female,
        'Height': np.concatenate([male_height, female_height]),
        'Weight': np.concatenate([male_weight, female_weight])
    })
    
    # Embaralhar
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Salvar
    data_path = get_data_path()
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    data.to_csv(data_path, index=False)
    print(f"[OK] Dataset sintético salvo em: {data_path}")
    
    return data

def exploratory_analysis(data):
    """Realiza análise exploratória dos dados"""
    print("\n" + "="*60)
    print("ANÁLISE EXPLORATÓRIA")
    print("="*60)
    
    print(f"\nInformações do Dataset:")
    print(f"• Número de amostras: {data.shape[0]}")
    print(f"• Número de features: {data.shape[1] - 1}")
    
    print(f"\nPrimeiras linhas:")
    print(data.head())
    
    print(f"\nEstatísticas descritivas:")
    print(data.describe())
    
    print(f"\nDistribuição de Gênero:")
    print(data['Gender'].value_counts())
    
    print(f"\nValores faltantes:")
    missing = data.isnull().sum()
    if missing.sum() == 0:
        print("Não há valores faltantes no dataset")
    else:
        print(missing)
    
    # Visualizações
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Distribuição de altura por gênero
    axes[0, 0].hist([data[data['Gender']=='Male']['Height'], 
                     data[data['Gender']=='Female']['Height']], 
                    label=['Male', 'Female'], alpha=0.7, bins=30)
    axes[0, 0].set_xlabel('Altura (cm)')
    axes[0, 0].set_ylabel('Frequência')
    axes[0, 0].set_title('Distribuição de Altura por Gênero')
    axes[0, 0].legend()
    
    # Distribuição de peso por gênero
    axes[0, 1].hist([data[data['Gender']=='Male']['Weight'], 
                     data[data['Gender']=='Female']['Weight']], 
                    label=['Male', 'Female'], alpha=0.7, bins=30)
    axes[0, 1].set_xlabel('Peso (kg)')
    axes[0, 1].set_ylabel('Frequência')
    axes[0, 1].set_title('Distribuição de Peso por Gênero')
    axes[0, 1].legend()
    
    # Scatter plot
    for gender, color in [('Male', 'blue'), ('Female', 'red')]:
        mask = data['Gender'] == gender
        axes[1, 0].scatter(data[mask]['Height'], data[mask]['Weight'], 
                          c=color, alpha=0.5, label=gender, s=10)
    axes[1, 0].set_xlabel('Altura (cm)')
    axes[1, 0].set_ylabel('Peso (kg)')
    axes[1, 0].set_title('Relação Altura vs Peso')
    axes[1, 0].legend()
    
    # Boxplots
    data.boxplot(column=['Height', 'Weight'], by='Gender', ax=axes[1, 1])
    axes[1, 1].set_title('Boxplot - Altura e Peso por Gênero')
    
    plt.tight_layout()
    plt.savefig('weight_height_exploratory.png', dpi=300, bbox_inches='tight')
    print("\n[OK] Gráfico salvo: weight_height_exploratory.png")
    plt.show()

def preprocess_data(data):
    """Pré-processamento dos dados"""
    print("\n" + "="*60)
    print("PRÉ-PROCESSAMENTO")
    print("="*60)
    
    # Separar features e target
    X = data[['Height', 'Weight']].values
    y = data['Gender'].values
    
    # Codificar labels (Male=1, Female=0)
    y_encoded = np.where(y == 'Male', 1, 0)
    
    print(f"\nShape das features: {X.shape}")
    print(f"Shape do target: {y_encoded.shape}")
    print(f"Classes: Male(1), Female(0)")
    
    # Normalizar features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Features normalizadas: [OK]")
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    
    print(f"\nDivisão dos dados:")
    print(f"• Treino: {X_train.shape[0]} amostras ({X_train.shape[0]/X_scaled.shape[0]*100:.1f}%)")
    print(f"• Teste: {X_test.shape[0]} amostras ({X_test.shape[0]/X_scaled.shape[0]*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test, scaler

def train_models(X_train, X_test, y_train, y_test):
    """Treina e avalia múltiplos modelos"""
    print("\n" + "="*60)
    print("TREINAMENTO E AVALIAÇÃO DOS MODELOS")
    print("="*60)
    
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "SVM": SVC(random_state=42, kernel='rbf'),
        "Naive Bayes": GaussianNB(),
        "K-NN": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\nTreinando {name}...")
        
        # Treinar
        model.fit(X_train, y_train)
        
        # Predições
        y_pred = model.predict(X_test)
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        
        # Validação cruzada
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'CV Mean': cv_mean,
            'CV Std': cv_std
        })
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  CV Score: {cv_mean:.4f} (+/- {cv_std:.4f})")
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("RESULTADOS COMPARATIVOS")
    print("="*60)
    print(results_df.round(4).to_string(index=False))
    
    # Salvar resultados
    results_df.to_csv('weight_height_results.csv', index=False)
    print("\n[OK] Resultados salvos em: weight_height_results.csv")
    
    return results_df, models

def visualize_results(results_df):
    """Visualiza os resultados dos modelos"""
    print("\n" + "="*60)
    print("VISUALIZAÇÕES DOS RESULTADOS")
    print("="*60)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        bars = ax.bar(results_df['Model'], results_df[metric])
        ax.set_title(f'{metric} por Modelo')
        ax.set_ylabel(metric)
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        ax.axhline(y=0.70, color='red', linestyle='--', alpha=0.7, label='Meta 70%')
        ax.set_ylim(0, 1.1)
        
        # Colorir barras
        for bar, value in zip(bars, results_df[metric]):
            if value >= 0.70:
                bar.set_color('green')
            else:
                bar.set_color('orange')
            
            # Adicionar valores
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('weight_height_results.png', dpi=300, bbox_inches='tight')
    print("[OK] Gráfico salvo: weight_height_results.png")
    plt.show()

def generate_report(data, results_df):
    """Gera relatório final"""
    print("\n" + "="*60)
    print("RELATÓRIO FINAL")
    print("="*60)
    
    print(f"\nDataset: Weight-Height")
    print(f"• Total de amostras: {len(data)}")
    print(f"• Features: Height, Weight")
    print(f"• Target: Gender (Male/Female)")
    
    print(f"\nModelos avaliados: {len(results_df)}")
    
    best_model = results_df.loc[results_df['Accuracy'].idxmax()]
    print(f"\nMelhor modelo: {best_model['Model']}")
    print(f"• Accuracy: {best_model['Accuracy']:.4f}")
    print(f"• Precision: {best_model['Precision']:.4f}")
    print(f"• Recall: {best_model['Recall']:.4f}")
    print(f"• F1-Score: {best_model['F1-Score']:.4f}")
    
    # Contar modelos acima da meta
    above_threshold = (results_df['Accuracy'] >= 0.70).sum()
    print(f"\nModelos com Accuracy >= 70%: {above_threshold}/{len(results_df)}")
    
    avg_accuracy = results_df['Accuracy'].mean()
    print(f"\nAccuracy média: {avg_accuracy:.4f}")
    
    if avg_accuracy >= 0.90:
        print("Classificação: EXCELENTE")
    elif avg_accuracy >= 0.80:
        print("Classificação: BOA")
    elif avg_accuracy >= 0.70:
        print("Classificação: SATISFATÓRIA")
    else:
        print("Classificação: ABAIXO DO ESPERADO")

def main():
    """Função principal"""
    print("="*60)
    print("ANÁLISE SUPERVISIONADA - WEIGHT-HEIGHT DATASET")
    print("="*60)
    
    # 1. Carregar dados
    data = load_data()
    
    # 2. Análise exploratória
    exploratory_analysis(data)
    
    # 3. Pré-processamento
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
    
    # 4. Treinar modelos
    results_df, models = train_models(X_train, X_test, y_train, y_test)
    
    # 5. Visualizar resultados
    visualize_results(results_df)
    
    # 6. Gerar relatório
    generate_report(data, results_df)
    
    print("\n" + "="*60)
    print("ANÁLISE CONCLUÍDA")
    print("="*60)

if __name__ == "__main__":
    main()
