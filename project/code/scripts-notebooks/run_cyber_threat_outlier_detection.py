# cyber_threat_outlier_detection.py
# Análise de Detecção de Outliers em Cybersecurity
# Dataset: Text-Based Cyber Threat Detection

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)

warnings.filterwarnings('ignore')
np.random.seed(42)

# Configurações
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output_images')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_plot(filename):
    """Salva o gráfico atual"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"[SALVO] {filename}")

def load_data():
    """Carrega o dataset (sintético para demonstração)"""
    print("="*60)
    print("CARREGAMENTO DO DATASET")
    print("="*60)
    
    # Dataset sintético para demonstração
    # Em produção, substitua por: pd.read_csv('seu_arquivo.csv')
    print("\n[INFO] Gerando dataset sintético para demonstração...")
    
    np.random.seed(42)
    n_normal = 9000
    n_malicious = 1000
    
    # Dados normais
    normal_data = {
        'text_length': np.random.normal(50, 15, n_normal),
        'special_chars': np.random.poisson(2, n_normal),
        'numeric_ratio': np.random.beta(2, 5, n_normal),
        'uppercase_ratio': np.random.beta(2, 5, n_normal),
        'entropy': np.random.normal(3.5, 0.5, n_normal),
        'label': [0] * n_normal  # 0 = normal
    }
    
    # Dados maliciosos (outliers)
    malicious_data = {
        'text_length': np.random.normal(80, 25, n_malicious),
        'special_chars': np.random.poisson(8, n_malicious),
        'numeric_ratio': np.random.beta(5, 2, n_malicious),
        'uppercase_ratio': np.random.beta(5, 2, n_malicious),
        'entropy': np.random.normal(4.5, 0.8, n_malicious),
        'label': [1] * n_malicious  # 1 = malicious
    }
    
    # Combinar
    df_normal = pd.DataFrame(normal_data)
    df_malicious = pd.DataFrame(malicious_data)
    df = pd.concat([df_normal, df_malicious], ignore_index=True)
    
    # Embaralhar
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n[OK] Dataset carregado")
    print(f"Total de amostras: {len(df)}")
    print(f"Normal: {(df['label']==0).sum()} ({(df['label']==0).sum()/len(df)*100:.1f}%)")
    print(f"Malicioso: {(df['label']==1).sum()} ({(df['label']==1).sum()/len(df)*100:.1f}%)")
    
    return df

def exploratory_analysis(df):
    """Análise exploratória dos dados"""
    print("\n" + "="*60)
    print("ANÁLISE EXPLORATÓRIA")
    print("="*60)
    
    print("\nEstatísticas descritivas:")
    print(df.describe())
    
    # Visualizações
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    features = ['text_length', 'special_chars', 'numeric_ratio', 
                'uppercase_ratio', 'entropy']
    
    for idx, feature in enumerate(features):
        ax = axes[idx // 3, idx % 3]
        
        # Histograma por classe
        df[df['label']==0][feature].hist(ax=ax, alpha=0.6, label='Normal', bins=30)
        df[df['label']==1][feature].hist(ax=ax, alpha=0.6, label='Malicioso', bins=30)
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequência')
        ax.set_title(f'Distribuição: {feature}')
        ax.legend()
    
    # Correlation matrix
    ax = axes[1, 2]
    corr = df[features].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    ax.set_title('Matriz de Correlação')
    
    plt.tight_layout()
    save_plot('01_exploratory_analysis.png')
    plt.close()
    
    print("\n[OK] Análise exploratória concluída")

def apply_outlier_detection(df):
    """Aplica múltiplas técnicas de detecção de outliers"""
    print("\n" + "="*60)
    print("DETECÇÃO DE OUTLIERS - MÚLTIPLAS TÉCNICAS")
    print("="*60)
    
    # Preparar dados
    features = ['text_length', 'special_chars', 'numeric_ratio', 
                'uppercase_ratio', 'entropy']
    X = df[features].values
    y_true = df['label'].values
    
    # Normalizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dicionário para armazenar resultados
    results = {}
    predictions = {}
    
    # 1. Isolation Forest
    print("\n[1/5] Isolation Forest...")
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    pred_iso = iso_forest.fit_predict(X_scaled)
    pred_iso = np.where(pred_iso == -1, 1, 0)  # Converter: -1 (outlier) -> 1
    predictions['Isolation Forest'] = pred_iso
    
    # 2. Local Outlier Factor
    print("[2/5] Local Outlier Factor...")
    lof = LocalOutlierFactor(contamination=0.1, novelty=False)
    pred_lof = lof.fit_predict(X_scaled)
    pred_lof = np.where(pred_lof == -1, 1, 0)
    predictions['LOF'] = pred_lof
    
    # 3. One-Class SVM
    print("[3/5] One-Class SVM...")
    ocsvm = OneClassSVM(nu=0.1, gamma='scale')
    pred_ocsvm = ocsvm.fit_predict(X_scaled)
    pred_ocsvm = np.where(pred_ocsvm == -1, 1, 0)
    predictions['One-Class SVM'] = pred_ocsvm
    
    # 4. Elliptic Envelope
    print("[4/5] Elliptic Envelope...")
    ee = EllipticEnvelope(contamination=0.1, random_state=42)
    pred_ee = ee.fit_predict(X_scaled)
    pred_ee = np.where(pred_ee == -1, 1, 0)
    predictions['Elliptic Envelope'] = pred_ee
    
    # 5. DBSCAN (outliers como noise)
    print("[5/5] DBSCAN...")
    dbscan = DBSCAN(eps=0.5, min_samples=50)
    clusters = dbscan.fit_predict(X_scaled)
    pred_dbscan = np.where(clusters == -1, 1, 0)  # Noise = outlier
    predictions['DBSCAN'] = pred_dbscan
    
    print("\n[OK] Todas as técnicas aplicadas")
    
    return predictions, y_true, X_scaled

def evaluate_methods(predictions, y_true):
    """Avalia a performance de cada método"""
    print("\n" + "="*60)
    print("AVALIAÇÃO DOS MÉTODOS")
    print("="*60)
    
    results = []
    
    for method, y_pred in predictions.items():
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # Contagem de outliers detectados
        n_outliers_detected = np.sum(y_pred == 1)
        n_outliers_real = np.sum(y_true == 1)
        
        results.append({
            'Método': method,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Outliers Detectados': n_outliers_detected,
            'Outliers Reais': n_outliers_real
        })
        
        print(f"\n{method}:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  Outliers detectados: {n_outliers_detected} / {n_outliers_real}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'evaluation_metrics.csv'), index=False)
    print("\n[OK] Métricas salvas em: evaluation_metrics.csv")
    
    return results_df

def visualize_results(results_df):
    """Visualiza os resultados comparativos"""
    print("\n" + "="*60)
    print("VISUALIZAÇÃO DOS RESULTADOS")
    print("="*60)
    
    # Gráfico 1: Métricas comparativas
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        bars = ax.bar(results_df['Método'], results_df[metric])
        ax.set_title(f'{metric} por Método de Detecção', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric, fontsize=12)
        ax.set_xticklabels(results_df['Método'], rotation=45, ha='right')
        ax.axhline(y=0.70, color='red', linestyle='--', alpha=0.7, label='Meta 70%')
        ax.set_ylim(0, 1.1)
        
        # Colorir barras
        for bar, value in zip(bars, results_df[metric]):
            bar.set_color('green' if value >= 0.70 else 'orange')
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax.legend()
    
    plt.tight_layout()
    save_plot('02_comparative_metrics.png')
    plt.close()
    
    # Gráfico 2: Outliers detectados vs reais
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(results_df))
    width = 0.35
    
    ax.bar(x - width/2, results_df['Outliers Reais'], width, label='Outliers Reais', color='red', alpha=0.7)
    ax.bar(x + width/2, results_df['Outliers Detectados'], width, label='Outliers Detectados', color='blue', alpha=0.7)
    
    ax.set_xlabel('Método', fontsize=12)
    ax.set_ylabel('Quantidade', fontsize=12)
    ax.set_title('Outliers Detectados vs Outliers Reais', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Método'], rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    save_plot('03_outliers_comparison.png')
    plt.close()
    
    print("\n[OK] Visualizações salvas")

def visualize_confusion_matrices(predictions, y_true):
    """Gera matrizes de confusão para cada método"""
    print("\n" + "="*60)
    print("MATRIZES DE CONFUSÃO")
    print("="*60)
    
    n_methods = len(predictions)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for idx, (method, y_pred) in enumerate(predictions.items()):
        cm = confusion_matrix(y_true, y_pred)
        
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{method}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predito')
        ax.set_ylabel('Real')
        ax.set_xticklabels(['Normal', 'Malicioso'])
        ax.set_yticklabels(['Normal', 'Malicioso'])
    
    # Remover eixo extra
    if n_methods < 6:
        for idx in range(n_methods, 6):
            axes[idx].axis('off')
    
    plt.tight_layout()
    save_plot('04_confusion_matrices.png')
    plt.close()
    
    print("\n[OK] Matrizes de confusão salvas")

def generate_final_report(results_df):
    """Gera relatório final da análise"""
    print("\n" + "="*70)
    print("RELATÓRIO FINAL - DETECÇÃO DE OUTLIERS EM CYBERSECURITY")
    print("="*70)
    
    print("\nDATASET:")
    print("  Fonte: Text-Based Cyber Threat Detection (sintético para demo)")
    print("  Total de amostras: 10.000")
    print("  Normal: 9.000 (90%)")
    print("  Malicioso: 1.000 (10%)")
    
    print("\nMÉTODOS AVALIADOS:")
    for idx, method in enumerate(results_df['Método'], 1):
        print(f"  {idx}. {method}")
    
    print("\nRESULTADOS:")
    
    # Melhor método por métrica
    best_accuracy = results_df.loc[results_df['Accuracy'].idxmax()]
    best_precision = results_df.loc[results_df['Precision'].idxmax()]
    best_recall = results_df.loc[results_df['Recall'].idxmax()]
    best_f1 = results_df.loc[results_df['F1-Score'].idxmax()]
    
    print(f"\n  Melhor Accuracy:  {best_accuracy['Método']} ({best_accuracy['Accuracy']:.4f})")
    print(f"  Melhor Precision: {best_precision['Método']} ({best_precision['Precision']:.4f})")
    print(f"  Melhor Recall:    {best_recall['Método']} ({best_recall['Recall']:.4f})")
    print(f"  Melhor F1-Score:  {best_f1['Método']} ({best_f1['F1-Score']:.4f})")
    
    # Análise geral
    avg_metrics = results_df[['Accuracy', 'Precision', 'Recall', 'F1-Score']].mean()
    print(f"\n  Média Accuracy:  {avg_metrics['Accuracy']:.4f}")
    print(f"  Média Precision: {avg_metrics['Precision']:.4f}")
    print(f"  Média Recall:    {avg_metrics['Recall']:.4f}")
    print(f"  Média F1-Score:  {avg_metrics['F1-Score']:.4f}")
    
    # Métodos acima da meta
    above_threshold = (results_df['F1-Score'] >= 0.70).sum()
    print(f"\n  Métodos com F1-Score >= 70%: {above_threshold}/{len(results_df)}")
    
    print("\nCONCLUSÕES:")
    print("  1. Os métodos de detecção de outliers podem identificar ameaças cibernéticas")
    print("  2. A performance varia significativamente entre diferentes técnicas")
    print("  3. É recomendado usar ensemble de métodos para maior robustez")
    print("  4. A calibração de hiperparâmetros é crucial para performance ótima")
    
    print("\nARQUIVOS GERADOS:")
    print(f"  - 01_exploratory_analysis.png")
    print(f"  - 02_comparative_metrics.png")
    print(f"  - 03_outliers_comparison.png")
    print(f"  - 04_confusion_matrices.png")
    print(f"  - evaluation_metrics.csv")
    
    print(f"\n  Diretório: {OUTPUT_DIR}")
    print("\n" + "="*70)

def main():
    """Função principal"""
    print("\n")
    print("="*70)
    print("ANÁLISE DE DETECÇÃO DE OUTLIERS EM CYBERSECURITY")
    print("Projeto: Mitigação de Ataques em Aprendizado Federado")
    print("Instituição: Faculdade Impacta")
    print("="*70)
    
    # 1. Carregar dados
    df = load_data()
    
    # 2. Análise exploratória
    exploratory_analysis(df)
    
    # 3. Aplicar detecção de outliers
    predictions, y_true, X_scaled = apply_outlier_detection(df)
    
    # 4. Avaliar métodos
    results_df = evaluate_methods(predictions, y_true)
    
    # 5. Visualizar resultados
    visualize_results(results_df)
    
    # 6. Matrizes de confusão
    visualize_confusion_matrices(predictions, y_true)
    
    # 7. Relatório final
    generate_final_report(results_df)
    
    print("\n[CONCLUÍDO] Análise finalizada com sucesso!")

if __name__ == "__main__":
    main()
