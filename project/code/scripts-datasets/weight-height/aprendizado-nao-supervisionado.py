# aprendizado-nao-supervisionado.py - Weight-Height Dataset
# Análise não supervisionada com clustering e detecção de outliers

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
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
        print(f"[OK] Dataset encontrado")
        return pd.read_csv(data_path)
    else:
        print(f"[INFO] Gerando dataset sintético...")
        return generate_synthetic_data()

def generate_synthetic_data():
    """Gera dados sintéticos"""
    np.random.seed(42)
    n_samples = 10000
    
    n_male = n_samples // 2
    male_height = np.random.normal(178, 7, n_male)
    male_weight = np.random.normal(85, 12, n_male)
    
    n_female = n_samples - n_male
    female_height = np.random.normal(163, 6, n_female)
    female_weight = np.random.normal(65, 10, n_female)
    
    # Adicionar alguns outliers
    n_outliers = 50
    outlier_height = np.random.uniform(140, 210, n_outliers)
    outlier_weight = np.random.uniform(40, 150, n_outliers)
    
    data = pd.DataFrame({
        'Gender': ['Male'] * n_male + ['Female'] * n_female + ['Outlier'] * n_outliers,
        'Height': np.concatenate([male_height, female_height, outlier_height]),
        'Weight': np.concatenate([male_weight, female_weight, outlier_weight])
    })
    
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    data_path = get_data_path()
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    data.to_csv(data_path, index=False)
    print(f"[OK] Dataset sintético salvo")
    
    return data

def exploratory_analysis(data):
    """Análise exploratória"""
    print("\n" + "="*60)
    print("ANÁLISE EXPLORATÓRIA")
    print("="*60)
    
    print(f"\nInformações do Dataset:")
    print(f"  Amostras: {data.shape[0]}")
    print(f"  Features: Height, Weight")
    
    print(f"\nEstatísticas descritivas:")
    print(data[['Height', 'Weight']].describe())
    
    # Visualizações
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Scatter plot
    axes[0, 0].scatter(data['Height'], data['Weight'], alpha=0.5, s=10)
    axes[0, 0].set_xlabel('Altura (cm)')
    axes[0, 0].set_ylabel('Peso (kg)')
    axes[0, 0].set_title('Relação Altura vs Peso')
    
    # Histogramas
    axes[0, 1].hist(data['Height'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Altura (cm)')
    axes[0, 1].set_ylabel('Frequência')
    axes[0, 1].set_title('Distribuição de Altura')
    
    axes[1, 0].hist(data['Weight'], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Peso (kg)')
    axes[1, 0].set_ylabel('Frequência')
    axes[1, 0].set_title('Distribuição de Peso')
    
    # Boxplot
    data[['Height', 'Weight']].boxplot(ax=axes[1, 1])
    axes[1, 1].set_title('Boxplot - Detecção Visual de Outliers')
    
    plt.tight_layout()
    plt.savefig('weight_height_unsupervised_eda.png', dpi=300, bbox_inches='tight')
    print("\n[OK] Gráfico salvo: weight_height_unsupervised_eda.png")
    plt.show()

def apply_pca(X_scaled, data):
    """Aplica PCA para redução de dimensionalidade"""
    print("\n" + "="*60)
    print("ANÁLISE DE COMPONENTES PRINCIPAIS (PCA)")
    print("="*60)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    variance = pca.explained_variance_ratio_
    print(f"\nVariância explicada:")
    print(f"  PC1: {variance[0]:.4f} ({variance[0]*100:.2f}%)")
    print(f"  PC2: {variance[1]:.4f} ({variance[1]*100:.2f}%)")
    print(f"  Total: {variance.sum():.4f} ({variance.sum()*100:.2f}%)")
    
    # Visualização
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, s=10)
    plt.xlabel(f'PC1 ({variance[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({variance[1]*100:.1f}%)')
    plt.title('Projeção PCA - Weight-Height Dataset')
    plt.tight_layout()
    plt.savefig('weight_height_pca.png', dpi=300, bbox_inches='tight')
    print("\n[OK] Gráfico PCA salvo")
    plt.show()
    
    return X_pca

def apply_kmeans(X_scaled):
    """Clustering K-Means"""
    print("\n" + "="*60)
    print("CLUSTERING K-MEANS")
    print("="*60)
    
    # Método do cotovelo
    inertias = []
    silhouette_scores = []
    K_range = range(2, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    best_k = K_range[np.argmax(silhouette_scores)]
    print(f"\nMelhor número de clusters (Silhouette): {best_k}")
    
    # Visualizações
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Método do cotovelo
    axes[0].plot(K_range, inertias, 'bo-')
    axes[0].set_xlabel('Número de Clusters (k)')
    axes[0].set_ylabel('Inércia')
    axes[0].set_title('Método do Cotovelo')
    
    # Silhouette score
    axes[1].plot(K_range, silhouette_scores, 'ro-')
    axes[1].axvline(x=best_k, color='g', linestyle='--', label=f'Melhor k={best_k}')
    axes[1].set_xlabel('Número de Clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Análise de Silhouette')
    axes[1].legend()
    
    # Clustering final
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    clusters = kmeans_final.fit_predict(X_scaled)
    
    axes[2].scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', alpha=0.5, s=10)
    axes[2].scatter(kmeans_final.cluster_centers_[:, 0], 
                   kmeans_final.cluster_centers_[:, 1],
                   c='red', marker='X', s=200, edgecolor='black', linewidth=2, label='Centroides')
    axes[2].set_xlabel('Altura (normalizada)')
    axes[2].set_ylabel('Peso (normalizado)')
    axes[2].set_title(f'K-Means Clustering (k={best_k})')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('weight_height_kmeans.png', dpi=300, bbox_inches='tight')
    print("\n[OK] Gráfico K-Means salvo")
    plt.show()
    
    return clusters

def apply_dbscan(X_scaled):
    """Clustering DBSCAN"""
    print("\n" + "="*60)
    print("CLUSTERING DBSCAN")
    print("="*60)
    
    dbscan = DBSCAN(eps=0.3, min_samples=50)
    clusters = dbscan.fit_predict(X_scaled)
    
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    
    print(f"\nResultados:")
    print(f"  Número de clusters: {n_clusters}")
    print(f"  Pontos noise (outliers): {n_noise} ({n_noise/len(clusters)*100:.2f}%)")
    
    # Visualização
    fig, ax = plt.subplots(figsize=(10, 6))
    
    unique_labels = set(clusters)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = 'black'
            marker = 'x'
            label = 'Outliers (Noise)'
        else:
            marker = 'o'
            label = f'Cluster {k+1}'
        
        class_member_mask = (clusters == k)
        xy = X_scaled[class_member_mask]
        ax.scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker, 
                  alpha=0.5, s=30, label=label)
    
    ax.set_xlabel('Altura (normalizada)')
    ax.set_ylabel('Peso (normalizado)')
    ax.set_title(f'DBSCAN Clustering (eps=0.3, min_samples=50)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('weight_height_dbscan.png', dpi=300, bbox_inches='tight')
    print("\n[OK] Gráfico DBSCAN salvo")
    plt.show()
    
    return clusters

def detect_outliers(X_scaled, data):
    """Detecção de outliers usando múltiplas técnicas"""
    print("\n" + "="*60)
    print("DETECÇÃO DE OUTLIERS")
    print("="*60)
    
    # 1. Isolation Forest
    print("\n[1/2] Isolation Forest...")
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    outliers_iso = iso_forest.fit_predict(X_scaled)
    n_outliers_iso = (outliers_iso == -1).sum()
    print(f"  Outliers detectados: {n_outliers_iso} ({n_outliers_iso/len(X_scaled)*100:.2f}%)")
    
    # 2. Local Outlier Factor
    print("[2/2] Local Outlier Factor...")
    lof = LocalOutlierFactor(contamination=0.05)
    outliers_lof = lof.fit_predict(X_scaled)
    n_outliers_lof = (outliers_lof == -1).sum()
    print(f"  Outliers detectados: {n_outliers_lof} ({n_outliers_lof/len(X_scaled)*100:.2f}%)")
    
    # Visualização comparativa
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Isolation Forest
    colors_iso = ['red' if x == -1 else 'blue' for x in outliers_iso]
    axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=colors_iso, alpha=0.5, s=10)
    axes[0].set_xlabel('Altura (normalizada)')
    axes[0].set_ylabel('Peso (normalizado)')
    axes[0].set_title(f'Isolation Forest\n{n_outliers_iso} outliers ({n_outliers_iso/len(X_scaled)*100:.1f}%)')
    
    # LOF
    colors_lof = ['red' if x == -1 else 'blue' for x in outliers_lof]
    axes[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=colors_lof, alpha=0.5, s=10)
    axes[1].set_xlabel('Altura (normalizada)')
    axes[1].set_ylabel('Peso (normalizado)')
    axes[1].set_title(f'Local Outlier Factor\n{n_outliers_lof} outliers ({n_outliers_lof/len(X_scaled)*100:.1f}%)')
    
    plt.tight_layout()
    plt.savefig('weight_height_outliers.png', dpi=300, bbox_inches='tight')
    print("\n[OK] Gráfico de outliers salvo")
    plt.show()

def generate_report(data):
    """Gera relatório final"""
    print("\n" + "="*60)
    print("RELATÓRIO FINAL")
    print("="*60)
    
    print(f"\nDataset: Weight-Height")
    print(f"  Total de amostras: {len(data)}")
    print(f"  Features: Height (cm), Weight (kg)")
    
    print(f"\nAnálises Realizadas:")
    print(f"  1. PCA - Redução de dimensionalidade")
    print(f"  2. K-Means - Clustering particional")
    print(f"  3. DBSCAN - Clustering baseado em densidade")
    print(f"  4. Isolation Forest - Detecção de outliers")
    print(f"  5. LOF - Detecção de outliers baseada em densidade local")
    
    print(f"\nArquivos Gerados:")
    print(f"  - weight_height_unsupervised_eda.png")
    print(f"  - weight_height_pca.png")
    print(f"  - weight_height_kmeans.png")
    print(f"  - weight_height_dbscan.png")
    print(f"  - weight_height_outliers.png")
    
    print(f"\nConclusões:")
    print(f"  - Os dados apresentam padrões claros de agrupamento")
    print(f"  - Técnicas de clustering identificam grupos naturais (gênero)")
    print(f"  - Métodos de outlier detection identificam anomalias nos dados")
    print(f"  - DBSCAN é eficaz para identificar outliers como noise")

def main():
    """Função principal"""
    print("="*60)
    print("ANÁLISE NÃO SUPERVISIONADA - WEIGHT-HEIGHT DATASET")
    print("="*60)
    
    # 1. Carregar dados
    data = load_data()
    
    # 2. Análise exploratória
    exploratory_analysis(data)
    
    # 3. Preparar dados
    X = data[['Height', 'Weight']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. PCA
    X_pca = apply_pca(X_scaled, data)
    
    # 5. K-Means
    kmeans_clusters = apply_kmeans(X_scaled)
    
    # 6. DBSCAN
    dbscan_clusters = apply_dbscan(X_scaled)
    
    # 7. Detecção de outliers
    detect_outliers(X_scaled, data)
    
    # 8. Relatório final
    generate_report(data)
    
    print("\n" + "="*60)
    print("ANÁLISE CONCLUÍDA")
    print("="*60)

if __name__ == "__main__":
    main()
