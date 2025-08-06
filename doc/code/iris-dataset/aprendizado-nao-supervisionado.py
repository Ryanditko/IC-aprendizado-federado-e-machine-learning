# iris_unsupervised.py - Aprendizado Não Supervisionado com Dataset Iris

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import TSNE

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
    
    print("❌ Não foi possível carregar o dataset!")
    return None

def explore_data(data):
    """Análise exploratória dos dados"""
    print("\n" + "="*50)
    print("📊 ANÁLISE EXPLORATÓRIA DOS DADOS")
    print("="*50)
    
    print(f"\n📋 Informações gerais:")
    print(f"• Número de amostras: {data.shape[0]}")
    print(f"• Número de features: {data.shape[1] - 1}")  # -1 para excluir a coluna target
    
    print(f"\n📈 Estatísticas descritivas:")
    features = data.select_dtypes(include=[np.number])
    print(features.describe())
    
    print(f"\n🎯 Distribuição das classes (para referência):")
    if 'species' in data.columns:
        print(data['species'].value_counts())
    
    # Verificar valores missing
    print(f"\n❓ Valores faltantes:")
    missing = data.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("Não há valores faltantes!")

def visualize_data(data):
    """Visualizações dos dados"""
    print("\n" + "="*50)
    print("📊 VISUALIZAÇÕES DOS DADOS")
    print("="*50)
    
    # Separar features numéricas
    features = data.select_dtypes(include=[np.number])
    
    # 1. Matriz de correlação
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    correlation = features.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('Matriz de Correlação')
    
    # 2. Distribuição das features
    plt.subplot(2, 2, 2)
    features.hist(bins=20, alpha=0.7)
    plt.suptitle('Distribuição das Features')
    
    # 3. Boxplot
    plt.subplot(2, 2, 3)
    features.boxplot()
    plt.xticks(rotation=45)
    plt.title('Boxplot das Features')
    
    # 4. Pairplot (apenas se species existir)
    if 'species' in data.columns:
        plt.subplot(2, 2, 4)
        # Para o subplot, vamos fazer um scatter simples
        plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=pd.Categorical(data['species']).codes, alpha=0.7)
        plt.xlabel(data.columns[0])
        plt.ylabel(data.columns[1])
        plt.title('Scatter das 2 primeiras features')
    
    plt.tight_layout()
    plt.show()
    
    # Pairplot separado (mais detalhado)
    if 'species' in data.columns:
        plt.figure(figsize=(12, 10))
        sns.pairplot(data, hue='species', diag_kind='hist')
        plt.suptitle('Pairplot com Classes Verdadeiras', y=1.02)
        plt.show()

def perform_pca(data):
    """Análise de Componentes Principais"""
    print("\n" + "="*50)
    print("🔍 ANÁLISE DE COMPONENTES PRINCIPAIS (PCA)")
    print("="*50)
    
    # Preparar dados (apenas features numéricas)
    features = data.select_dtypes(include=[np.number])
    
    # Padronizar os dados
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # PCA
    pca = PCA()
    pca_result = pca.fit_transform(features_scaled)
    
    # Variância explicada
    variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_ratio)
    
    print(f"📊 Variância explicada por componente:")
    for i, var in enumerate(variance_ratio):
        print(f"• PC{i+1}: {var:.3f} ({var*100:.1f}%)")
    
    print(f"\n📈 Variância acumulada:")
    for i, cum_var in enumerate(cumulative_variance):
        print(f"• PC1-PC{i+1}: {cum_var:.3f} ({cum_var*100:.1f}%)")
    
    # Visualizações
    plt.figure(figsize=(15, 5))
    
    # 1. Variância explicada
    plt.subplot(1, 3, 1)
    plt.bar(range(1, len(variance_ratio) + 1), variance_ratio)
    plt.xlabel('Componentes Principais')
    plt.ylabel('Variância Explicada')
    plt.title('Variância Explicada por Componente')
    
    # 2. Variância acumulada
    plt.subplot(1, 3, 2)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95%')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Variância Acumulada')
    plt.title('Variância Acumulada')
    plt.legend()
    
    # 3. Projeção 2D
    plt.subplot(1, 3, 3)
    if 'species' in data.columns:
        colors = pd.Categorical(data['species']).codes
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=colors, alpha=0.7)
        plt.colorbar(scatter, label='Espécies')
    else:
        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
    plt.xlabel(f'PC1 ({variance_ratio[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({variance_ratio[1]*100:.1f}%)')
    plt.title('Projeção PCA 2D')
    
    plt.tight_layout()
    plt.show()
    
    return pca_result, pca, scaler

def kmeans_clustering(data, features_scaled, max_clusters=10):
    """Clustering K-Means"""
    print("\n" + "="*50)
    print("🎯 CLUSTERING K-MEANS")
    print("="*50)
    
    # Método do cotovelo
    inertias = []
    silhouette_scores = []
    K_range = range(2, max_clusters + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(features_scaled, kmeans.labels_))
    
    # Encontrar o melhor k pelo silhouette score
    best_k = K_range[np.argmax(silhouette_scores)]
    print(f"🏆 Melhor número de clusters (Silhouette): {best_k}")
    
    # Visualizar métricas
    plt.figure(figsize=(15, 5))
    
    # Método do cotovelo
    plt.subplot(1, 3, 1)
    plt.plot(K_range, inertias, 'bo-')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inércia')
    plt.title('Método do Cotovelo')
    
    # Silhouette score
    plt.subplot(1, 3, 2)
    plt.plot(K_range, silhouette_scores, 'ro-')
    plt.axvline(x=best_k, color='g', linestyle='--', label=f'Melhor k={best_k}')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score')
    plt.legend()
    
    # Clustering final
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    clusters = kmeans_final.fit_predict(features_scaled)
    
    # PCA para visualização
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_scaled)
    
    plt.subplot(1, 3, 3)
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    plt.scatter(pca.transform(kmeans_final.cluster_centers_)[:, 0], 
                pca.transform(kmeans_final.cluster_centers_)[:, 1], 
                c='red', marker='x', s=200, linewidths=3, label='Centroides')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'K-Means Clustering (k={best_k})')
    plt.colorbar(scatter, label='Clusters')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Avaliar qualidade se temos labels verdadeiros
    if 'species' in data.columns:
        true_labels = pd.Categorical(data['species']).codes
        ari = adjusted_rand_score(true_labels, clusters)
        nmi = normalized_mutual_info_score(true_labels, clusters)
        print(f"\n📊 Métricas de avaliação:")
        print(f"• Adjusted Rand Index: {ari:.3f}")
        print(f"• Normalized Mutual Information: {nmi:.3f}")
        print(f"• Silhouette Score: {silhouette_scores[best_k-2]:.3f}")
    
    return clusters, kmeans_final

def hierarchical_clustering(data, features_scaled):
    """Clustering Hierárquico"""
    print("\n" + "="*50)
    print("🌳 CLUSTERING HIERÁRQUICO")
    print("="*50)
    
    # Calcular linkage
    linkage_matrix = linkage(features_scaled, method='ward')
    
    plt.figure(figsize=(15, 10))
    
    # Dendrograma
    plt.subplot(2, 2, 1)
    dendrogram(linkage_matrix, truncate_mode='level', p=5)
    plt.title('Dendrograma')
    plt.xlabel('Amostras')
    plt.ylabel('Distância')
    
    # Testar diferentes números de clusters
    silhouette_scores = []
    cluster_range = range(2, 8)
    
    for n_clusters in cluster_range:
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        clusters = hierarchical.fit_predict(features_scaled)
        silhouette_scores.append(silhouette_score(features_scaled, clusters))
    
    best_n = cluster_range[np.argmax(silhouette_scores)]
    print(f"🏆 Melhor número de clusters: {best_n}")
    
    # Silhouette scores
    plt.subplot(2, 2, 2)
    plt.plot(cluster_range, silhouette_scores, 'go-')
    plt.axvline(x=best_n, color='r', linestyle='--', label=f'Melhor n={best_n}')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score - Hierárquico')
    plt.legend()
    
    # Clustering final
    hierarchical_final = AgglomerativeClustering(n_clusters=best_n)
    clusters = hierarchical_final.fit_predict(features_scaled)
    
    # Visualização 2D (PCA)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_scaled)
    
    plt.subplot(2, 2, 3)
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='plasma', alpha=0.7)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'Clustering Hierárquico (n={best_n})')
    plt.colorbar(scatter, label='Clusters')
    
    # Comparação com classes verdadeiras (se disponível)
    if 'species' in data.columns:
        true_labels = pd.Categorical(data['species']).codes
        plt.subplot(2, 2, 4)
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=true_labels, cmap='Set1', alpha=0.7)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Classes Verdadeiras')
        plt.colorbar(scatter, label='Espécies')
        
        # Métricas
        ari = adjusted_rand_score(true_labels, clusters)
        nmi = normalized_mutual_info_score(true_labels, clusters)
        print(f"\n📊 Métricas de avaliação:")
        print(f"• Adjusted Rand Index: {ari:.3f}")
        print(f"• Normalized Mutual Information: {nmi:.3f}")
        print(f"• Silhouette Score: {max(silhouette_scores):.3f}")
    
    plt.tight_layout()
    plt.show()
    
    return clusters, hierarchical_final

def dbscan_clustering(data, features_scaled):
    """Clustering DBSCAN"""
    print("\n" + "="*50)
    print("🔍 CLUSTERING DBSCAN")
    print("="*50)
    
    # Testar diferentes valores de eps
    eps_values = np.arange(0.1, 2.0, 0.1)
    silhouette_scores = []
    n_clusters_list = []
    n_noise_list = []
    
    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=5)
        clusters = dbscan.fit_predict(features_scaled)
        
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = list(clusters).count(-1)
        
        n_clusters_list.append(n_clusters)
        n_noise_list.append(n_noise)
        
        if n_clusters > 1:
            # Calcular silhouette score apenas para pontos não-noise
            non_noise_mask = clusters != -1
            if np.sum(non_noise_mask) > 1:
                score = silhouette_score(features_scaled[non_noise_mask], clusters[non_noise_mask])
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(-1)
        else:
            silhouette_scores.append(-1)
    
    # Encontrar melhor eps
    valid_scores = [s for s in silhouette_scores if s > -1]
    if valid_scores:
        best_idx = np.argmax(silhouette_scores)
        best_eps = eps_values[best_idx]
        print(f"🏆 Melhor eps: {best_eps:.2f}")
    else:
        best_eps = 0.5
        print(f"⚠️ Usando eps padrão: {best_eps}")
    
    # Visualizar métricas
    plt.figure(figsize=(15, 10))
    
    # Número de clusters vs eps
    plt.subplot(2, 3, 1)
    plt.plot(eps_values, n_clusters_list, 'b-')
    plt.axvline(x=best_eps, color='r', linestyle='--', label=f'Melhor eps={best_eps:.2f}')
    plt.xlabel('eps')
    plt.ylabel('Número de Clusters')
    plt.title('Número de Clusters vs eps')
    plt.legend()
    
    # Número de pontos noise vs eps
    plt.subplot(2, 3, 2)
    plt.plot(eps_values, n_noise_list, 'g-')
    plt.axvline(x=best_eps, color='r', linestyle='--', label=f'Melhor eps={best_eps:.2f}')
    plt.xlabel('eps')
    plt.ylabel('Pontos Noise')
    plt.title('Pontos Noise vs eps')
    plt.legend()
    
    # Silhouette score vs eps
    plt.subplot(2, 3, 3)
    plt.plot(eps_values, silhouette_scores, 'r-')
    plt.axvline(x=best_eps, color='r', linestyle='--', label=f'Melhor eps={best_eps:.2f}')
    plt.xlabel('eps')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs eps')
    plt.legend()
    
    # Clustering final
    dbscan_final = DBSCAN(eps=best_eps, min_samples=5)
    clusters = dbscan_final.fit_predict(features_scaled)
    
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    
    print(f"\n📊 Resultados finais:")
    print(f"• Número de clusters: {n_clusters}")
    print(f"• Pontos noise: {n_noise}")
    print(f"• Porcentagem noise: {n_noise/len(clusters)*100:.1f}%")
    
    # Visualização 2D
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_scaled)
    
    plt.subplot(2, 3, 4)
    unique_labels = set(clusters)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Pontos noise em preto
            col = 'black'
            marker = 'x'
        else:
            marker = 'o'
        
        class_member_mask = (clusters == k)
        xy = pca_result[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker, alpha=0.7, s=50)
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'DBSCAN (eps={best_eps:.2f})')
    
    # Comparação com classes verdadeiras
    if 'species' in data.columns:
        true_labels = pd.Categorical(data['species']).codes
        plt.subplot(2, 3, 5)
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=true_labels, cmap='Set1', alpha=0.7)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Classes Verdadeiras')
        plt.colorbar(scatter, label='Espécies')
        
        # Métricas (excluindo pontos noise)
        non_noise_mask = clusters != -1
        if np.sum(non_noise_mask) > 0:
            ari = adjusted_rand_score(true_labels[non_noise_mask], clusters[non_noise_mask])
            nmi = normalized_mutual_info_score(true_labels[non_noise_mask], clusters[non_noise_mask])
            
            if len(set(clusters[non_noise_mask])) > 1:
                sil_score = silhouette_score(features_scaled[non_noise_mask], clusters[non_noise_mask])
            else:
                sil_score = -1
            
            print(f"\n📊 Métricas de avaliação (sem noise):")
            print(f"• Adjusted Rand Index: {ari:.3f}")
            print(f"• Normalized Mutual Information: {nmi:.3f}")
            print(f"• Silhouette Score: {sil_score:.3f}")
    
    plt.tight_layout()
    plt.show()
    
    return clusters, dbscan_final

def compare_algorithms(data, features_scaled):
    """Comparar todos os algoritmos de clustering"""
    print("\n" + "="*50)
    print("🏆 COMPARAÇÃO DE ALGORITMOS")
    print("="*50)
    
    # Executar todos os algoritmos
    algorithms = {}
    
    # K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans_clusters = kmeans.fit_predict(features_scaled)
    algorithms['K-Means'] = kmeans_clusters
    
    # Hierárquico
    hierarchical = AgglomerativeClustering(n_clusters=3)
    hierarchical_clusters = hierarchical.fit_predict(features_scaled)
    algorithms['Hierárquico'] = hierarchical_clusters
    
    # DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_clusters = dbscan.fit_predict(features_scaled)
    algorithms['DBSCAN'] = dbscan_clusters
    
    # Visualizações comparativas
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_scaled)
    
    plt.figure(figsize=(15, 10))
    
    for i, (name, clusters) in enumerate(algorithms.items(), 1):
        plt.subplot(2, 3, i)
        if name == 'DBSCAN':
            # Tratamento especial para DBSCAN (pontos noise)
            unique_labels = set(clusters)
            colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
            
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    col = 'black'
                    marker = 'x'
                else:
                    marker = 'o'
                
                class_member_mask = (clusters == k)
                xy = pca_result[class_member_mask]
                plt.scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker, alpha=0.7)
        else:
            scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, alpha=0.7)
            plt.colorbar(scatter, label='Clusters')
        
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title(name)
    
    # Classes verdadeiras
    if 'species' in data.columns:
        true_labels = pd.Categorical(data['species']).codes
        plt.subplot(2, 3, 4)
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=true_labels, cmap='Set1', alpha=0.7)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Classes Verdadeiras')
        plt.colorbar(scatter, label='Espécies')
        
        # Tabela de métricas
        print(f"\n📊 Tabela de Métricas:")
        print(f"{'Algoritmo':<15} {'ARI':<6} {'NMI':<6} {'Silhouette':<10}")
        print("-" * 45)
        
        for name, clusters in algorithms.items():
            if name == 'DBSCAN':
                # Para DBSCAN, considerar apenas pontos não-noise
                non_noise_mask = clusters != -1
                if np.sum(non_noise_mask) > 0:
                    ari = adjusted_rand_score(true_labels[non_noise_mask], clusters[non_noise_mask])
                    nmi = normalized_mutual_info_score(true_labels[non_noise_mask], clusters[non_noise_mask])
                    if len(set(clusters[non_noise_mask])) > 1:
                        sil = silhouette_score(features_scaled[non_noise_mask], clusters[non_noise_mask])
                    else:
                        sil = -1
                else:
                    ari, nmi, sil = -1, -1, -1
            else:
                ari = adjusted_rand_score(true_labels, clusters)
                nmi = normalized_mutual_info_score(true_labels, clusters)
                if len(set(clusters)) > 1:
                    sil = silhouette_score(features_scaled, clusters)
                else:
                    sil = -1
            
            print(f"{name:<15} {ari:<6.3f} {nmi:<6.3f} {sil:<10.3f}")
    
    plt.tight_layout()
    plt.show()

def main():
    """Função principal"""
    print("🌸 ANÁLISE DE CLUSTERING - DATASET IRIS")
    print("="*60)
    
    # 1. Carregar dados
    data = load_iris_dataset()
    if data is None:
        print("❌ Não foi possível carregar o dataset!")
        return
    
    print(f"✅ Dataset carregado com sucesso!")
    print(f"📊 Shape: {data.shape}")
    
    # 2. Análise exploratória
    explore_data(data)
    
    # 3. Visualizações
    visualize_data(data)
    
    # 4. Preparar dados para clustering
    features = data.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 5. PCA
    pca_result, pca_model, scaler_model = perform_pca(data)
    
    # 6. K-Means
    kmeans_clusters, kmeans_model = kmeans_clustering(data, features_scaled)
    
    # 7. Clustering Hierárquico
    hierarchical_clusters, hierarchical_model = hierarchical_clustering(data, features_scaled)
    
    # 8. DBSCAN
    dbscan_clusters, dbscan_model = dbscan_clustering(data, features_scaled)
    
    # 9. Comparação
    compare_algorithms(data, features_scaled)
    
    print("\n" + "="*60)
    print("🎉 ANÁLISE COMPLETA!")
    print("="*60)
    print("\n📋 Resumo dos resultados:")
    print("• PCA: Redução de dimensionalidade realizada")
    print("• K-Means: Clustering particional executado")
    print("• Hierárquico: Clustering hierárquico executado")
    print("• DBSCAN: Clustering baseado em densidade executado")
    print("• Comparação: Todos os algoritmos foram comparados")
    
    if 'species' in data.columns:
        print("\n💡 Dicas para interpretação:")
        print("• ARI próximo de 1: clustering muito similar às classes verdadeiras")
        print("• NMI próximo de 1: alta informação mútua entre clusters e classes")
        print("• Silhouette próximo de 1: clusters bem definidos e separados")

if __name__ == "__main__":
    main()
