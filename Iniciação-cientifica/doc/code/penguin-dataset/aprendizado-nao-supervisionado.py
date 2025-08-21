# penguin_unsupervised.py - Aprendizado Não Supervisionado com Dataset Penguin

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import TSNE

# Suprimir warnings desnecessários
warnings.filterwarnings('ignore')

def get_penguin_path():
    """Retorna o caminho completo para o arquivo penguins.csv"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "penguins.csv")

def load_penguin_dataset():
    """Carrega o dataset penguin de diferentes fontes"""
    print("🔍 Procurando dataset penguin...")
    
    penguin_path = get_penguin_path()
    
    # Tentar carregar de arquivo local primeiro
    if os.path.exists(penguin_path):
        print("✅ Arquivo local encontrado!")
        return pd.read_csv(penguin_path)
    
    # Tentar via seaborn
    try:
        print("📥 Tentando baixar via seaborn...")
        penguin = sns.load_dataset("penguins")
        if penguin is not None and not penguin.empty:
            # Salvar para uso futuro
            penguin.to_csv(penguin_path, index=False)
            print("✅ Dataset baixado via seaborn e salvo localmente!")
            return penguin
    except Exception as e:
        print(f"❌ Erro no seaborn: {e}")
    
    # Tentar via URL direta
    try:
        print("📥 Tentando baixar via URL...")
        url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv"
        penguin = pd.read_csv(url)
        # Salvar para uso futuro
        penguin.to_csv(penguin_path, index=False)
        print("✅ Dataset baixado via URL e salvo localmente!")
        return penguin
    except Exception as e:
        print(f"❌ Erro no download: {e}")
    
    print("❌ Não foi possível carregar o dataset!")
    return None

def clean_and_prepare_data(data):
    """Limpa e prepara os dados para análise"""
    print("\n" + "="*50)
    print("🧹 LIMPEZA E PREPARAÇÃO DOS DADOS")
    print("="*50)
    
    print(f"📊 Dados originais: {data.shape}")
    print(f"❓ Valores faltantes por coluna:")
    missing_before = data.isnull().sum()
    print(missing_before[missing_before > 0])
    
    # Fazer uma cópia para não modificar o original
    data_clean = data.copy()
    
    # Remover linhas com muitos valores faltantes
    data_clean = data_clean.dropna()
    
    print(f"\n📊 Dados após limpeza: {data_clean.shape}")
    print(f"✅ Removidas {data.shape[0] - data_clean.shape[0]} linhas com valores faltantes")
    
    # Codificar variáveis categóricas se necessário
    categorical_columns = data_clean.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    if len(categorical_columns) > 0:
        print(f"\n🏷️ Codificando variáveis categóricas: {list(categorical_columns)}")
        for col in categorical_columns:
            if col != 'species':  # Manter species para comparação
                le = LabelEncoder()
                data_clean[col] = le.fit_transform(data_clean[col])
                label_encoders[col] = le
    
    return data_clean, label_encoders

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
    
    print(f"\n🎯 Distribuição das espécies (para referência):")
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
    plt.figure(figsize=(15, 12))
    plt.subplot(3, 2, 1)
    correlation = features.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Matriz de Correlação')
    
    # 2. Distribuição das features
    plt.subplot(3, 2, 2)
    features.hist(bins=20, alpha=0.7)
    plt.suptitle('Distribuição das Features')
    
    # 3. Boxplot
    plt.subplot(3, 2, 3)
    features.boxplot()
    plt.xticks(rotation=45)
    plt.title('Boxplot das Features')
    
    # 4. Scatter das primeiras duas features
    plt.subplot(3, 2, 4)
    if 'species' in data.columns:
        colors = pd.Categorical(data['species']).codes
        scatter = plt.scatter(features.iloc[:, 0], features.iloc[:, 1], c=colors, alpha=0.7)
        plt.colorbar(scatter, label='Espécies')
    else:
        plt.scatter(features.iloc[:, 0], features.iloc[:, 1], alpha=0.7)
    plt.xlabel(features.columns[0])
    plt.ylabel(features.columns[1])
    plt.title('Scatter das 2 primeiras features')
    
    # 5. Distribuição por espécie (se disponível)
    if 'species' in data.columns:
        plt.subplot(3, 2, 5)
        data['species'].value_counts().plot(kind='bar')
        plt.title('Distribuição das Espécies')
        plt.xticks(rotation=45)
        
        # 6. Violinplot de uma feature por espécie
        if len(features.columns) > 0:
            plt.subplot(3, 2, 6)
            sns.violinplot(data=data, x='species', y=features.columns[0])
            plt.title(f'Distribuição de {features.columns[0]} por Espécie')
            plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Pairplot separado (mais detalhado)
    if 'species' in data.columns and len(features.columns) <= 6:
        plt.figure(figsize=(15, 12))
        data_for_pair = data[list(features.columns) + ['species']]
        sns.pairplot(data_for_pair, hue='species', diag_kind='hist')
        plt.suptitle('Pairplot com Espécies Verdadeiras', y=1.02)
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
    plt.figure(figsize=(18, 6))
    
    # 1. Variância explicada
    plt.subplot(1, 4, 1)
    plt.bar(range(1, len(variance_ratio) + 1), variance_ratio)
    plt.xlabel('Componentes Principais')
    plt.ylabel('Variância Explicada')
    plt.title('Variância Explicada por Componente')
    
    # 2. Variância acumulada
    plt.subplot(1, 4, 2)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95%')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Variância Acumulada')
    plt.title('Variância Acumulada')
    plt.legend()
    
    # 3. Projeção 2D
    plt.subplot(1, 4, 3)
    if 'species' in data.columns:
        colors = pd.Categorical(data['species']).codes
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=colors, alpha=0.7)
        plt.colorbar(scatter, label='Espécies')
    else:
        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
    plt.xlabel(f'PC1 ({variance_ratio[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({variance_ratio[1]*100:.1f}%)')
    plt.title('Projeção PCA 2D')
    
    # 4. Projeção 3D (se temos pelo menos 3 componentes)
    if len(variance_ratio) >= 3:
        ax = plt.subplot(1, 4, 4, projection='3d')
        if 'species' in data.columns:
            colors = pd.Categorical(data['species']).codes
            scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=colors, alpha=0.7)
        else:
            ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], alpha=0.7)
        ax.set_xlabel(f'PC1 ({variance_ratio[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({variance_ratio[1]*100:.1f}%)')
        ax.set_zlabel(f'PC3 ({variance_ratio[2]*100:.1f}%)')
        plt.title('Projeção PCA 3D')
    
    plt.tight_layout()
    plt.show()
    
    return pca_result, pca, scaler

def kmeans_clustering(data, features_scaled, max_clusters=10):
    """Clustering K-Means"""
    print("\n" + "="*50)
    print("🎯 CLUSTERING K-MEANS")
    print("="*50)
    
    # Método do cotovelo e silhouette
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
    plt.figure(figsize=(18, 6))
    
    # Método do cotovelo
    plt.subplot(1, 4, 1)
    plt.plot(K_range, inertias, 'bo-')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inércia')
    plt.title('Método do Cotovelo')
    
    # Silhouette score
    plt.subplot(1, 4, 2)
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
    
    plt.subplot(1, 4, 3)
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    # Plotar centroides
    centroids_pca = pca.transform(kmeans_final.cluster_centers_)
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
                c='red', marker='x', s=200, linewidths=3, label='Centroides')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'K-Means Clustering (k={best_k})')
    plt.colorbar(scatter, label='Clusters')
    plt.legend()
    
    # Distribuição dos clusters
    plt.subplot(1, 4, 4)
    unique, counts = np.unique(clusters, return_counts=True)
    plt.bar(unique, counts)
    plt.xlabel('Cluster')
    plt.ylabel('Número de Amostras')
    plt.title('Distribuição dos Clusters')
    
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
        
        # Matriz de confusão
        plt.figure(figsize=(8, 6))
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true_labels, clusters)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusão: Classes Verdadeiras vs Clusters')
        plt.xlabel('Clusters Preditos')
        plt.ylabel('Classes Verdadeiras')
        plt.show()
    
    return clusters, kmeans_final

def hierarchical_clustering(data, features_scaled):
    """Clustering Hierárquico"""
    print("\n" + "="*50)
    print("🌳 CLUSTERING HIERÁRQUICO")
    print("="*50)
    
    # Calcular linkage
    linkage_matrix = linkage(features_scaled, method='ward')
    
    plt.figure(figsize=(18, 12))
    
    # Dendrograma
    plt.subplot(3, 2, 1)
    dendrogram(linkage_matrix, truncate_mode='level', p=5)
    plt.title('Dendrograma (Ward)')
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
    plt.subplot(3, 2, 2)
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
    
    plt.subplot(3, 2, 3)
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='plasma', alpha=0.7)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'Clustering Hierárquico (n={best_n})')
    plt.colorbar(scatter, label='Clusters')
    
    # Distribuição dos clusters
    plt.subplot(3, 2, 4)
    unique, counts = np.unique(clusters, return_counts=True)
    plt.bar(unique, counts)
    plt.xlabel('Cluster')
    plt.ylabel('Número de Amostras')
    plt.title('Distribuição dos Clusters')
    
    # Comparação com classes verdadeiras (se disponível)
    if 'species' in data.columns:
        true_labels = pd.Categorical(data['species']).codes
        plt.subplot(3, 2, 5)
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
        
        # Matriz de confusão
        plt.subplot(3, 2, 6)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true_labels, clusters)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
        plt.title('Matriz de Confusão')
        plt.xlabel('Clusters Preditos')
        plt.ylabel('Classes Verdadeiras')
    
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
    plt.figure(figsize=(18, 12))
    
    # Número de clusters vs eps
    plt.subplot(3, 3, 1)
    plt.plot(eps_values, n_clusters_list, 'b-')
    plt.axvline(x=best_eps, color='r', linestyle='--', label=f'Melhor eps={best_eps:.2f}')
    plt.xlabel('eps')
    plt.ylabel('Número de Clusters')
    plt.title('Número de Clusters vs eps')
    plt.legend()
    
    # Número de pontos noise vs eps
    plt.subplot(3, 3, 2)
    plt.plot(eps_values, n_noise_list, 'g-')
    plt.axvline(x=best_eps, color='r', linestyle='--', label=f'Melhor eps={best_eps:.2f}')
    plt.xlabel('eps')
    plt.ylabel('Pontos Noise')
    plt.title('Pontos Noise vs eps')
    plt.legend()
    
    # Silhouette score vs eps
    plt.subplot(3, 3, 3)
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
    
    plt.subplot(3, 3, 4)
    unique_labels = set(clusters)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Pontos noise em preto
            col = 'black'
            marker = 'x'
            label = 'Noise'
        else:
            marker = 'o'
            label = f'Cluster {k}'
        
        class_member_mask = (clusters == k)
        xy = pca_result[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker, alpha=0.7, s=50, label=label)
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'DBSCAN (eps={best_eps:.2f})')
    plt.legend()
    
    # Distribuição dos clusters (excluindo noise)
    plt.subplot(3, 3, 5)
    non_noise_clusters = clusters[clusters != -1]
    if len(non_noise_clusters) > 0:
        unique, counts = np.unique(non_noise_clusters, return_counts=True)
        plt.bar(unique, counts)
        plt.xlabel('Cluster')
        plt.ylabel('Número de Amostras')
        plt.title('Distribuição dos Clusters (sem noise)')
    
    # Comparação com classes verdadeiras
    if 'species' in data.columns:
        true_labels = pd.Categorical(data['species']).codes
        plt.subplot(3, 3, 6)
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
        
        # Matriz de confusão (sem noise)
        if np.sum(non_noise_mask) > 0:
            plt.subplot(3, 3, 7)
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(true_labels[non_noise_mask], clusters[non_noise_mask])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
            plt.title('Matriz de Confusão (sem noise)')
            plt.xlabel('Clusters Preditos')
            plt.ylabel('Classes Verdadeiras')
    
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
    
    plt.figure(figsize=(20, 12))
    
    for i, (name, clusters) in enumerate(algorithms.items(), 1):
        plt.subplot(2, 4, i)
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
        plt.subplot(2, 4, 4)
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=true_labels, cmap='Set1', alpha=0.7)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Classes Verdadeiras')
        plt.colorbar(scatter, label='Espécies')
        
        # Gráfico de barras com métricas
        plt.subplot(2, 4, 5)
        metrics_data = []
        algorithm_names = []
        
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
            
            metrics_data.append([ari, nmi, sil])
            algorithm_names.append(name)
        
        metrics_df = pd.DataFrame(metrics_data, columns=['ARI', 'NMI', 'Silhouette'], index=algorithm_names)
        metrics_df.plot(kind='bar', ax=plt.gca())
        plt.title('Comparação de Métricas')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend()
        
        # Tabela de métricas
        print(f"\n📊 Tabela de Métricas:")
        print(f"{'Algoritmo':<15} {'ARI':<6} {'NMI':<6} {'Silhouette':<10}")
        print("-" * 45)
        
        for name, (ari, nmi, sil) in zip(algorithm_names, metrics_data):
            print(f"{name:<15} {ari:<6.3f} {nmi:<6.3f} {sil:<10.3f}")
    
    plt.tight_layout()
    plt.show()

def analyze_clusters_characteristics(data, clusters, algorithm_name):
    """Analisa as características dos clusters encontrados"""
    print(f"\n📊 ANÁLISE DE CARACTERÍSTICAS - {algorithm_name}")
    print("="*50)
    
    # Adicionar clusters ao dataframe
    data_with_clusters = data.copy()
    data_with_clusters['cluster'] = clusters
    
    # Features numéricas
    numeric_features = data.select_dtypes(include=[np.number])
    
    # Estatísticas por cluster
    print("\n📈 Estatísticas por cluster:")
    cluster_stats = data_with_clusters.groupby('cluster')[numeric_features.columns].mean()
    print(cluster_stats)
    
    # Visualização das características
    if len(numeric_features.columns) <= 6:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, feature in enumerate(numeric_features.columns):
            if i < 6:
                data_with_clusters.boxplot(column=feature, by='cluster', ax=axes[i])
                axes[i].set_title(f'{feature} por Cluster')
                axes[i].set_xlabel('Cluster')
        
        plt.tight_layout()
        plt.show()

def main():
    """Função principal"""
    print("🐧 ANÁLISE DE CLUSTERING - DATASET PENGUIN")
    print("="*60)
    
    # 1. Carregar dados
    data = load_penguin_dataset()
    if data is None:
        print("❌ Não foi possível carregar o dataset!")
        return
    
    print(f"✅ Dataset carregado com sucesso!")
    print(f"📊 Shape: {data.shape}")
    
    # 2. Limpar e preparar dados
    data_clean, label_encoders = clean_and_prepare_data(data)
    
    # 3. Análise exploratória
    explore_data(data_clean)
    
    # 4. Visualizações
    visualize_data(data_clean)
    
    # 5. Preparar dados para clustering
    features = data_clean.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 6. PCA
    pca_result, pca_model, scaler_model = perform_pca(data_clean)
    
    # 7. K-Means
    kmeans_clusters, kmeans_model = kmeans_clustering(data_clean, features_scaled)
    analyze_clusters_characteristics(data_clean, kmeans_clusters, "K-MEANS")
    
    # 8. Clustering Hierárquico
    hierarchical_clusters, hierarchical_model = hierarchical_clustering(data_clean, features_scaled)
    analyze_clusters_characteristics(data_clean, hierarchical_clusters, "HIERÁRQUICO")
    
    # 9. DBSCAN
    dbscan_clusters, dbscan_model = dbscan_clustering(data_clean, features_scaled)
    # Para DBSCAN, analisar apenas pontos não-noise
    non_noise_mask = dbscan_clusters != -1
    if np.sum(non_noise_mask) > 0:
        analyze_clusters_characteristics(data_clean[non_noise_mask], dbscan_clusters[non_noise_mask], "DBSCAN (sem noise)")
    
    # 10. Comparação
    compare_algorithms(data_clean, features_scaled)
    
    print("\n" + "="*60)
    print("🎉 ANÁLISE COMPLETA!")
    print("="*60)
    print("\n📋 Resumo dos resultados:")
    print("• Dados limpos e preparados")
    print("• PCA: Redução de dimensionalidade realizada")
    print("• K-Means: Clustering particional executado")
    print("• Hierárquico: Clustering hierárquico executado")
    print("• DBSCAN: Clustering baseado em densidade executado")
    print("• Comparação: Todos os algoritmos foram comparados")
    print("• Características: Análise detalhada dos clusters")
    
    if 'species' in data_clean.columns:
        print("\n💡 Dicas para interpretação:")
        print("• ARI próximo de 1: clustering muito similar às classes verdadeiras")
        print("• NMI próximo de 1: alta informação mútua entre clusters e classes")
        print("• Silhouette próximo de 1: clusters bem definidos e separados")
        print("• DBSCAN pode identificar outliers como pontos de noise")

if __name__ == "__main__":
    main()
