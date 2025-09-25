"""
Script Simplificado para Avaliação dos Datasets
Gera resultados em CSV para preenchimento da planilha.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist
import os

def carregar_iris():
    """Carrega o dataset Iris."""
    from sklearn.datasets import load_iris
    iris = load_iris()
    return pd.DataFrame(iris.data, columns=iris.feature_names)

def carregar_penguins():
    """Tenta carregar o dataset Penguins."""
    caminho_penguins = r"penguin-dataset\penguins.csv"
    
    if os.path.exists(caminho_penguins):
        dados = pd.read_csv(caminho_penguins)
        colunas_numericas = dados.select_dtypes(include=[np.number]).columns
        dados_numericos = dados[colunas_numericas].dropna()
        print(f"Dataset Penguins carregado: {dados_numericos.shape}")
        return dados_numericos
    else:
        print("Arquivo penguins.csv não encontrado.")
        return None

def avaliar_dataset_simples(nome_dataset, dados):
    """Avalia um dataset e retorna métricas principais."""
    print(f"\n{'='*50}")
    print(f"AVALIANDO {nome_dataset.upper()}")
    print(f"{'='*50}")
    
    # Normalizar dados
    scaler = StandardScaler()
    dados_norm = scaler.fit_transform(dados)
    
    resultados = {
        'Dataset': nome_dataset,
        'N_Amostras': dados.shape[0],
        'N_Features': dados.shape[1]
    }
    
    # 1. K-Means - encontrar melhor k
    print("Avaliando K-Means...")
    melhor_silhueta = -1
    melhor_k = 2
    
    for k in range(2, 8):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(dados_norm)
        silhueta = silhouette_score(dados_norm, labels)
        davies_bouldin = davies_bouldin_score(dados_norm, labels)
        
        if silhueta > melhor_silhueta:
            melhor_silhueta = silhueta
            melhor_k = k
            resultados['KMeans_Melhor_K'] = k
            resultados['KMeans_Silhueta'] = round(silhueta, 3)
            resultados['KMeans_Davies_Bouldin'] = round(davies_bouldin, 3)
        
        print(f"  K={k}: Silhueta={silhueta:.3f}, Davies-Bouldin={davies_bouldin:.3f}")
    
    # 2. Clustering Hierárquico
    print("Avaliando Clustering Hierárquico...")
    distancias_orig = pdist(dados_norm)
    melhor_cofenetico = -1
    melhor_metodo = ""
    
    for metodo in ['ward', 'complete', 'average']:
        Z = linkage(dados_norm, method=metodo)
        coef_cofenetico, _ = cophenet(Z, distancias_orig)
        
        if coef_cofenetico > melhor_cofenetico:
            melhor_cofenetico = coef_cofenetico
            melhor_metodo = metodo
        
        print(f"  {metodo}: Coef. Cofenético = {coef_cofenetico:.3f}")
    
    resultados['Hierarquico_Melhor_Metodo'] = melhor_metodo
    resultados['Hierarquico_Coef_Cofenetico'] = round(melhor_cofenetico, 3)
    
    # 3. PCA
    print("Avaliando PCA...")
    pca = PCA()
    pca.fit(dados_norm)
    
    variancia_acum = np.cumsum(pca.explained_variance_ratio_)
    comp_95 = np.argmax(variancia_acum >= 0.95) + 1
    comp_90 = np.argmax(variancia_acum >= 0.90) + 1
    comp_80 = np.argmax(variancia_acum >= 0.80) + 1
    
    resultados['PCA_Componentes_80pct'] = comp_80
    resultados['PCA_Componentes_90pct'] = comp_90
    resultados['PCA_Componentes_95pct'] = comp_95
    resultados['PCA_Variancia_PC1'] = round(pca.explained_variance_ratio_[0], 3)
    resultados['PCA_Variancia_PC2'] = round(pca.explained_variance_ratio_[1], 3)
    
    print(f"  Componentes para 80%: {comp_80}")
    print(f"  Componentes para 90%: {comp_90}")
    print(f"  Componentes para 95%: {comp_95}")
    print(f"  Variância PC1: {pca.explained_variance_ratio_[0]:.3f}")
    print(f"  Variância PC2: {pca.explained_variance_ratio_[1]:.3f}")
    
    return resultados

def main():
    """Função principal."""
    print("SISTEMA DE AVALIAÇÃO SIMPLIFICADO")
    print("="*50)
    
    todos_resultados = []
    
    # Avaliar Iris
    print("Carregando dataset Iris...")
    dados_iris = carregar_iris()
    resultado_iris = avaliar_dataset_simples("Iris", dados_iris)
    todos_resultados.append(resultado_iris)
    
    # Tentar avaliar Penguins
    dados_penguins = carregar_penguins()
    if dados_penguins is not None:
        resultado_penguins = avaliar_dataset_simples("Penguins", dados_penguins)
        todos_resultados.append(resultado_penguins)
    
    # Criar DataFrame e salvar
    df_resultados = pd.DataFrame(todos_resultados)
    df_resultados.to_csv('resultados_avaliacao.csv', index=False)
    
    print(f"\n{'='*60}")
    print("RESUMO FINAL PARA PLANILHA")
    print(f"{'='*60}")
    
    for resultado in todos_resultados:
        dataset = resultado['Dataset']
        print(f"\n--- {dataset} ---")
        print(f"K-Means (melhor k={resultado['KMeans_Melhor_K']}): Silhueta = {resultado['KMeans_Silhueta']}")
        print(f"Hierárquico ({resultado['Hierarquico_Melhor_Metodo']}): Cofenético = {resultado['Hierarquico_Coef_Cofenetico']}")
        print(f"PCA: {resultado['PCA_Componentes_95pct']} componentes para 95% variância")
    
    print(f"\nArquivo 'resultados_avaliacao.csv' criado!")
    print("Use estes valores para preencher sua planilha.")
    
    return df_resultados

if __name__ == "__main__":
    df = main()
