"""
Script Especializado: Aprendizado Não Supervisionado em Cybersecurity
Implementa APENAS técnicas não supervisionadas conforme objetivo da pesquisa:
1. Agrupamento Particional (K-Means) 
2. Agrupamento Hierárquico (AGNES)
3. Redução de Dimensionalidade (PCA)
"""

import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
import os
import warnings
warnings.filterwarnings('ignore')

class UnsupervisedCybersecurityAnalyzer:
    """
    Analisador especializado em técnicas não supervisionadas para cybersecurity
    Foco nas três frentes principais da pesquisa
    """
    
    def __init__(self):
        self.dataset_path = None
        self.data = None
        self.processed_data = None
        self.results = {
            'dataset_info': {},
            'kmeans_analysis': {},
            'hierarchical_analysis': {},
            'pca_analysis': {},
            'comparative_results': {}
        }
    
    def download_dataset(self):
        """Baixa o dataset da Kaggle"""
        print("📥 Baixando dataset de cybersecurity da Kaggle...")
        
        try:
            path = kagglehub.dataset_download("ramoliyafenil/text-based-cyber-threat-detection")
            self.dataset_path = path
            print(f"✅ Dataset baixado com sucesso!")
            print(f"📁 Localização: {path}")
            
            files = os.listdir(path)
            print(f"📋 Arquivos encontrados: {files}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao baixar dataset: {str(e)}")
            print("💡 Verifique se você tem as credenciais da Kaggle configuradas")
            return False
    
    def load_and_explore_data(self):
        """Carrega e explora os dados focando em features numéricas"""
        print("\n🔍 Explorando estrutura dos dados...")
        
        if not self.dataset_path:
            print("❌ Dataset não foi baixado. Execute download_dataset() primeiro.")
            return False
        
        try:
            files = os.listdir(self.dataset_path)
            csv_files = [f for f in files if f.endswith('.csv')]
            
            if not csv_files:
                print("❌ Nenhum arquivo CSV encontrado")
                return False
            
            main_file = csv_files[0]
            file_path = os.path.join(self.dataset_path, main_file)
            print(f"📊 Carregando arquivo: {main_file}")
            
            self.data = pd.read_csv(file_path)
            
            print(f"\n📈 Informações do Dataset:")
            print(f"   Linhas: {self.data.shape[0]:,}")
            print(f"   Colunas: {self.data.shape[1]}")
            
            # Focar apenas em features numéricas para análise não supervisionada
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            print(f"   Colunas numéricas: {len(numeric_cols)}")
            
            if len(numeric_cols) == 0:
                print("⚠️ Nenhuma coluna numérica encontrada. Tentando conversão...")
                # Tentar converter colunas categóricas em numéricas
                for col in self.data.columns:
                    if self.data[col].dtype == 'object':
                        try:
                            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                        except:
                            pass
                numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                print("❌ Dados insuficientes para análise não supervisionada (mínimo 2 features numéricas)")
                return False
            
            print(f"\n📋 Features numéricas para análise:")
            for i, col in enumerate(numeric_cols, 1):
                print(f"   {i:2d}. {col}")
            
            # Salvar informações básicas
            self.results['dataset_info'] = {
                'total_samples': self.data.shape[0],
                'total_features': self.data.shape[1],
                'numeric_features': len(numeric_cols),
                'feature_names': list(numeric_cols),
                'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024**2
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {str(e)}")
            return False
    
    def preprocess_data(self):
        """Pré-processamento focado em análise não supervisionada"""
        print("\n🔧 Preparando dados para análise não supervisionada...")
        
        if self.data is None:
            print("❌ Dados não carregados.")
            return False
        
        try:
            # Selecionar apenas colunas numéricas
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            data_numeric = self.data[numeric_cols].copy()
            
            # Remover linhas com valores faltantes
            initial_rows = len(data_numeric)
            data_numeric = data_numeric.dropna()
            final_rows = len(data_numeric)
            
            if final_rows < initial_rows:
                print(f"🧹 Removidas {initial_rows - final_rows} linhas com valores faltantes")
            
            if len(data_numeric) < 50:
                print("❌ Dados insuficientes após limpeza (mínimo 50 amostras)")
                return False
            
            # Normalização dos dados
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_numeric)
            
            print(f"✅ Dados pré-processados:")
            print(f"   Amostras finais: {len(data_scaled):,}")
            print(f"   Features: {data_scaled.shape[1]}")
            print(f"   Dados normalizados: Média ≈ 0, Desvio ≈ 1")
            
            # Salvar dados processados
            self.processed_data = {
                'original': data_numeric,
                'scaled': data_scaled,
                'feature_names': list(numeric_cols),
                'scaler': scaler,
                'n_samples': len(data_scaled),
                'n_features': data_scaled.shape[1]
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Erro no pré-processamento: {str(e)}")
            return False
    
    def kmeans_analysis(self):
        """
        1. AGRUPAMENTO PARTICIONAL (K-MEANS)
        Métricas: coesão intracluster, separação intercluster, coeficiente de silhueta
        """
        print("\n🔵 1. ANÁLISE K-MEANS (Agrupamento Particional)")
        print("=" * 50)
        
        if not self.processed_data:
            print("❌ Dados não processados.")
            return False
        
        try:
            X = self.processed_data['scaled']
            
            # Testar diferentes valores de K
            k_range = range(2, min(11, len(X)//10))
            results = []
            
            print("🔍 Testando diferentes valores de K:")
            
            for k in k_range:
                # Aplicar K-means
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                
                # Calcular métricas
                silhouette_avg = silhouette_score(X, labels)
                
                # Coesão intracluster (WCSS - Within-Cluster Sum of Squares)
                wcss = kmeans.inertia_
                
                # Separação intercluster (distância entre centroides)
                centroids = kmeans.cluster_centers_
                inter_distances = []
                for i in range(len(centroids)):
                    for j in range(i+1, len(centroids)):
                        dist = np.linalg.norm(centroids[i] - centroids[j])
                        inter_distances.append(dist)
                
                avg_inter_distance = np.mean(inter_distances) if inter_distances else 0
                
                results.append({
                    'k': k,
                    'silhouette_score': silhouette_avg,
                    'wcss': wcss,
                    'avg_inter_distance': avg_inter_distance,
                    'labels': labels,
                    'centroids': centroids
                })
                
                print(f"   K={k:2d}: Silhouette={silhouette_avg:.4f}, WCSS={wcss:8.0f}, Sep={avg_inter_distance:.4f}")
            
            # Encontrar melhor K baseado no silhouette score
            best_result = max(results, key=lambda x: x['silhouette_score'])
            best_k = best_result['k']
            
            print(f"\n🏆 Melhor configuração:")
            print(f"   K ótimo: {best_k}")
            print(f"   Silhouette Score: {best_result['silhouette_score']:.4f}")
            print(f"   Coesão (WCSS): {best_result['wcss']:.0f}")
            print(f"   Separação média: {best_result['avg_inter_distance']:.4f}")
            
            # Análise detalhada do melhor modelo
            best_labels = best_result['labels']
            cluster_counts = pd.Series(best_labels).value_counts().sort_index()
            
            print(f"\n📊 Distribuição dos clusters (K={best_k}):")
            for cluster_id, count in cluster_counts.items():
                percentage = (count / len(best_labels)) * 100
                print(f"   Cluster {cluster_id}: {count:4d} pontos ({percentage:5.1f}%)")
            
            # Calcular silhouette por amostra para análise detalhada
            sample_silhouette_values = silhouette_samples(X, best_labels)
            
            # Salvar resultados
            self.results['kmeans_analysis'] = {
                'tested_k_values': [r['k'] for r in results],
                'silhouette_scores': [r['silhouette_score'] for r in results],
                'wcss_values': [r['wcss'] for r in results],
                'separation_values': [r['avg_inter_distance'] for r in results],
                'best_k': best_k,
                'best_silhouette': best_result['silhouette_score'],
                'best_wcss': best_result['wcss'],
                'best_separation': best_result['avg_inter_distance'],
                'cluster_distribution': cluster_counts.to_dict(),
                'silhouette_per_sample': sample_silhouette_values.tolist()
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Erro na análise K-means: {str(e)}")
            return False
    
    def hierarchical_analysis(self):
        """
        2. AGRUPAMENTO HIERÁRQUICO (AGNES)
        Métrica: coeficiente de correlação cofenética (rc)
        """
        print("\n🔴 2. ANÁLISE HIERÁRQUICA (AGNES)")
        print("=" * 50)
        
        if not self.processed_data:
            print("❌ Dados não processados.")
            return False
        
        try:
            X = self.processed_data['scaled']
            
            # Para datasets grandes, usar amostra para análise hierárquica
            max_samples = 1000  # Limite para performance
            if len(X) > max_samples:
                print(f"📉 Dataset grande ({len(X)} amostras). Usando amostra de {max_samples} para análise hierárquica.")
                sample_idx = np.random.choice(len(X), max_samples, replace=False)
                X_sample = X[sample_idx]
            else:
                X_sample = X
            
            # Testar diferentes métodos de linkage
            linkage_methods = ['ward', 'complete', 'average', 'single']
            results = {}
            
            print("🔍 Testando métodos de linkage:")
            
            for method in linkage_methods:
                try:
                    # Calcular matriz de linkage
                    if method == 'ward':
                        linkage_matrix = linkage(X_sample, method=method)
                    else:
                        # Para outros métodos, usar distância euclidiana
                        distances = pdist(X_sample, metric='euclidean')
                        linkage_matrix = linkage(distances, method=method)
                    
                    # Calcular coeficiente de correlação cofenética
                    original_distances = pdist(X_sample, metric='euclidean')
                    cophenetic_corr, _ = cophenet(linkage_matrix, original_distances)
                    
                    # Teste com diferentes números de clusters
                    silhouette_scores = []
                    n_clusters_range = range(2, min(11, len(X_sample)//10))
                    
                    for n_clusters in n_clusters_range:
                        # Aplicar clustering hierárquico
                        hierarchical = AgglomerativeClustering(
                            n_clusters=n_clusters, 
                            linkage=method
                        )
                        cluster_labels = hierarchical.fit_predict(X_sample)
                        
                        # Calcular silhouette
                        sil_score = silhouette_score(X_sample, cluster_labels)
                        silhouette_scores.append(sil_score)
                    
                    # Melhor configuração para este método
                    best_n_clusters = n_clusters_range[np.argmax(silhouette_scores)]
                    best_silhouette = max(silhouette_scores)
                    
                    results[method] = {
                        'cophenetic_correlation': cophenetic_corr,
                        'linkage_matrix': linkage_matrix,
                        'best_n_clusters': best_n_clusters,
                        'best_silhouette': best_silhouette,
                        'silhouette_scores': silhouette_scores,
                        'n_clusters_range': list(n_clusters_range)
                    }
                    
                    print(f"   {method:8s}: rc={cophenetic_corr:.4f}, Melhor K={best_n_clusters} (Sil={best_silhouette:.4f})")
                    
                except Exception as method_error:
                    print(f"   {method:8s}: ❌ Erro - {str(method_error)}")
                    continue
            
            if not results:
                print("❌ Nenhum método de linkage funcionou")
                return False
            
            # Encontrar melhor método baseado no coeficiente cofenético
            best_method = max(results.keys(), key=lambda m: results[m]['cophenetic_correlation'])
            best_cophenetic = results[best_method]['cophenetic_correlation']
            
            print(f"\n🏆 Melhor método de linkage:")
            print(f"   Método: {best_method}")
            print(f"   Coeficiente Cofenético (rc): {best_cophenetic:.4f}")
            print(f"   Melhor K: {results[best_method]['best_n_clusters']}")
            print(f"   Silhouette Score: {results[best_method]['best_silhouette']:.4f}")
            
            # Interpretação do coeficiente cofenético
            if best_cophenetic > 0.8:
                interpretation = "Excelente preservação das distâncias"
            elif best_cophenetic > 0.7:
                interpretation = "Boa preservação das distâncias"
            elif best_cophenetic > 0.6:
                interpretation = "Preservação moderada das distâncias"
            else:
                interpretation = "Baixa preservação das distâncias"
            
            print(f"   Interpretação: {interpretation}")
            
            # Salvar resultados
            self.results['hierarchical_analysis'] = {
                'tested_methods': list(results.keys()),
                'cophenetic_correlations': {method: results[method]['cophenetic_correlation'] 
                                          for method in results.keys()},
                'best_method': best_method,
                'best_cophenetic_correlation': best_cophenetic,
                'best_n_clusters': results[best_method]['best_n_clusters'],
                'best_silhouette': results[best_method]['best_silhouette'],
                'interpretation': interpretation,
                'sample_size_used': len(X_sample),
                'total_samples': len(X)
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Erro na análise hierárquica: {str(e)}")
            return False
    
    def pca_analysis(self):
        """
        3. REDUÇÃO DE DIMENSIONALIDADE (PCA)
        Métrica: variância explicada
        """
        print("\n🟡 3. ANÁLISE PCA (Redução de Dimensionalidade)")
        print("=" * 50)
        
        if not self.processed_data:
            print("❌ Dados não processados.")
            return False
        
        try:
            X = self.processed_data['scaled']
            feature_names = self.processed_data['feature_names']
            n_features = X.shape[1]
            
            print(f"📊 Analisando {n_features} features originais...")
            
            # Aplicar PCA completo para análise de variância
            pca_full = PCA()
            pca_full.fit(X)
            
            # Variância explicada por cada componente
            explained_variance_ratio = pca_full.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            
            print(f"\n📈 Variância explicada por componente:")
            for i in range(min(10, len(explained_variance_ratio))):  # Mostrar até 10 primeiros
                print(f"   PC{i+1:2d}: {explained_variance_ratio[i]:.4f} ({explained_variance_ratio[i]*100:5.1f}%)")
            
            # Encontrar número de componentes para diferentes níveis de variância
            variance_thresholds = [0.80, 0.90, 0.95, 0.99]
            components_needed = {}
            
            print(f"\n🎯 Componentes necessários para preservar variância:")
            for threshold in variance_thresholds:
                n_components = np.argmax(cumulative_variance >= threshold) + 1
                components_needed[threshold] = n_components
                print(f"   {threshold*100:4.0f}%: {n_components:2d} componentes ({n_components/n_features*100:5.1f}% das features)")
            
            # Análise detalhada para 95% da variância (padrão comum)
            n_components_95 = components_needed[0.95]
            pca_95 = PCA(n_components=n_components_95)
            X_pca_95 = pca_95.fit_transform(X)
            
            print(f"\n🔍 Análise detalhada (95% variância - {n_components_95} componentes):")
            print(f"   Redução de dimensionalidade: {n_features} → {n_components_95}")
            print(f"   Economia de features: {n_features - n_components_95} ({(1-n_components_95/n_features)*100:.1f}%)")
            print(f"   Variância preservada: {pca_95.explained_variance_ratio_.sum():.4f} ({pca_95.explained_variance_ratio_.sum()*100:.1f}%)")
            
            # Análise dos componentes principais
            components = pca_95.components_
            print(f"\n📋 Contribuição das features nos primeiros componentes:")
            
            for i in range(min(3, n_components_95)):  # Mostrar até 3 primeiros componentes
                print(f"\n   PC{i+1} (variância: {pca_95.explained_variance_ratio_[i]:.3f}):")
                
                # Features com maior contribuição absoluta
                feature_contributions = [(abs(components[i][j]), feature_names[j], components[i][j]) 
                                       for j in range(len(feature_names))]
                feature_contributions.sort(reverse=True)
                
                for contrib_abs, feature_name, contrib_real in feature_contributions[:5]:
                    signal = "+" if contrib_real > 0 else "-"
                    print(f"      {signal} {feature_name}: {contrib_abs:.3f}")
            
            # Testar clustering no espaço reduzido
            if n_components_95 >= 2:
                print(f"\n🔄 Teste de clustering no espaço PCA...")
                
                # K-means no espaço reduzido
                k_range = range(2, min(8, len(X_pca_95)//20))
                pca_silhouette_scores = []
                
                for k in k_range:
                    kmeans_pca = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels_pca = kmeans_pca.fit_predict(X_pca_95)
                    sil_score = silhouette_score(X_pca_95, labels_pca)
                    pca_silhouette_scores.append(sil_score)
                
                best_k_pca = k_range[np.argmax(pca_silhouette_scores)]
                best_sil_pca = max(pca_silhouette_scores)
                
                print(f"   Melhor K no espaço PCA: {best_k_pca}")
                print(f"   Silhouette Score PCA: {best_sil_pca:.4f}")
            
            # Salvar resultados
            self.results['pca_analysis'] = {
                'original_features': n_features,
                'explained_variance_ratio': explained_variance_ratio.tolist(),
                'cumulative_variance': cumulative_variance.tolist(),
                'components_for_variance': components_needed,
                'recommended_components': n_components_95,
                'variance_preserved_95': float(pca_95.explained_variance_ratio_.sum()),
                'dimensionality_reduction_percent': float((1-n_components_95/n_features)*100),
                'feature_names': feature_names,
                'principal_components': components.tolist() if n_components_95 <= 10 else components[:10].tolist(),
                'pca_clustering_performance': {
                    'best_k': best_k_pca if n_components_95 >= 2 else None,
                    'best_silhouette': best_sil_pca if n_components_95 >= 2 else None
                } if n_components_95 >= 2 else None
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Erro na análise PCA: {str(e)}")
            return False
    
    def comparative_analysis(self):
        """Análise comparativa entre as três técnicas"""
        print("\n📊 4. ANÁLISE COMPARATIVA")
        print("=" * 50)
        
        # Verificar se todas as análises foram executadas
        required_analyses = ['kmeans_analysis', 'hierarchical_analysis', 'pca_analysis']
        missing_analyses = [analysis for analysis in required_analyses 
                          if analysis not in self.results or not self.results[analysis]]
        
        if missing_analyses:
            print(f"⚠️ Análises pendentes: {', '.join(missing_analyses)}")
            return False
        
        # Extrair métricas principais
        kmeans = self.results['kmeans_analysis']
        hierarchical = self.results['hierarchical_analysis']
        pca = self.results['pca_analysis']
        
        print("🏆 RESUMO COMPARATIVO:")
        print(f"   K-Means (Melhor K={kmeans['best_k']}):")
        print(f"      • Silhouette Score: {kmeans['best_silhouette']:.4f}")
        print(f"      • Coesão (WCSS): {kmeans['best_wcss']:.0f}")
        print(f"      • Separação Intercluster: {kmeans['best_separation']:.4f}")
        
        print(f"   Hierárquico ({hierarchical['best_method']}):")
        print(f"      • Coef. Cofenético (rc): {hierarchical['best_cophenetic_correlation']:.4f}")
        print(f"      • Silhouette Score: {hierarchical['best_silhouette']:.4f}")
        print(f"      • Melhor K: {hierarchical['best_n_clusters']}")
        
        print(f"   PCA ({pca['recommended_components']} componentes):")
        print(f"      • Variância Preservada: {pca['variance_preserved_95']:.4f} ({pca['variance_preserved_95']*100:.1f}%)")
        print(f"      • Redução Dimensional: {pca['dimensionality_reduction_percent']:.1f}%")
        if pca['pca_clustering_performance']:
            print(f"      • Clustering no PCA: K={pca['pca_clustering_performance']['best_k']}, Sil={pca['pca_clustering_performance']['best_silhouette']:.4f}")
        
        # Avaliação qualitativa
        print(f"\n📝 AVALIAÇÃO QUALITATIVA:")
        
        # K-means
        if kmeans['best_silhouette'] > 0.5:
            kmeans_quality = "Excelente"
        elif kmeans['best_silhouette'] > 0.3:
            kmeans_quality = "Boa"
        elif kmeans['best_silhouette'] > 0.1:
            kmeans_quality = "Moderada"
        else:
            kmeans_quality = "Baixa"
        
        # Hierárquico
        if hierarchical['best_cophenetic_correlation'] > 0.8:
            hier_quality = "Excelente"
        elif hierarchical['best_cophenetic_correlation'] > 0.7:
            hier_quality = "Boa"
        elif hierarchical['best_cophenetic_correlation'] > 0.6:
            hier_quality = "Moderada"
        else:
            hier_quality = "Baixa"
        
        # PCA
        if pca['variance_preserved_95'] > 0.95:
            pca_quality = "Excelente"
        elif pca['variance_preserved_95'] > 0.90:
            pca_quality = "Boa"
        elif pca['variance_preserved_95'] > 0.80:
            pca_quality = "Moderada"
        else:
            pca_quality = "Baixa"
        
        print(f"   K-Means: {kmeans_quality} qualidade de clustering")
        print(f"   Hierárquico: {hier_quality} preservação de distâncias")
        print(f"   PCA: {pca_quality} preservação de variância")
        
        # Recomendações
        print(f"\n💡 RECOMENDAÇÕES:")
        
        best_silhouette_technique = "K-Means" if kmeans['best_silhouette'] >= hierarchical['best_silhouette'] else "Hierárquico"
        print(f"   • Melhor clustering: {best_silhouette_technique}")
        
        if pca['dimensionality_reduction_percent'] > 50:
            print(f"   • PCA recomendado: Reduz {pca['dimensionality_reduction_percent']:.0f}% das features mantendo {pca['variance_preserved_95']*100:.0f}% da informação")
        else:
            print(f"   • PCA opcional: Redução modest de {pca['dimensionality_reduction_percent']:.0f}%")
        
        # Salvar resultados comparativos
        self.results['comparative_results'] = {
            'kmeans_quality': kmeans_quality,
            'hierarchical_quality': hier_quality,
            'pca_quality': pca_quality,
            'best_clustering_technique': best_silhouette_technique,
            'best_silhouette_score': max(kmeans['best_silhouette'], hierarchical['best_silhouette']),
            'pca_recommendation': "Recomendado" if pca['dimensionality_reduction_percent'] > 50 else "Opcional",
            'summary': {
                'kmeans': {
                    'technique': 'Agrupamento Particional (K-Means)',
                    'best_k': kmeans['best_k'],
                    'silhouette_score': kmeans['best_silhouette'],
                    'wcss': kmeans['best_wcss'],
                    'separation': kmeans['best_separation'],
                    'quality': kmeans_quality
                },
                'hierarchical': {
                    'technique': 'Agrupamento Hierárquico (AGNES)',
                    'best_method': hierarchical['best_method'],
                    'cophenetic_correlation': hierarchical['best_cophenetic_correlation'],
                    'best_k': hierarchical['best_n_clusters'],
                    'silhouette_score': hierarchical['best_silhouette'],
                    'quality': hier_quality
                },
                'pca': {
                    'technique': 'Redução de Dimensionalidade (PCA)',
                    'components_used': pca['recommended_components'],
                    'variance_explained': pca['variance_preserved_95'],
                    'reduction_percent': pca['dimensionality_reduction_percent'],
                    'quality': pca_quality
                }
            }
        }
        
        return True
    
    def generate_results_table(self):
        """Gera tabela com resultados para inclusão na pesquisa"""
        print("\n📋 5. GERANDO TABELA DE RESULTADOS")
        print("=" * 50)
        
        if 'comparative_results' not in self.results:
            print("❌ Análise comparativa não foi executada")
            return False
        
        # Preparar dados para tabela
        summary = self.results['comparative_results']['summary']
        
        # Criar tabela formatada
        table_data = []
        
        # K-means
        table_data.append({
            'Técnica': 'K-means',
            'Métrica Principal': 'Silhouette Score',
            'Valor': f"{summary['kmeans']['silhouette_score']:.4f}",
            'Parâmetros': f"K = {summary['kmeans']['best_k']}",
            'Métricas Adicionais': f"WCSS: {summary['kmeans']['wcss']:.0f}, Sep: {summary['kmeans']['separation']:.4f}",
            'Qualidade': summary['kmeans']['quality']
        })
        
        # Hierárquico
        table_data.append({
            'Técnica': 'AGNES (Hierárquico)',
            'Métrica Principal': 'Coef. Cofenético (rc)',
            'Valor': f"{summary['hierarchical']['cophenetic_correlation']:.4f}",
            'Parâmetros': f"Linkage: {summary['hierarchical']['best_method']}, K = {summary['hierarchical']['best_k']}",
            'Métricas Adicionais': f"Silhouette: {summary['hierarchical']['silhouette_score']:.4f}",
            'Qualidade': summary['hierarchical']['quality']
        })
        
        # PCA
        table_data.append({
            'Técnica': 'PCA',
            'Métrica Principal': 'Variância Explicada',
            'Valor': f"{summary['pca']['variance_explained']:.4f}",
            'Parâmetros': f"Componentes: {summary['pca']['components_used']}",
            'Métricas Adicionais': f"Redução: {summary['pca']['reduction_percent']:.1f}%",
            'Qualidade': summary['pca']['quality']
        })
        
        # Criar DataFrame para melhor visualização
        df_results = pd.DataFrame(table_data)
        
        print("📊 TABELA DE RESULTADOS - APRENDIZADO NÃO SUPERVISIONADO:")
        print("=" * 100)
        print(df_results.to_string(index=False, max_colwidth=30))
        
        # Salvar em diferentes formatos
        output_dir = r"c:\Users\Administrador\Faculdade-Impacta\Iniciação-cientifica\doc\code\cybersecurity-datasets"
        os.makedirs(output_dir, exist_ok=True)
        
        # CSV para uso em planilhas
        csv_file = os.path.join(output_dir, "resultados_aprendizado_nao_supervisionado.csv")
        df_results.to_csv(csv_file, index=False, encoding='utf-8')
        
        # Markdown para documentação
        md_file = os.path.join(output_dir, "tabela_resultados_nao_supervisionado.md")
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write("# Resultados - Aprendizado Não Supervisionado em Cybersecurity\n\n")
            f.write(f"**Dataset:** {self.results['dataset_info']['total_samples']:,} amostras, ")
            f.write(f"{self.results['dataset_info']['numeric_features']} features numéricas\n\n")
            f.write("## Tabela de Resultados\n\n")
            f.write(df_results.to_markdown(index=False))
            f.write("\n\n## Interpretação\n\n")
            f.write("### Agrupamento Particional (K-Means)\n")
            f.write(f"- **Silhouette Score**: {summary['kmeans']['silhouette_score']:.4f}\n")
            f.write(f"- **Coesão (WCSS)**: {summary['kmeans']['wcss']:.0f}\n") 
            f.write(f"- **Separação Intercluster**: {summary['kmeans']['separation']:.4f}\n")
            f.write(f"- **Avaliação**: {summary['kmeans']['quality']}\n\n")
            f.write("### Agrupamento Hierárquico (AGNES)\n")
            f.write(f"- **Coeficiente Cofenético (rc)**: {summary['hierarchical']['cophenetic_correlation']:.4f}\n")
            f.write(f"- **Método de Linkage**: {summary['hierarchical']['best_method']}\n")
            f.write(f"- **Avaliação**: {summary['hierarchical']['quality']}\n\n")
            f.write("### Redução de Dimensionalidade (PCA)\n")
            f.write(f"- **Variância Explicada**: {summary['pca']['variance_explained']:.4f} ({summary['pca']['variance_explained']*100:.1f}%)\n")
            f.write(f"- **Componentes Utilizados**: {summary['pca']['components_used']}\n")
            f.write(f"- **Redução Dimensional**: {summary['pca']['reduction_percent']:.1f}%\n")
            f.write(f"- **Avaliação**: {summary['pca']['quality']}\n")
        
        print(f"\n✅ Resultados salvos em:")
        print(f"   📊 CSV: {csv_file}")
        print(f"   📋 Markdown: {md_file}")
        
        return df_results
    
    def run_complete_analysis(self):
        """Executa análise completa focada em aprendizado não supervisionado"""
        print("🚀 ANÁLISE COMPLETA - APRENDIZADO NÃO SUPERVISIONADO")
        print("=" * 60)
        print("Implementando as três frentes principais da pesquisa:")
        print("1. Agrupamento Particional (K-Means)")
        print("2. Agrupamento Hierárquico (AGNES)")  
        print("3. Redução de Dimensionalidade (PCA)")
        print("=" * 60)
        
        # 1. Download
        if not self.download_dataset():
            return False
        
        # 2. Carregamento
        if not self.load_and_explore_data():
            return False
        
        # 3. Pré-processamento
        if not self.preprocess_data():
            return False
        
        # 4. Análises específicas
        self.kmeans_analysis()
        self.hierarchical_analysis()
        self.pca_analysis()
        
        # 5. Análise comparativa
        self.comparative_analysis()
        
        # 6. Gerar tabela de resultados
        results_table = self.generate_results_table()
        
        print("\n" + "=" * 60)
        print("🎉 ANÁLISE NÃO SUPERVISIONADA FINALIZADA!")
        print("📊 Confira os resultados na pasta cybersecurity-datasets/")
        print("📋 Tabela pronta para inclusão na pesquisa acadêmica")
        
        return results_table


def main():
    """Função principal"""
    try:
        plt.ioff()  # Desabilitar plots interativos
        
        analyzer = UnsupervisedCybersecurityAnalyzer()
        results = analyzer.run_complete_analysis()
        
        if results is not None:
            print(f"\n🎯 PRÓXIMOS PASSOS:")
            print("1. Confira os arquivos gerados na pasta cybersecurity-datasets/")
            print("2. Use a tabela CSV/Markdown na sua pesquisa")
            print("3. Analise os resultados para discussão acadêmica")
        
    except Exception as e:
        print(f"❌ Erro durante a execução: {str(e)}")
        print("💡 Verifique se:")
        print("   1. As dependências estão instaladas")
        print("   2. As credenciais da Kaggle estão configuradas")
        print("   3. Há conexão com a internet")


if __name__ == "__main__":
    main()
