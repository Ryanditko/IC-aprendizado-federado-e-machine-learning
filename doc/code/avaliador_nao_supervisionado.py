"""
Implementação Prática: Métricas de Avaliação para Aprendizado Não Supervisionado

Este módulo implementa todas as principais métricas de avaliação para técnicas
de aprendizado não supervisionado, permitindo validação automática e geração
de relatórios para preenchimento de planilhas.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

class AvaliadorNaoSupervisionado:
    """
    Classe principal para avaliação de técnicas de aprendizado não supervisionado.
    """
    
    def __init__(self, dados):
        """
        Inicializa o avaliador com os dados.
        
        Args:
            dados: DataFrame pandas ou array numpy com os dados
        """
        if isinstance(dados, pd.DataFrame):
            self.dados = dados.values
            self.nomes_colunas = dados.columns.tolist()
        else:
            self.dados = dados
            self.nomes_colunas = [f'Feature_{i}' for i in range(dados.shape[1])]
        
        # Normalizar os dados
        self.scaler = StandardScaler()
        self.dados_normalizados = self.scaler.fit_transform(self.dados)
        
        self.relatorio = {}
    
    def avaliar_agrupamento_particional(self, k_min=2, k_max=10, random_state=42):
        """
        Avalia diferentes configurações de K-Means.
        
        Returns:
            dict: Relatório com métricas para cada valor de k
        """
        print("=== AVALIAÇÃO DE AGRUPAMENTO PARTICIONAL (K-MEANS) ===")
        
        resultados = {
            'k': [],
            'inercia': [],
            'silhueta': [],
            'davies_bouldin': [],
            'calinski_harabasz': []
        }
        
        for k in range(k_min, k_max + 1):
            # Treinar K-Means
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(self.dados_normalizados)
            
            # Calcular métricas
            inercia = kmeans.inertia_
            silhueta = silhouette_score(self.dados_normalizados, labels)
            davies_bouldin = davies_bouldin_score(self.dados_normalizados, labels)
            calinski_harabasz = calinski_harabasz_score(self.dados_normalizados, labels)
            
            resultados['k'].append(k)
            resultados['inercia'].append(inercia)
            resultados['silhueta'].append(silhueta)
            resultados['davies_bouldin'].append(davies_bouldin)
            resultados['calinski_harabasz'].append(calinski_harabasz)
            
            print(f"K={k}: Silhueta={silhueta:.3f}, Davies-Bouldin={davies_bouldin:.3f}")
        
        self.relatorio['agrupamento_particional'] = resultados
        return resultados
    
    def encontrar_k_otimo(self, metrica='silhueta'):
        """
        Encontra o valor ótimo de k baseado na métrica escolhida.
        
        Args:
            metrica: 'silhueta', 'davies_bouldin' ou 'cotovelo'
        
        Returns:
            int: Valor ótimo de k
        """
        if 'agrupamento_particional' not in self.relatorio:
            self.avaliar_agrupamento_particional()
        
        resultados = self.relatorio['agrupamento_particional']
        
        if metrica == 'silhueta':
            idx_otimo = np.argmax(resultados['silhueta'])
        elif metrica == 'davies_bouldin':
            idx_otimo = np.argmin(resultados['davies_bouldin'])
        elif metrica == 'cotovelo':
            # Método do cotovelo - encontra o ponto de maior curvatura
            inercias = np.array(resultados['inercia'])
            # Calcular segunda derivada
            diffs = np.diff(inercias, 2)
            idx_otimo = np.argmax(diffs) + 1  # +1 porque diff remove um elemento
        
        k_otimo = resultados['k'][idx_otimo]
        print(f"K ótimo baseado em {metrica}: {k_otimo}")
        return k_otimo
    
    def avaliar_agrupamento_hierarquico(self, metodos=['ward', 'complete', 'average']):
        """
        Avalia clustering hierárquico com diferentes métodos de linkage.
        
        Returns:
            dict: Relatório com coeficiente de correlação cofenética para cada método
        """
        print("\n=== AVALIAÇÃO DE AGRUPAMENTO HIERÁRQUICO ===")
        
        resultados = {}
        distancias_originais = pdist(self.dados_normalizados)
        
        for metodo in metodos:
            # Criar linkage matrix
            Z = linkage(self.dados_normalizados, method=metodo)
            
            # Calcular coeficiente de correlação cofenética
            coef_cofenetico, _ = cophenet(Z, distancias_originais)
            
            resultados[metodo] = {
                'coeficiente_cofenetico': coef_cofenetico,
                'linkage_matrix': Z
            }
            
            print(f"Método {metodo}: Coeficiente Cofenético = {coef_cofenetico:.3f}")
        
        self.relatorio['agrupamento_hierarquico'] = resultados
        return resultados
    
    def avaliar_reducao_dimensionalidade(self, n_componentes_max=None):
        """
        Avalia PCA para redução de dimensionalidade.
        
        Returns:
            dict: Relatório com variância explicada
        """
        print("\n=== AVALIAÇÃO DE REDUÇÃO DE DIMENSIONALIDADE (PCA) ===")
        
        if n_componentes_max is None:
            n_componentes_max = min(self.dados.shape) - 1
        
        # Aplicar PCA
        pca = PCA(n_components=n_componentes_max)
        dados_pca = pca.fit_transform(self.dados_normalizados)
        
        # Calcular métricas
        variancia_explicada = pca.explained_variance_ratio_
        variancia_acumulada = np.cumsum(variancia_explicada)
        
        # Encontrar número de componentes para 80%, 90%, 95% da variância
        componentes_80 = np.argmax(variancia_acumulada >= 0.8) + 1
        componentes_90 = np.argmax(variancia_acumulada >= 0.9) + 1
        componentes_95 = np.argmax(variancia_acumulada >= 0.95) + 1
        
        resultados = {
            'variancia_explicada': variancia_explicada,
            'variancia_acumulada': variancia_acumulada,
            'componentes_80_pct': componentes_80,
            'componentes_90_pct': componentes_90,
            'componentes_95_pct': componentes_95,
            'dados_transformados': dados_pca
        }
        
        print(f"Componentes para 80% da variância: {componentes_80}")
        print(f"Componentes para 90% da variância: {componentes_90}")
        print(f"Componentes para 95% da variância: {componentes_95}")
        
        self.relatorio['reducao_dimensionalidade'] = resultados
        return resultados
    
    def avaliar_deteccao_anomalias(self, metodos=['isolation_forest', 'one_class_svm', 'lof']):
        """
        Avalia diferentes métodos de detecção de anomalias.
        
        Returns:
            dict: Relatório com scores de anomalia para cada método
        """
        print("\n=== AVALIAÇÃO DE DETECÇÃO DE ANOMALIAS ===")
        
        resultados = {}
        
        for metodo in metodos:
            if metodo == 'isolation_forest':
                detector = IsolationForest(contamination=0.1, random_state=42)
                anomalia_scores = detector.fit_predict(self.dados_normalizados)
                scores = detector.decision_function(self.dados_normalizados)
                
            elif metodo == 'one_class_svm':
                detector = OneClassSVM(gamma='auto')
                anomalia_scores = detector.fit_predict(self.dados_normalizados)
                scores = detector.decision_function(self.dados_normalizados)
                
            elif metodo == 'lof':
                detector = LocalOutlierFactor(contamination=0.1)
                anomalia_scores = detector.fit_predict(self.dados_normalizados)
                scores = detector.negative_outlier_factor_
            
            # Contar anomalias detectadas
            n_anomalias = np.sum(anomalia_scores == -1)
            taxa_contaminacao = n_anomalias / len(anomalia_scores)
            
            resultados[metodo] = {
                'predicoes': anomalia_scores,
                'scores': scores,
                'n_anomalias': n_anomalias,
                'taxa_contaminacao': taxa_contaminacao
            }
            
            print(f"{metodo}: {n_anomalias} anomalias ({taxa_contaminacao:.1%})")
        
        self.relatorio['deteccao_anomalias'] = resultados
        return resultados
    
    def gerar_relatorio_completo(self):
        """
        Gera um relatório completo de todas as avaliações.
        
        Returns:
            dict: Relatório completo formatado para análise
        """
        print("\n" + "="*60)
        print("RELATÓRIO COMPLETO DE AVALIAÇÃO")
        print("="*60)
        
        relatorio_final = {
            'resumo_dados': {
                'n_amostras': self.dados.shape[0],
                'n_features': self.dados.shape[1],
                'nomes_features': self.nomes_colunas
            }
        }
        
        # Executar todas as avaliações se ainda não foram feitas
        if 'agrupamento_particional' not in self.relatorio:
            self.avaliar_agrupamento_particional()
        
        if 'agrupamento_hierarquico' not in self.relatorio:
            self.avaliar_agrupamento_hierarquico()
        
        if 'reducao_dimensionalidade' not in self.relatorio:
            self.avaliar_reducao_dimensionalidade()
        
        if 'deteccao_anomalias' not in self.relatorio:
            self.avaliar_deteccao_anomalias()
        
        # Compilar melhores resultados
        part_results = self.relatorio['agrupamento_particional']
        melhor_k_silhueta = part_results['k'][np.argmax(part_results['silhueta'])]
        melhor_silhueta = max(part_results['silhueta'])
        
        hier_results = self.relatorio['agrupamento_hierarquico']
        melhor_metodo_hier = max(hier_results.keys(), 
                                key=lambda x: hier_results[x]['coeficiente_cofenetico'])
        melhor_cofenetico = hier_results[melhor_metodo_hier]['coeficiente_cofenetico']
        
        pca_results = self.relatorio['reducao_dimensionalidade']
        
        relatorio_final.update({
            'melhores_resultados': {
                'agrupamento_particional': {
                    'melhor_k': melhor_k_silhueta,
                    'silhueta_score': melhor_silhueta,
                    'metrica_usada': 'silhueta'
                },
                'agrupamento_hierarquico': {
                    'melhor_metodo': melhor_metodo_hier,
                    'coeficiente_cofenetico': melhor_cofenetico
                },
                'reducao_dimensionalidade': {
                    'componentes_95_pct': pca_results['componentes_95_pct'],
                    'variancia_total_95': pca_results['variancia_acumulada'][pca_results['componentes_95_pct']-1]
                }
            },
            'detalhes_completos': self.relatorio
        })
        
        return relatorio_final
    
    def exportar_para_planilha(self, nome_arquivo='relatorio_avaliacao.xlsx'):
        """
        Exporta os resultados para uma planilha Excel.
        """
        with pd.ExcelWriter(nome_arquivo, engine='openpyxl') as writer:
            # Aba 1: Resumo
            resumo_data = []
            if 'agrupamento_particional' in self.relatorio:
                part_results = self.relatorio['agrupamento_particional']
                melhor_idx = np.argmax(part_results['silhueta'])
                resumo_data.append({
                    'Técnica': 'K-means',
                    'Métrica': 'Coeficiente de Silhueta',
                    'Melhor Valor': part_results['silhueta'][melhor_idx],
                    'Parâmetro': f"k={part_results['k'][melhor_idx]}"
                })
                resumo_data.append({
                    'Técnica': 'K-means',
                    'Métrica': 'Davies-Bouldin',
                    'Melhor Valor': min(part_results['davies_bouldin']),
                    'Parâmetro': f"k={part_results['k'][np.argmin(part_results['davies_bouldin'])]}"
                })
            
            if 'agrupamento_hierarquico' in self.relatorio:
                for metodo, resultados in self.relatorio['agrupamento_hierarquico'].items():
                    resumo_data.append({
                        'Técnica': f'Hierárquico ({metodo})',
                        'Métrica': 'Coeficiente Cofenético',
                        'Melhor Valor': resultados['coeficiente_cofenetico'],
                        'Parâmetro': metodo
                    })
            
            if 'reducao_dimensionalidade' in self.relatorio:
                pca_results = self.relatorio['reducao_dimensionalidade']
                resumo_data.append({
                    'Técnica': 'PCA',
                    'Métrica': 'Componentes para 95% variância',
                    'Melhor Valor': pca_results['componentes_95_pct'],
                    'Parâmetro': f"{pca_results['variancia_acumulada'][pca_results['componentes_95_pct']-1]:.3f} var. acum."
                })
            
            df_resumo = pd.DataFrame(resumo_data)
            df_resumo.to_excel(writer, sheet_name='Resumo', index=False)
            
            # Aba 2: Detalhes K-means
            if 'agrupamento_particional' in self.relatorio:
                df_kmeans = pd.DataFrame(self.relatorio['agrupamento_particional'])
                df_kmeans.to_excel(writer, sheet_name='K-means Detalhado', index=False)
            
            # Aba 3: Variância PCA
            if 'reducao_dimensionalidade' in self.relatorio:
                pca_data = self.relatorio['reducao_dimensionalidade']
                df_pca = pd.DataFrame({
                    'Componente': range(1, len(pca_data['variancia_explicada']) + 1),
                    'Variância Explicada': pca_data['variancia_explicada'],
                    'Variância Acumulada': pca_data['variancia_acumulada']
                })
                df_pca.to_excel(writer, sheet_name='PCA Variância', index=False)
        
        print(f"\nRelatório exportado para: {nome_arquivo}")
    
    def plotar_resultados(self):
        """
        Gera visualizações dos resultados.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Método do cotovelo
        if 'agrupamento_particional' in self.relatorio:
            part_results = self.relatorio['agrupamento_particional']
            axes[0, 0].plot(part_results['k'], part_results['inercia'], 'bo-')
            axes[0, 0].set_xlabel('Número de Clusters (k)')
            axes[0, 0].set_ylabel('Inércia (WCSS)')
            axes[0, 0].set_title('Método do Cotovelo')
            axes[0, 0].grid(True)
        
        # Plot 2: Coeficiente de Silhueta
        if 'agrupamento_particional' in self.relatorio:
            axes[0, 1].plot(part_results['k'], part_results['silhueta'], 'ro-')
            axes[0, 1].set_xlabel('Número de Clusters (k)')
            axes[0, 1].set_ylabel('Coeficiente de Silhueta')
            axes[0, 1].set_title('Coeficiente de Silhueta vs K')
            axes[0, 1].grid(True)
        
        # Plot 3: Variância Explicada PCA
        if 'reducao_dimensionalidade' in self.relatorio:
            pca_results = self.relatorio['reducao_dimensionalidade']
            axes[1, 0].plot(range(1, len(pca_results['variancia_acumulada']) + 1), 
                           pca_results['variancia_acumulada'], 'go-')
            axes[1, 0].axhline(y=0.95, color='r', linestyle='--', label='95% variância')
            axes[1, 0].set_xlabel('Número de Componentes')
            axes[1, 0].set_ylabel('Variância Acumulada')
            axes[1, 0].set_title('Variância Explicada - PCA')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Plot 4: Comparação de métodos hierárquicos
        if 'agrupamento_hierarquico' in self.relatorio:
            hier_results = self.relatorio['agrupamento_hierarquico']
            metodos = list(hier_results.keys())
            coeficientes = [hier_results[m]['coeficiente_cofenetico'] for m in metodos]
            axes[1, 1].bar(metodos, coeficientes, color=['blue', 'orange', 'green'])
            axes[1, 1].set_ylabel('Coeficiente Cofenético')
            axes[1, 1].set_title('Comparação Métodos Hierárquicos')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()


def exemplo_uso_iris():
    """
    Exemplo de uso com o dataset Iris.
    """
    print("EXEMPLO DE USO - DATASET IRIS")
    print("="*50)
    
    # Carregar dados
    from sklearn.datasets import load_iris
    iris = load_iris()
    dados = pd.DataFrame(iris.data, columns=iris.feature_names)
    
    # Criar avaliador
    avaliador = AvaliadorNaoSupervisionado(dados)
    
    # Executar avaliações
    avaliador.avaliar_agrupamento_particional(k_min=2, k_max=8)
    avaliador.avaliar_agrupamento_hierarquico()
    avaliador.avaliar_reducao_dimensionalidade()
    avaliador.avaliar_deteccao_anomalias()
    
    # Gerar relatório
    relatorio = avaliador.gerar_relatorio_completo()
    
    # Exportar para planilha
    avaliador.exportar_para_planilha('iris_avaliacao.xlsx')
    
    # Plotar resultados
    avaliador.plotar_resultados()
    
    return avaliador, relatorio


if __name__ == "__main__":
    # Executar exemplo
    avaliador, relatorio = exemplo_uso_iris()
    
    print("\n" + "="*60)
    print("RESUMO DOS MELHORES RESULTADOS")
    print("="*60)
    
    melhores = relatorio['melhores_resultados']
    print(f"Melhor K para K-means: {melhores['agrupamento_particional']['melhor_k']}")
    print(f"Melhor Score Silhueta: {melhores['agrupamento_particional']['silhueta_score']:.3f}")
    print(f"Melhor método hierárquico: {melhores['agrupamento_hierarquico']['melhor_metodo']}")
    print(f"Coeficiente Cofenético: {melhores['agrupamento_hierarquico']['coeficiente_cofenetico']:.3f}")
    print(f"Componentes PCA (95%): {melhores['reducao_dimensionalidade']['componentes_95_pct']}")
