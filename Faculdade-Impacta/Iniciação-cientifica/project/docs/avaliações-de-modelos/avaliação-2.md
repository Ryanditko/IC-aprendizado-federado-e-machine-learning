METODOLOGIA DE AVALIAÇÃO DE MODELOS EM APRENDIZADO NÃO SUPERVISIONADO

Resumo:
Este documento aborda metodologias de avaliação para modelos de aprendizado não supervisionado, focalizando três áreas principais: agrupamento particional, agrupamento hierárquico e redução de dimensionalidade. Diferentemente do aprendizado supervisionado, a ausência de rótulos verdadeiros demanda métricas intrínsecas e critérios específicos de avaliação.

1. Desafios na Avaliação Não Supervisionada

1.1 Características Distintivas:
- Ausência de ground truth (rótulos verdadeiros)
- Métricas baseadas em propriedades internas dos dados
- Avaliação subjetiva em alguns contextos
- Dependência de conhecimento do domínio

1.2 Abordagens de Validação:
- Métricas internas: Baseadas na estrutura dos dados
- Métricas externas: Quando rótulos estão disponíveis (validação)
- Avaliação visual: Inspeção gráfica dos resultados

2. Avaliação de Agrupamento Particional

2.1 Métricas Internas:

Coesão Intracluster (Within-Cluster Sum of Squares - WCSS):
- Definição: Soma das distâncias quadráticas de pontos ao centróide do cluster
- Interpretação: Valores menores indicam clusters mais compactos
- Fórmula: WCSS = Σᵢ Σₓ∈Cᵢ ||x - μᵢ||²

Separação Intercluster:
- Definição: Distância entre centróides de clusters diferentes
- Interpretação: Valores maiores indicam melhor separação
- Objetivo: Maximizar separação mantendo coesão

Coeficiente de Silhueta:
- Definição: Combinação de coesão e separação
- Fórmula: s(i) = (b(i) - a(i)) / max(a(i), b(i))
  * a(i): distância média intracluster
  * b(i): distância média ao cluster mais próximo
- Interpretação:
  * s(i) ≈ 1: Ponto bem agrupado
  * s(i) ≈ 0: Ponto na fronteira entre clusters
  * s(i) < 0: Possivelmente no cluster errado
- Valor médio: silhueta global do modelo

Índice Davies-Bouldin:
- Definição: Razão média entre dispersão intra e inter-cluster
- Interpretação: Valores menores indicam melhor agrupamento
- Fórmula: DB = (1/k) Σᵢ maxⱼ≠ᵢ [(σᵢ + σⱼ)/d(cᵢ,cⱼ)]

Índice Calinski-Harabasz:
- Definição: Razão entre dispersão entre clusters e dentro de clusters
- Interpretação: Valores maiores indicam melhor definição de clusters
- Adequado para comparação de diferentes k

2.2 Métricas Externas (quando rótulos disponíveis):
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Homogeneidade e Completude
- V-measure

2.3 Implementação:
```python
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

# Coeficiente de Silhueta
silhouette = silhouette_score(X, labels)

# Índice Davies-Bouldin
davies_bouldin = davies_bouldin_score(X, labels)

# Índice Calinski-Harabasz
calinski = calinski_harabasz_score(X, labels)

print(f"Silhueta: {silhouette:.3f}")
print(f"Davies-Bouldin: {davies_bouldin:.3f}")
print(f"Calinski-Harabasz: {calinski:.3f}")
```

3. Avaliação de Agrupamento Hierárquico

3.1 Coeficiente de Correlação Cofenética:
- Definição: Correlação entre distâncias originais e distâncias cofenéticas (do dendrograma)
- Interpretação: Mede preservação da estrutura de distâncias
- Valores:
  * rc > 0.8: Excelente preservação
  * 0.6 < rc < 0.8: Boa preservação
  * rc < 0.6: Preservação insatisfatória

3.2 Implementação:
```python
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist

# Agrupamento hierárquico
Z = linkage(X, method='average')

# Cálculo do coeficiente cofenético
coph_corr, coph_dists = cophenet(Z, pdist(X))
print(f"Correlação Cofenética: {coph_corr:.4f}")
```

3.3 Avaliação Visual:
- Dendrograma: Análise da estrutura hierárquica
- Identificação de "saltos" significativos (número ótimo de clusters)
- Consistência com conhecimento do domínio

4. Avaliação de Redução de Dimensionalidade

4.1 PCA (Principal Component Analysis):

Variância Explicada:
- Definição: Proporção da variância total capturada
- Critério comum: Reter componentes que explicam ≥ 95% da variância
- Análise individual e acumulada

Implementação:
```python
from sklearn.decomposition import PCA

# PCA com todos os componentes
pca = PCA()
pca.fit(X)

# Análise de variância explicada
var_explicada = pca.explained_variance_ratio_
var_acumulada = var_explicada.cumsum()

print(f"Variância explicada por componente: {var_explicada}")
print(f"Variância acumulada: {var_acumulada}")

# Determinar número de componentes para 95%
n_comp_95 = (var_acumulada >= 0.95).argmax() + 1
print(f"Componentes para 95% variância: {n_comp_95}")
```

4.2 Erro de Reconstrução:
- Diferença entre dados originais e reconstruídos
- Menor erro indica melhor preservação

4.3 Métricas para t-SNE e UMAP:
- Trustworthiness: Vizinhos no espaço reduzido eram vizinhos no original
- Continuity: Vizinhos no original continuam próximos no reduzido
- Stress: Distorção global das distâncias

5. Seleção do Número Ótimo de Clusters

5.1 Método do Cotovelo (Elbow Method):
- Gráfico de WCSS vs número de clusters
- Identificação do "cotovelo" (ponto de diminuição marginal)

5.2 Gap Statistic:
- Comparação com distribuição de referência uniforme
- Seleciona k que maximiza a diferença (gap)

5.3 Implementação:
```python
import matplotlib.pyplot as plt

# Método do cotovelo
inertias = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(K_range, inertias, 'bx-')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia (WCSS)')
plt.title('Método do Cotovelo')
plt.show()
```

6. Boas Práticas de Avaliação

6.1 Múltiplas Métricas:
- Não confiar em métrica única
- Combinar métricas internas e visuais
- Considerar contexto do problema

6.2 Validação Contextual:
- Interpretabilidade dos clusters
- Alinhamento com conhecimento do domínio
- Utilidade prática dos agrupamentos

6.3 Análise de Sensibilidade:
- Robustez a diferentes inicializações
- Estabilidade dos clusters
- Impacto de outliers

7. Considerações Finais
A avaliação de modelos em aprendizado não supervisionado demanda abordagem multifacetada, combinando métricas quantitativas, análise visual e validação contextual. Diferentemente do aprendizado supervisionado, onde métricas como acurácia fornecem indicador claro, a avaliação não supervisionada requer interpretação cuidadosa e consideração de múltiplos aspectos. O uso criterioso de ferramentas computacionais, como Python e scikit-learn, viabiliza a aplicação sistemática dessas metodologias, permitindo ajuste e otimização de modelos baseados em métricas internas confiáveis.

Referências:
- Métricas: scikit-learn, scipy - Implementações de métricas de avaliação
- Metodologias: Rousseeuw (1987) - Silhouettes, Davies & Bouldin (1979)

Contribuições Esperadas:
- Framework de avaliação para modelos não supervisionados
- Diretrizes para seleção de métricas apropriadas
- Metodologia para validação de agrupamentos e redução dimensional

