# Resultados - Aprendizado Não Supervisionado em Cybersecurity

**Dataset:** 1,500 amostras, 8 features numéricas

## Tabela de Resultados

| Técnica             | Métrica Principal     |   Valor | Parâmetros              | Métricas Adicionais     | Qualidade   |
|:--------------------|:----------------------|--------:|:------------------------|:------------------------|:------------|
| K-means             | Silhouette Score      |  0.7225 | K = 2                   | WCSS: 6561, Sep: 6.3288 | Excelente   |
| AGNES (Hierárquico) | Coef. Cofenético (rc) |  0.9726 | Linkage: average, K = 2 | Silhouette: 0.9244      | Excelente   |
| PCA                 | Variância Explicada   |  0.9574 | Componentes: 6          | Redução: 25.0%          | Excelente   |

## Interpretação

### Agrupamento Particional (K-Means)
- **Silhouette Score**: 0.7225
- **Coesão (WCSS)**: 6561
- **Separação Intercluster**: 6.3288
- **Avaliação**: Excelente

### Agrupamento Hierárquico (AGNES)
- **Coeficiente Cofenético (rc)**: 0.9726
- **Método de Linkage**: average
- **Avaliação**: Excelente

### Redução de Dimensionalidade (PCA)
- **Variância Explicada**: 0.9574 (95.7%)
- **Componentes Utilizados**: 6
- **Redução Dimensional**: 25.0%
- **Avaliação**: Excelente
