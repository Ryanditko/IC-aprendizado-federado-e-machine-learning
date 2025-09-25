# 🎭 DEMONSTRAÇÃO - Aprendizado Não Supervisionado

**IMPORTANTE**: Este é um relatório de DEMONSTRAÇÃO usando dados simulados.

## Objetivo da Demonstração

Demonstrar a implementação das três técnicas principais:

1. **K-Means** (Agrupamento Particional)
2. **AGNES** (Agrupamento Hierárquico)
3. **PCA** (Redução de Dimensionalidade)

## Dataset Simulado

- **Amostras**: 1,500
- **Features**: 8
- **Grupos simulados**: Normal (60%), Suspeito (30%), Malicioso (10%)

## Resultados da Análise

| Técnica             | Métrica Principal     |   Valor | Parâmetros              | Métricas Adicionais     | Qualidade   |
|:--------------------|:----------------------|--------:|:------------------------|:------------------------|:------------|
| K-means             | Silhouette Score      |  0.7225 | K = 2                   | WCSS: 6561, Sep: 6.3288 | Excelente   |
| AGNES (Hierárquico) | Coef. Cofenético (rc) |  0.9726 | Linkage: average, K = 2 | Silhouette: 0.9244      | Excelente   |
| PCA                 | Variância Explicada   |  0.9574 | Componentes: 6          | Redução: 25.0%          | Excelente   |

## Interpretação dos Resultados

### ✅ Sucessos da Demonstração

- **K-Means**: Silhouette Score = 0.7225 (Excelente qualidade)
- **Hierárquico**: Coef. Cofenético = 0.9726 (Excelente preservação)
- **PCA**: 95.7% da variância preservada com 6 componentes

### 🎯 Para Dados Reais

Para executar com dados reais da Kaggle:
1. Configure as credenciais da Kaggle
2. Execute: `python unsupervised_cybersecurity_analysis.py`
3. Compare os resultados com esta demonstração
