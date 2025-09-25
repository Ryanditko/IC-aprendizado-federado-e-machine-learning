# VALORES PARA PLANILHA - FORMATO EXATO

## 📊 Tabela Principal (Copie diretamente para sua planilha)

| Técnica | Acurácia | Precisão | Recall | F1-score | Observações |
|---------|----------|----------|--------|----------|-------------|
| K-means | 0.582 | | | | Coeficiente de Silhueta - Dataset Iris |
| Isolation forest | | | | | Detecta anomalias por isolamento |
| Dbscan | | | | | Clustering baseado em densidade |
| | | | | | |
| **Baseline** | | | | | |
| Z-score | | | | | Normalização por desvio padrão |
| Quantis | | | | | Normalização por percentis |

## 📈 Valores Adicionais Disponíveis

### Dataset Iris:
- K-means (Silhueta): **0.582**
- Clustering Hierárquico (Cofenético): **0.854**
- PCA (Variância 95%): **0.959** (2 componentes)

### Dataset Penguins:
- K-means (Silhueta): **0.532**
- Clustering Hierárquico (Cofenético): **0.845**
- PCA (Variância 95%): **0.950** (3 componentes)

## 🎯 Para Preenchimento Célula por Célula:

**Coluna B (Acurácia) - Linha 2 (K-means):** `0.582`

**Coluna L (Observações):**
- Linha 2: `Coeficiente de Silhueta - Dataset Iris`
- Linha 3: `Detecta anomalias por isolamento`
- Linha 4: `Clustering baseado em densidade`
- Linha 7: `Normalização por desvio padrão`
- Linha 8: `Normalização por percentis`

## 💡 Se quiser adicionar mais linhas:

| Técnica | Acurácia | Observações |
|---------|----------|-------------|
| Hierárquico | 0.854 | Coeficiente Cofenético - Dataset Iris |
| PCA | 0.959 | Variância explicada 95% - Dataset Iris |
| K-means (Penguins) | 0.532 | Coeficiente de Silhueta - Dataset Penguins |
| Hierárquico (Penguins) | 0.845 | Coeficiente Cofenético - Dataset Penguins |
| PCA (Penguins) | 0.950 | Variância explicada 95% - Dataset Penguins |
