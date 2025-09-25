# RESUMO EXECUTIVO: Resultados da Avaliação dos Datasets

## 📊 **Dados Processados**
- **Dataset Iris**: 150 amostras, 4 características (sépalas e pétalas)
- **Dataset Penguins**: 342 amostras, 4 características numéricas

---

## 🎯 **Resultados Principais para sua Planilha**

### **Dataset Iris**
| Técnica | Métrica | Valor | Interpretação |
|---------|---------|-------|---------------|
| **K-means** | Coeficiente de Silhueta | **0.582** | Clusters moderadamente bem definidos |
| **K-means** | Índice Davies-Bouldin | **0.593** | Valor baixo = boa separação entre clusters |
| **Clustering Hierárquico** | Coeficiente Cofenético | **0.854** | Muito boa preservação das distâncias originais |
| **PCA** | Componentes para 95% variância | **2** | Dataset muito compressível |

### **Dataset Penguins**
| Técnica | Métrica | Valor | Interpretação |
|---------|---------|-------|---------------|
| **K-means** | Coeficiente de Silhueta | **0.532** | Clusters moderadamente bem definidos |
| **K-means** | Índice Davies-Bouldin | **0.714** | Boa separação entre clusters |
| **Clustering Hierárquico** | Coeficiente Cofenético | **0.845** | Muito boa preservação das distâncias originais |
| **PCA** | Componentes para 95% variância | **3** | Necessita mais componentes que Iris |

---

## 📈 **Análise Comparativa**

### **Agrupamento (K-means)**
- **Melhor k para ambos**: 2 clusters
- **Iris vs Penguins**: Iris tem silhueta ligeiramente melhor (0.582 vs 0.532)
- **Conclusão**: Ambos datasets apresentam estrutura natural de 2 grupos

### **Clustering Hierárquico**
- **Melhor método para ambos**: Average linkage
- **Resultados similares**: Iris (0.854) vs Penguins (0.845)
- **Conclusão**: Ambos preservam muito bem as distâncias originais

### **Redução de Dimensionalidade (PCA)**
- **Iris**: 2 componentes capturam 95% da variância
  - PC1: 73% da variância
  - PC2: 23% da variância
- **Penguins**: 3 componentes necessários para 95%
  - PC1: 68.8% da variância
  - PC2: 19.3% da variância
- **Conclusão**: Iris é mais compressível que Penguins

---

## 🔍 **Valores Específicos para Preenchimento da Planilha**

### **Agrupamento Particional**
```
Iris - K-means:
- Melhor k: 2
- Silhueta: 0.582
- Davies-Bouldin: 0.593

Penguins - K-means:
- Melhor k: 2
- Silhueta: 0.532
- Davies-Bouldin: 0.714
```

### **Agrupamento Hierárquico**
```
Iris - Hierárquico:
- Melhor método: average
- Coeficiente Cofenético: 0.854

Penguins - Hierárquico:
- Melhor método: average
- Coeficiente Cofenético: 0.845
```

### **Redução de Dimensionalidade**
```
Iris - PCA:
- Componentes 95%: 2
- Variância PC1: 0.730
- Variância PC2: 0.229

Penguins - PCA:
- Componentes 95%: 3
- Variância PC1: 0.688
- Variância PC2: 0.193
```

---

## 📝 **Interpretação dos Resultados**

### **1. Coeficiente de Silhueta**
- **Escala**: -1 a 1
- **Interpretação**:
  - 0.5-0.7: Estrutura moderada de clusters ✅
  - > 0.7: Estrutura forte de clusters
  - < 0.3: Estrutura fraca

### **2. Índice Davies-Bouldin**
- **Escala**: 0 a ∞ (menor é melhor)
- **Interpretação**:
  - < 1.0: Boa separação entre clusters ✅
  - > 2.0: Clusters mal separados

### **3. Coeficiente Cofenético**
- **Escala**: 0 a 1
- **Interpretação**:
  - > 0.8: Excelente preservação ✅
  - 0.6-0.8: Boa preservação
  - < 0.6: Preservação insatisfatória

### **4. Variância Explicada (PCA)**
- **95% da variância**: Limiar comum para preservar informação
- **Menos componentes**: Dataset mais compressível

---

## 🎯 **Conclusões Principais**

1. **Ambos datasets têm estrutura natural de 2 clusters** (confirmado por K-means)
2. **Clustering hierárquico funciona muito bem** (coeficientes > 0.84)
3. **Iris é mais compressível** (2 vs 3 componentes para 95% variância)
4. **Métodos não supervisionados revelam padrões claros** nos dados

---

## 📁 **Arquivos Gerados**
- `resultados_avaliacao.csv` - Tabela completa de resultados
- `guia-tecnicas-avaliacao.md` - Guia completo das técnicas
- `avaliacao_simples.py` - Script de avaliação automática

**Use estes valores diretamente em sua planilha de avaliação!**

---

## 📋 **Formato Exato da sua Planilha**

| Técnica | Acurácia | Precisão | Recall | F1-score | Observações |
|---------|----------|----------|--------|----------|-------------|
| K-means | 0.582 | | | | Coeficiente de Silhueta - Dataset Iris |
| Isolation forest | | | | | Detecta anomalias por isolamento |
| Dbscan | | | | | Clustering baseado em densidade |
| | | | | | |
| **Baseline** | | | | | |
| Z-score | | | | | Normalização por desvio padrão |
| Quantis | | | | | Normalização por percentis |
- Valores representam o Coeficiente de Correlação Cofenética
- Mede preservação das distâncias originais no dendrograma (0 a 1, maior é melhor)
- Método utilizado: Average linkage

### **Redução de Dimensionalidade (PCA)**
| Técnica | Acurácia | Precisão | Recall | F1-score |
|---------|----------|----------|--------|----------|
| PCA (Iris) | 0.959 | - | - | - |
| PCA (Penguins) | 0.950 | - | - | - |

**Observações:**
- Valores representam a Variância Acumulada preservada
- Iris: 2 componentes preservam 95.9% da variância
- Penguins: 3 componentes preservam 95.0% da variância

### **Baseline (Para Comparação)**
| Técnica | Acurácia | Precisão | Recall | F1-score |
|---------|----------|----------|--------|----------|
| Z-score | - | - | - | - |
| Quantis | - | - | - | - |

**Observações:**
- Z-score e Quantis são técnicas de normalização, não de avaliação
- Utilizadas no pré-processamento dos dados antes da aplicação das técnicas principais

---

## 🔢 **Valores Numéricos Específicos para Cópia**

### **Para a Coluna "Acurácia" da sua Planilha:**
```
K-means (Iris): 0.582
K-means (Penguins): 0.532
Hierárquico (Iris): 0.854  
Hierárquico (Penguins): 0.845
PCA (Iris): 0.959
PCA (Penguins): 0.950
```

### **Métricas Adicionais Disponíveis:**
```
Davies-Bouldin (Iris): 0.593
Davies-Bouldin (Penguins): 0.714
Componentes PCA 95% (Iris): 2
Componentes PCA 95% (Penguins): 3
```

---
