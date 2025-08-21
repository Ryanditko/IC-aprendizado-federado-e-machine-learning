# 📏 Metodologia de Avaliação de Modelos de Detecção de Outliers

## 🎯 Objetivo

Este documento detalha a metodologia utilizada para avaliar e comparar os diferentes algoritmos de detecção de outliers implementados no projeto.

## 📊 Métricas de Avaliação

### 1. Métricas Básicas de Classificação

#### Matriz de Confusão
```
                 Predito
              Normal  Outlier
Real Normal     TN     FP
     Outlier    FN     TP
```

Onde:
- **TP (True Positives)**: Outliers corretamente identificados
- **TN (True Negatives)**: Pontos normais corretamente identificados  
- **FP (False Positives)**: Pontos normais incorretamente marcados como outliers
- **FN (False Negatives)**: Outliers não detectados

#### Acurácia
```
Acurácia = (TP + TN) / (TP + TN + FP + FN)
```
**Interpretação**: Proporção de predições corretas sobre o total.

#### Precisão
```
Precisão = TP / (TP + FP)
```
**Interpretação**: Dos pontos marcados como outliers, quantos realmente são outliers.

#### Recall (Sensibilidade)
```
Recall = TP / (TP + FN)
```
**Interpretação**: Dos outliers reais, quantos foram detectados pelo modelo.

#### F1-Score
```
F1-Score = 2 × (Precisão × Recall) / (Precisão + Recall)
```
**Interpretação**: Média harmônica entre precisão e recall.

### 2. Métricas Específicas para Detecção de Outliers

#### Taxa de Falsos Positivos (FPR)
```
FPR = FP / (FP + TN)
```
**Interpretação**: Proporção de pontos normais incorretamente classificados como outliers.

#### Taxa de Falsos Negativos (FNR)
```
FNR = FN / (FN + TP)
```
**Interpretação**: Proporção de outliers que não foram detectados.

## 📈 Resultados Detalhados por Método

### Z-Score (threshold=3)

**Matriz de Confusão:**
```
           Predito
        Normal  Outlier
Normal    999      1    (1000 total)
Outlier     1     49    (50 total)
```

**Métricas:**
- Acurácia: 99,81% (1048/1050)
- Precisão: 98,00% (49/50)
- Recall: 98,00% (49/50)
- F1-Score: 98,00%
- FPR: 0,10% (1/1000)
- FNR: 2,00% (1/50)

**Análise**: Desempenho quase perfeito, com apenas 1 falso positivo e 1 falso negativo.

---

### IQR (multiplier=1.5)

**Matriz de Confusão:**
```
           Predito
        Normal  Outlier
Normal    992      8    (1000 total)
Outlier     0     50    (50 total)
```

**Métricas:**
- Acurácia: 99,24% (1042/1050)
- Precisão: 86,21% (50/58)
- Recall: 100,00% (50/50)
- F1-Score: 92,59%
- FPR: 0,80% (8/1000)
- FNR: 0,00% (0/50)

**Análise**: Recall perfeito, mas com alguns falsos positivos. Método mais conservador.

---

### Isolation Forest (contamination=0.1)

**Matriz de Confusão:**
```
           Predito
        Normal  Outlier
Normal    945     55    (1000 total)
Outlier     0     50    (50 total)
```

**Métricas:**
- Acurácia: 94,76% (995/1050)
- Precisão: 47,62% (50/105)
- Recall: 100,00% (50/50)
- F1-Score: 64,52%
- FPR: 5,50% (55/1000)
- FNR: 0,00% (0/50)

**Análise**: Recall excelente, mas muitos falsos positivos devido ao parâmetro de contaminação.

---

### Local Outlier Factor (LOF)

**Matriz de Confusão:**
```
           Predito
        Normal  Outlier
Normal    895    105    (1000 total)
Outlier    50      0    (50 total)
```

**Métricas:**
- Acurácia: 85,24% (895/1050)
- Precisão: 0,00% (0/105)
- Recall: 0,00% (0/50)
- F1-Score: 0,00%
- FPR: 10,50% (105/1000)
- FNR: 100,00% (50/50)

**Análise**: Falha completa - não detectou nenhum outlier real, apenas falsos positivos.

---

### DBSCAN (eps=0.5, min_samples=5)

**Matriz de Confusão:**
```
           Predito
        Normal  Outlier
Normal    999      1    (1000 total)
Outlier    50      0    (50 total)
```

**Métricas:**
- Acurácia: 95,14% (999/1050)
- Precisão: 0,00% (0/1)
- Recall: 0,00% (0/50)
- F1-Score: 0,00%
- FPR: 0,10% (1/1000)
- FNR: 100,00% (50/50)

**Análise**: Extremamente conservador - quase não detectou outliers.

---

### One-Class SVM (nu=0.1)

**Matriz de Confusão:**
```
           Predito
        Normal  Outlier
Normal    937     63    (1000 total)
Outlier     8     42    (50 total)
```

**Métricas:**
- Acurácia: 93,24% (979/1050)
- Precisão: 40,00% (42/105)
- Recall: 84,00% (42/50)
- F1-Score: 54,19%
- FPR: 6,30% (63/1000)
- FNR: 16,00% (8/50)

**Análise**: Balanceamento médio entre precisão e recall, mas com muitos falsos positivos.

---

### Elliptic Envelope (contamination=0.1)

**Matriz de Confusão:**
```
           Predito
        Normal  Outlier
Normal    945     55    (1000 total)
Outlier     0     50    (50 total)
```

**Métricas:**
- Acurácia: 94,76% (995/1050)
- Precisão: 47,62% (50/105)
- Recall: 100,00% (50/50)
- F1-Score: 64,52%
- FPR: 5,50% (55/1000)
- FNR: 0,00% (0/50)

**Análise**: Idêntico ao Isolation Forest - parâmetro de contaminação domina o resultado.

## 🔗 Análise de Concordância entre Métodos

### Matriz de Concordância (Acurácia)

```
                   Z-Score  IQR   Isol.F  LOF   DBSCAN  SVM   Ellip.
Z-Score             100%   99.2%  94.8%  85.4%  95.3%  93.2%  94.8%
IQR                 99.2%  100%   95.5%  86.0%  94.6%  94.0%  95.5%
Isolation Forest    94.8%  95.5%  100%   84.6%  90.1%  94.5%  95.6%
LOF                 85.4%  86.0%  84.6%  100%   90.1%  89.9%  88.6%
DBSCAN              95.3%  94.6%  90.1%  90.1%  100%   90.1%  90.1%
One-Class SVM       93.2%  94.0%  94.5%  89.9%  90.1%  100%   96.4%
Elliptic Envelope   94.8%  95.5%  95.6%  88.6%  90.1%  96.4%  100%
```

### Insights da Concordância

1. **Maior concordância**: Z-Score e IQR (99,2%)
2. **Menor concordância**: LOF com outros métodos (~85-90%)
3. **Métodos similares**: Isolation Forest e Elliptic Envelope (95,6%)

## 🎭 Análise Ensemble

### Metodologia Ensemble
- **Tipo**: Voto majoritário (hard voting)
- **Critério**: Ponto é outlier se ≥ 4 dos 7 métodos assim classificarem
- **Objetivo**: Combinar pontos fortes de diferentes abordagens

### Resultados Ensemble

**Matriz de Confusão:**
```
           Predito
        Normal  Outlier
Normal    977     23    (1000 total)
Outlier     0     50    (50 total)
```

**Métricas:**
- Acurácia: 97,81% (1027/1050)
- Precisão: 68,49% (50/73)
- Recall: 100,00% (50/50)
- F1-Score: 81,30%
- FPR: 2,30% (23/1000)
- FNR: 0,00% (0/50)

### Vantagens do Ensemble
- Recall perfeito (detecta todos os outliers)
- Reduz falsos positivos comparado a métodos individuais conservadores
- Mais robusto que métodos individuais

## 📊 Comparação Final

### Ranking por Aplicação

**Para Máxima Precisão:**
1. Z-Score (98,0%)
2. IQR (86,2%)
3. Ensemble (68,5%)

**Para Máximo Recall:**
1. IQR, Isolation Forest, Elliptic Envelope, Ensemble (100%)
2. Z-Score (98,0%)
3. One-Class SVM (84,0%)

**Para Balanceamento (F1-Score):**
1. Z-Score (98,0%)
2. IQR (92,6%)
3. Ensemble (81,3%)

**Para Mínimos Falsos Positivos:**
1. Z-Score (0,1%)
2. DBSCAN (0,1%)
3. IQR (0,8%)

## 🎯 Recomendações de Uso

### Por Cenário de Aplicação

**Detecção Crítica (alta precisão necessária):**
- Use Z-Score
- Tolerância baixa a falsos positivos

**Exploração de Dados (alta sensibilidade):**
- Use IQR ou Ensemble
- Importante não perder outliers reais

**Análise Conservadora:**
- Use Z-Score com threshold mais alto
- Ou combine Z-Score + IQR

**Análise Exploratória:**
- Use Ensemble para visão abrangente
- Compare múltiplos métodos

## 🔧 Limitações da Avaliação

### Limitações Identificadas

1. **Dataset Sintético**: Pode não refletir complexidade de dados reais
2. **Parâmetros Fixos**: Não houve otimização sistemática de hiperparâmetros
3. **Distribuição Bivariada**: Avaliação limitada a duas dimensões
4. **Outliers Artificiais**: Outliers inseridos podem ser diferentes de anomalias naturais

### Trabalhos Futuros

- Avaliação com datasets reais diversos
- Otimização automática de hiperparâmetros
- Teste com dados de alta dimensionalidade
- Análise de diferentes tipos de outliers (globais, locais, contextuais)

---

*Metodologia estabelecida: 21 de agosto de 2025*
