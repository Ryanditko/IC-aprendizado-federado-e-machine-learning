# 🔍 Técnicas de Detecção de Outliers - Análise Detalhada

## 📊 Resumo Executivo

Este documento apresenta uma análise detalhada das técnicas de detecção de outliers implementadas no projeto, incluindo fundamentos teóricos, implementação prática e resultados obtidos.

## 1. Z-Score (Pontuação Padrão)

### 🔬 Fundamento Teórico
O Z-Score mede quantos desvios padrão um ponto está distante da média. É baseado na distribuição normal padrão.

**Fórmula:**
```
Z = (x - μ) / σ
```
Onde:
- x = valor observado
- μ = média da população
- σ = desvio padrão

### ⚙️ Implementação
```python
def z_score_detection(self, threshold=3):
    z_scores = np.abs(stats.zscore(self.data[self.numeric_columns]))
    outliers = (z_scores > threshold).any(axis=1)
    return outliers
```

### 📈 Resultados
- **Outliers detectados**: 50 (exatamente os verdadeiros)
- **Acurácia**: 99,8%
- **Precisão**: 98,0%
- **Recall**: 98,0%
- **F1-Score**: 98,0%

### ✅ Vantagens
- Simples de implementar e interpretar
- Computacionalmente eficiente
- Funciona bem com distribuições normais
- Resultado exato neste dataset

### ❌ Desvantagens
- Assume distribuição normal
- Sensível a outliers extremos na própria medição
- Pode não funcionar bem com distribuições assimétricas

---

## 2. IQR (Interquartile Range)

### 🔬 Fundamento Teórico
O método IQR identifica outliers usando quartis, sendo robusto a valores extremos. Define limites baseados na dispersão central dos dados.

**Limites de outliers:**
```
Limite inferior = Q1 - 1.5 × IQR
Limite superior = Q3 + 1.5 × IQR
```
Onde IQR = Q3 - Q1

### ⚙️ Implementação
```python
def iqr_detection(self, multiplier=1.5):
    outliers = np.zeros(len(self.data), dtype=bool)
    for col in self.numeric_columns:
        Q1 = self.data[col].quantile(0.25)
        Q3 = self.data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        col_outliers = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
        outliers |= col_outliers
    return outliers
```

### 📈 Resultados
- **Outliers detectados**: 58 (8 falsos positivos)
- **Acurácia**: 99,2%
- **Precisão**: 86,2%
- **Recall**: 100% (detectou todos os verdadeiros)
- **F1-Score**: 92,6%

### ✅ Vantagens
- Robusto a outliers extremos
- Não assume distribuição específica
- Fácil interpretação
- Excelente recall

### ❌ Desvantagens
- Pode ser muito conservador ou liberal dependendo do multiplicador
- Tratamento univariado por padrão

---

## 3. Isolation Forest

### 🔬 Fundamento Teórico
Algoritmo ensemble que isola outliers usando árvores de decisão aleatórias. Outliers são isolados mais rapidamente que pontos normais.

**Princípio**: Outliers requerem menos partições para serem isolados.

### ⚙️ Implementação
```python
def isolation_forest_detection(self, contamination=0.1, random_state=42):
    clf = IsolationForest(contamination=contamination, random_state=random_state)
    outliers = clf.fit_predict(self.data[self.numeric_columns]) == -1
    return outliers
```

### 📈 Resultados
- **Outliers detectados**: 105 (55 falsos positivos)
- **Acurácia**: 94,8%
- **Precisão**: 47,6%
- **Recall**: 100%
- **F1-Score**: 64,5%

### ✅ Vantagens
- Eficiente para grandes datasets
- Não assume distribuição específica
- Detecta outliers multivariados
- Bom recall

### ❌ Desvantagens
- Parâmetro de contaminação crítico
- Muitos falsos positivos neste caso
- Interpretabilidade limitada

---

## 4. Local Outlier Factor (LOF)

### 🔬 Fundamento Teórico
Mede o grau de "outlierness" baseado na densidade local comparada com a vizinhança. Identifica outliers locais em regiões de densidades variadas.

**LOF > 1**: Potencial outlier
**LOF ≈ 1**: Densidade similar aos vizinhos

### ⚙️ Implementação
```python
def lof_detection(self, n_neighbors=20, contamination=0.1):
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    outliers = clf.fit_predict(self.data[self.numeric_columns]) == -1
    return outliers
```

### 📈 Resultados
- **Outliers detectados**: 105 (todos falsos positivos)
- **Acurácia**: 85,2%
- **Precisão**: 0,0%
- **Recall**: 0,0%
- **F1-Score**: 0,0%

### ✅ Vantagens
- Detecta outliers locais
- Adaptável a diferentes densidades
- Considera contexto local

### ❌ Desvantagens
- Desempenho muito ruim neste dataset
- Sensível ao número de vizinhos
- Computacionalmente mais caro

---

## 5. DBSCAN

### 🔬 Fundamento Teórico
Algoritmo de clustering que classifica pontos em clusters, bordas ou ruído (outliers). Baseado em densidade e conectividade.

**Parâmetros principais:**
- `eps`: Raio de vizinhança
- `min_samples`: Mínimo de pontos para formar cluster

### ⚙️ Implementação
```python
def dbscan_detection(self, eps=0.5, min_samples=5):
    X_scaled = self.scaler.fit_transform(self.data[self.numeric_columns])
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(X_scaled)
    outliers = labels == -1
    return outliers
```

### 📈 Resultados
- **Outliers detectados**: 1 (muito conservador)
- **Acurácia**: 95,1%
- **Precisão**: 0,0%
- **Recall**: 0,0%
- **F1-Score**: 0,0%

### ✅ Vantagens
- Não requer número pré-definido de clusters
- Identifica ruído naturalmente
- Robusto a outliers

### ❌ Desvantagens
- Muito conservador com parâmetros padrão
- Sensível aos parâmetros eps e min_samples
- Perdeu quase todos os outliers

---

## 6. One-Class SVM

### 🔬 Fundamento Teórico
Cria uma fronteira de decisão ao redor dos dados normais. Mapeia dados para espaço de alta dimensão usando kernel.

**Parâmetro `nu`**: Fração esperada de outliers (similar à contaminação).

### ⚙️ Implementação
```python
def one_class_svm_detection(self, nu=0.1, gamma='scale'):
    X_scaled = self.scaler.fit_transform(self.data[self.numeric_columns])
    clf = OneClassSVM(nu=nu, gamma=gamma)
    outliers = clf.fit_predict(X_scaled) == -1
    return outliers
```

### 📈 Resultados
- **Outliers detectados**: 105 (55 falsos positivos)
- **Acurácia**: 93,2%
- **Precisão**: 40,0%
- **Recall**: 84,0%
- **F1-Score**: 54,2%

### ✅ Vantagens
- Flexível com diferentes kernels
- Pode capturar fronteiras complexas
- Bom recall

### ❌ Desvantagens
- Muitos falsos positivos
- Sensível aos hiperparâmetros
- Menos interpretável

---

## 7. Elliptic Envelope

### 🔬 Fundamento Teórico
Assume que dados normais seguem distribuição gaussiana multivariada. Ajusta uma elipse que contém a maioria dos dados.

**Base**: Estimativa robusta de localização e covariância.

### ⚙️ Implementação
```python
def elliptic_envelope_detection(self, contamination=0.1):
    clf = EllipticEnvelope(contamination=contamination, random_state=42)
    outliers = clf.fit_predict(self.data[self.numeric_columns]) == -1
    return outliers
```

### 📈 Resultados
- **Outliers detectados**: 105 (55 falsos positivos)
- **Acurácia**: 94,8%
- **Precisão**: 47,6%
- **Recall**: 100%
- **F1-Score**: 64,5%

### ✅ Vantagens
- Bom para dados gaussianos
- Considera correlações entre variáveis
- Estimativa robusta de covariância

### ❌ Desvantagens
- Assume distribuição gaussiana
- Muitos falsos positivos neste caso

---

## 📊 Análise Comparativa

### Ranking por Métrica

**Por Acurácia:**
1. Z-Score (99,8%)
2. IQR (99,2%)
3. Isolation Forest/Elliptic Envelope (94,8%)

**Por Precisão:**
1. Z-Score (98,0%)
2. IQR (86,2%)
3. Isolation Forest/Elliptic Envelope (47,6%)

**Por Recall:**
1. IQR, Isolation Forest, Elliptic Envelope (100%)
2. Z-Score (98,0%)
3. One-Class SVM (84,0%)

### 🎯 Conclusões

1. **Métodos estatísticos simples** (Z-Score, IQR) foram mais eficazes
2. **Métodos baseados em contaminação** sofreram com o parâmetro fixo
3. **DBSCAN e LOF** não se adequaram bem aos dados
4. **Ensemble** pode combinar pontos fortes de diferentes métodos

### 💡 Recomendações

- **Para dados similares**: Priorize Z-Score e IQR
- **Para exploração**: Use múltiplos métodos
- **Para otimização**: Ajuste hiperparâmetros específicos do dataset

---

*Documento atualizado: 21 de agosto de 2025*
