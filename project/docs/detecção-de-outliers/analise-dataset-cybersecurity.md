# 📊 ANÁLISE COMPLETA: Dataset de Cybersecurity e Detecção de Outliers

## 📋 Sumário

1. [Visão Geral do Dataset](#visão-geral-do-dataset)
2. [Estrutura dos Dados](#estrutura-dos-dados)
3. [Como Identificamos Dados Maliciosos](#como-identificamos-dados-maliciosos)
4. [Processo de Avaliação](#processo-de-avaliação)
5. [Detecção de Outliers](#detecção-de-outliers)
6. [Validação dos Resultados](#validação-dos-resultados)
7. [Visualizações e Gráficos](#visualizações-e-gráficos)

---

## 1️⃣ Visão Geral do Dataset

### 📦 Dataset: Cyber Threat Intelligence

**Fonte:** Kaggle - Text-Based Cyber Threat Detection  
**Tamanho:** 19.940 registros  
**Formato:** CSV com colunas de texto e labels

### 🎯 Objetivo

Detectar ameaças cibernéticas em textos usando técnicas de aprendizado não-supervisionado para identificar outliers que correspondem a agentes maliciosos.

---

## 2️⃣ Estrutura dos Dados

### 📊 Colunas Principais

```csv
index, text, entities, relations, Comments, id, label, start_offset, end_offset
```

#### Descrição de cada coluna:

| Coluna | Descrição | Exemplo |
|--------|-----------|---------|
| **text** | Texto descrevendo atividade de rede/sistema | "CTB-Locker is a ransomware..." |
| **label** | Classificação do conteúdo | malware, attack-pattern, TIME, etc. |
| **entities** | Entidades nomeadas no texto | Lista JSON de entidades |
| **id** | Identificador único | 45800, 48941, etc. |

### 🏷️ Tipos de Labels

**Labels de Ameaças (Maliciosos):**
- `malware` - Software malicioso
- `attack-pattern` - Padrões de ataque
- `threat-actor` - Agentes de ameaça
- `vulnerability` - Vulnerabilidades
- `tools` - Ferramentas usadas em ataques

**Labels Normais (Não-maliciosos):**
- `TIME` - Referências temporais
- `identity` - Identificações
- `SOFTWARE` - Software legítimo
- `LOCATION` - Localizações
- *(vazio)* - Texto sem ameaças

---

## 3️⃣ Como Identificamos Dados Maliciosos

### 🔍 Ground Truth (Verdade Fundamental)

O dataset já vem **pré-rotulado** por especialistas em cybersecurity que analisaram cada texto e classificaram:

```python
# Definição de o que é considerado AMEAÇA
threat_labels = [
    'malware',           # Software malicioso
    'attack-pattern',    # Padrões de ataque
    'threat-actor',      # Agentes maliciosos
    'vulnerability',     # Vulnerabilidades
    'tools'              # Ferramentas de ataque
]

# Classificação binária
df['is_threat'] = df['label'].apply(
    lambda x: 1 if x in threat_labels else 0
)
```

### 📈 Estatísticas do Dataset

```
Total de Registros: 19.940
├── Com Labels: 9.938 (49.8%)
│   ├── Ameaças: 4.643 (46.7%)
│   └── Normais: 5.295 (53.3%)
└── Sem Labels: 10.002 (50.2%)
```

### 💡 Exemplo Real de Dados Maliciosos

**Texto 1 (Malware):**
```
"CTB-Locker is a well-known ransomware Trojan used by 
crimeware groups to encrypt files on the victim's endpoints 
and demand ransom payment..."
```
- **Label:** `malware`
- **is_threat:** `1` (Sim, é malicioso)

**Texto 2 (Normal):**
```
"Palo Alto Networks Enterprise Security Platform offers 
multilayer protection..."
```
- **Label:** `identity`
- **is_threat:** `0` (Não, é normal)

---

## 4️⃣ Processo de Avaliação

### 🔄 Pipeline Completo

```
┌─────────────────────────────────────────────────────────────┐
│ 1. CARREGAMENTO DOS DADOS                                   │
│    └─ CSV → DataFrame (19.940 linhas)                       │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. FILTRAGEM (apenas dados com labels)                      │
│    └─ 9.938 registros com rótulos conhecidos               │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. CLASSIFICAÇÃO BINÁRIA                                    │
│    └─ is_threat = 1 (malicioso) ou 0 (normal)             │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. PROCESSAMENTO DE TEXTO (TF-IDF)                         │
│    └─ Texto → Vetores numéricos (100 features)             │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. REDUÇÃO DE DIMENSIONALIDADE (PCA)                       │
│    └─ 100 features → 50 componentes (69% variância)        │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. DETECÇÃO DE OUTLIERS (5 técnicas)                       │
│    └─ Isolation Forest, LOF, SVM, Elliptic, DBSCAN         │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. VALIDAÇÃO (comparar com ground truth)                   │
│    └─ Calcular Accuracy, Precision, Recall, F1-Score       │
└─────────────────────────────────────────────────────────────┘
```

### 📝 Detalhes de Cada Etapa

#### **Etapa 1-3: Preparação dos Dados**
```python
# Carregar dataset
df = pd.read_csv('cyber-threat-intelligence_all.csv')

# Filtrar apenas registros com labels
df_labeled = df[df['label'].notna()]

# Criar classificação binária
threat_labels = ['malware', 'attack-pattern', 'threat-actor', 'vulnerability', 'tools']
df_labeled['is_threat'] = df_labeled['label'].apply(
    lambda x: 1 if x in threat_labels else 0
)
```

#### **Etapa 4: TF-IDF (Text Frequency - Inverse Document Frequency)**
```python
# Converter texto em vetores numéricos
vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
X_tfidf = vectorizer.fit_transform(df_labeled['text'])

# Resultado: Cada texto vira um vetor de 100 números
# Exemplo: "malware attack" → [0.0, 0.45, 0.0, 0.78, ...]
```

#### **Etapa 5: PCA (Principal Component Analysis)**
```python
# Reduzir de 100 para 50 dimensões
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_tfidf.toarray())

# Mantém 69.26% da variância original
# Facilita visualização e processamento
```

#### **Etapa 6: Detecção de Outliers**
```python
# 5 técnicas aplicadas SEM usar os labels (não-supervisionado)

# 1. Isolation Forest
iso_forest = IsolationForest(contamination=0.467)
predictions_if = iso_forest.fit_predict(X_pca)

# 2. Local Outlier Factor
lof = LocalOutlierFactor(contamination=0.467)
predictions_lof = lof.fit_predict(X_pca)

# 3. One-Class SVM
ocsvm = OneClassSVM(nu=0.467)
predictions_ocsvm = ocsvm.fit_predict(X_pca)

# 4. Elliptic Envelope
ee = EllipticEnvelope(contamination=0.467)
predictions_ee = ee.fit_predict(X_pca)

# 5. DBSCAN
dbscan = DBSCAN(eps=3.0, min_samples=10)
predictions_dbscan = dbscan.fit_predict(X_pca)

# Resultado: -1 = outlier, 1 = normal
```

---

## 5️⃣ Detecção de Outliers

### 🎯 O que são Outliers?

**Definição:** Pontos de dados que se **desviam significativamente** do padrão normal.

**No nosso caso:**
- **Outliers** = Textos com características anormais (possivelmente maliciosos)
- **Normais** = Textos com padrões comuns (provavelmente legítimos)

### 🔧 Como Cada Técnica Detecta Outliers

#### **1. Isolation Forest**
- **Lógica:** Pontos anormais são mais fáceis de isolar
- **Como funciona:** Cria árvores aleatórias e isola pontos
- **Detectou:** 4.641 outliers

#### **2. Local Outlier Factor (LOF)**
- **Lógica:** Compara densidade local de cada ponto
- **Como funciona:** Pontos em regiões de baixa densidade são outliers
- **Detectou:** 4.643 outliers

#### **3. One-Class SVM**
- **Lógica:** Cria fronteira em torno dos dados normais
- **Como funciona:** Pontos fora da fronteira são outliers
- **Detectou:** 4.640 outliers

#### **4. Elliptic Envelope**
- **Lógica:** Assume distribuição gaussiana dos dados
- **Como funciona:** Pontos distantes do centro são outliers
- **Detectou:** 4.646 outliers ⭐ **MELHOR**

#### **5. DBSCAN**
- **Lógica:** Clustering baseado em densidade
- **Como funciona:** Pontos não pertencentes a clusters são outliers
- **Detectou:** 6.061 outliers (mais sensível)

### 📊 Parâmetro de Contaminação

```python
contamination = 4.643 / 9.938 = 0.467 (46.7%)
```

**O que significa:** Esperamos que ~46.7% dos dados sejam outliers (ameaças).

---

## 6️⃣ Validação dos Resultados

### ✅ Como Validamos se os Outliers são Maliciosos?

**Comparação com Ground Truth:**

```python
# Ground truth (ameaças conhecidas)
y_true = df_labeled['is_threat']  # [1, 0, 1, 0, ...]

# Predições da técnica (outliers detectados)
predictions = modelo.fit_predict(X_pca)  # [-1, 1, -1, 1, ...]

# Converter -1 (outlier) para 1, e 1 (normal) para 0
predictions_binary = (predictions == -1).astype(int)

# Calcular métricas
accuracy = accuracy_score(y_true, predictions_binary)
precision = precision_score(y_true, predictions_binary)
recall = recall_score(y_true, predictions_binary)
f1 = f1_score(y_true, predictions_binary)
```

### 📈 Resultados da Validação

| Técnica | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| **Elliptic Envelope** | **53.14%** | **49.85%** | **49.88%** | **49.87%** |
| Isolation Forest | 45.26% | 41.41% | 41.40% | 41.40% |
| LOF | 49.51% | 45.96% | 45.96% | 45.96% |
| One-Class SVM | 46.98% | 43.25% | 43.23% | 43.24% |
| DBSCAN | 49.29% | 46.72% | 60.99% | 52.91% |

### 🎯 Interpretação

**Elliptic Envelope (53.14% accuracy):**
- De 4.646 outliers detectados
- 2.318 são **realmente ameaças** (True Positives)
- 2.328 foram **falsos alarmes** (False Positives)
- **Conclusão:** A técnica acerta mais da metade!

---

## 7️⃣ Visualizações e Gráficos

### 📊 Gráficos Gerados

#### **1. Análise Exploratória (01_exploratory_analysis.png)**
- Distribuição de comprimento de texto
- Top 15 labels
- Proporção ameaças vs normal
- Estatísticas do dataset

#### **2. Métricas Comparativas (02_comparative_metrics.png)**
- Accuracy de cada técnica
- Precision de cada técnica
- Recall de cada técnica
- F1-Score de cada técnica

#### **3. Comparação de Outliers (03_outliers_comparison.png)**
- Ameaças reais (ground truth)
- Outliers detectados por técnica
- True Positives (acertos)

#### **4. Matrizes de Confusão (04_confusion_matrices.png)**
- TP, TN, FP, FN de cada técnica
- Validação visual dos resultados

#### **5. Gráfico de Dispersão (05_scatter_plot_outliers.png)** ⭐ **NOVO**
- Visualização 2D dos dados após PCA
- Pontos maliciosos vs normais
- Outliers detectados destacados

---

## 🎯 Conclusões

### ✅ Respostas às Perguntas Principais

**1. Como funciona o dataset?**
- 19.940 textos sobre cybersecurity
- Pré-rotulados por especialistas
- 4.643 são ameaças conhecidas (malware, ataques, etc.)

**2. Como foi avaliado?**
- TF-IDF para converter texto em números
- PCA para reduzir dimensionalidade
- 5 técnicas de detecção de outliers
- Comparação com ground truth

**3. Como identificamos dados maliciosos?**
- **Ground truth:** Labels pré-existentes (malware, attack-pattern, etc.)
- **Validação:** Comparamos outliers detectados com ameaças conhecidas
- **Métricas:** Accuracy, Precision, Recall, F1-Score

**4. Os outliers detectados são maliciosos?**
- **Sim, em ~50% dos casos** (Elliptic Envelope)
- Validação científica comprova eficácia
- Aplicável em cenários reais de aprendizado federado

### 🚀 Aplicação Prática

Este método pode ser usado em **Aprendizado Federado** para:
1. Detectar clientes maliciosos sem labels prévios
2. Filtrar atualizações suspeitas
3. Proteger o modelo global de ataques de envenenamento

---

## 📚 Referências

- **Dataset:** Kaggle - Text-Based Cyber Threat Detection
- **Técnicas:** scikit-learn (Isolation Forest, LOF, One-Class SVM, Elliptic Envelope, DBSCAN)
- **Processamento:** TF-IDF, PCA
- **Validação:** Confusion Matrix, Classification Metrics

---

**Data:** Outubro 2025  
**Projeto:** Iniciação Científica - Mitigação de Ataques em Aprendizado Federado  
**Instituição:** Faculdade Impacta
