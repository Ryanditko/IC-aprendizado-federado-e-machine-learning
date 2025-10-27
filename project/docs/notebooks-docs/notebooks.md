# Notebooks - Análises e Avaliações

## Visão Geral

Esta pasta contém análises completas de Machine Learning aplicadas a diferentes datasets, incluindo avaliações de modelos supervisionados, não supervisionados e detecção de outliers em contexto de cybersecurity.

## Estrutura de Pastas

```
notebooks/
├── cyber-outlier-detection/    # Detecção de ameaças cibernéticas
├── iris/                        # Avaliação dataset Iris
├── penguin/                     # Avaliação dataset Penguins
└── weight_height/              # Avaliação dataset Weight-Height
```

## 1. Cyber Outlier Detection

### Objetivo
Análise de detecção de outliers aplicada ao contexto de cybersecurity, com foco em identificação de agentes maliciosos através de técnicas de aprendizado não supervisionado.

### Arquivos

**cyber_threat_outlier_detection.py**
- Análise completa de detecção de outliers em dados de ameaças cibernéticas
- Implementa múltiplos algoritmos: Isolation Forest, LOF, One-Class SVM, Elliptic Envelope, DBSCAN
- Valida outliers detectados contra labels reais de ameaças
- Gera visualizações e relatórios detalhados

**outlier_detection_analysis.ipynb**
- Versão interativa da análise
- Permite experimentação com diferentes parâmetros
- Documentação step-by-step do processo

### Resultados
- **Melhor Técnica**: Elliptic Envelope (99.52% accuracy)
- **Segunda Melhor**: Isolation Forest (97.14% accuracy)
- Validação contra agentes maliciosos reais

## 2. Iris Dataset

### Objetivo
Avaliação completa de técnicas de classificação multiclasse e clustering no clássico dataset Iris.

### Características
- **Amostras**: 150
- **Features**: 4 (sepal length, sepal width, petal length, petal width)
- **Classes**: 3 espécies (Setosa, Versicolor, Virginica)

### Arquivos
- `iris_evaluation.ipynb`: Notebook interativo completo
- `run_iris.py`: Script Python executável

### Resultados Principais
- **Melhor Modelo Supervisionado**: SVM (93.33% accuracy)
- **Clustering**: Hierarchical (0.4803 Silhouette Score)
- **PCA**: 2 componentes explicam ~95% da variância

### Visualizações Geradas
- `iris_exploratory.png`: Distribuição das features
- `iris_supervised_metrics.png`: Métricas dos 6 modelos
- `iris_clustering_comparison.png`: K-Means vs Hierarchical
- `iris_pca_analysis.png`: Análise de componentes principais

## 3. Penguins Dataset

### Objetivo
Classificação de espécies de pinguins com features morfológicas e geográficas.

### Características
- **Amostras**: 333 (após limpeza)
- **Features**: 6 (bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, island, sex)
- **Classes**: 3 espécies (Adelie, Chinstrap, Gentoo)

### Arquivos
- `penguins_evaluation.ipynb`: Notebook interativo completo
- `run_penguins.py`: Script Python executável

### Resultados Principais
- **Melhor Modelo Supervisionado**: Random Forest, SVM, K-NN, Logistic Regression (**100% accuracy**)
- **Clustering**: Hierarchical (0.4548 Silhouette Score)
- **Encoding**: Variáveis categóricas (ilha, sexo) codificadas

### Visualizações Geradas
- `penguins_exploratory.png`: Bill vs Body dimensions, distribuições por ilha/sexo
- `penguins_supervised_metrics.png`: Métricas dos 6 modelos
- `penguins_clustering_comparison.png`: Comparação de clustering
- `penguins_pca_analysis.png`: Análise de componentes principais

## 4. Weight-Height Dataset

### Objetivo
Classificação binária de gênero baseada em altura e peso.

### Características
- **Amostras**: 10,000 (sintéticas)
- **Features**: 2 (Height, Weight)
- **Classes**: 2 (Male, Female)

### Arquivos
- `weight_height_evaluation.ipynb`: Notebook interativo completo
- `run_weight_height.py`: Script Python executável

### Resultados Principais
- **Melhor Modelo Supervisionado**: SVM (92.80% accuracy)
- **Clustering**: Hierarchical (0.4970 Silhouette Score)
- **Todos os modelos**: > 88% accuracy

### Visualizações Geradas
- `weight_height_exploratory.png`: Scatter plots, histogramas, boxplots
- `weight_height_supervised_metrics.png`: Métricas dos 6 modelos
- `weight_height_clustering_comparison.png`: Clustering com centróides

## Metodologia Geral

### Pipeline de Avaliação (Todos os Datasets)

#### 1. Carregamento e Exploração
- Importação dos dados
- Análise estatística descritiva
- Visualizações exploratórias
- Tratamento de valores faltantes

#### 2. Aprendizado Supervisionado

**Modelos Testados** (6 algoritmos):
1. **Decision Tree**: Árvore de decisão simples
2. **Random Forest**: Ensemble de árvores
3. **SVM**: Support Vector Machine
4. **Naive Bayes**: Classificador probabilístico
5. **K-NN**: K-Nearest Neighbors
6. **Logistic Regression**: Regressão logística

**Métricas Avaliadas**:
- Accuracy (acurácia global)
- Precision (precisão)
- Recall (sensibilidade)
- F1-Score (média harmônica)

**Meta de Performance**: ≥ 70% em todas as métricas

#### 3. Aprendizado Não Supervisionado

**Algoritmos de Clustering**:
1. **K-Means**: Clustering baseado em centróides
2. **Hierarchical**: Clustering hierárquico (linkage average)

**Métricas de Avaliação**:
- **Silhouette Score**: Coesão e separação dos clusters (-1 a 1)
- **Davies-Bouldin Index**: Similaridade intra-cluster vs inter-cluster

#### 4. Redução de Dimensionalidade
- **PCA (Principal Component Analysis)**
- Análise de variância explicada
- Visualização em 2D

#### 5. Detecção de Outliers (Cyber Threats)

**Técnicas Implementadas**:
1. **Isolation Forest**: Isola outliers usando árvores de decisão aleatórias
2. **Local Outlier Factor (LOF)**: Compara densidade local com vizinhança
3. **One-Class SVM**: Cria fronteira de decisão para dados normais
4. **Elliptic Envelope**: Assume distribuição gaussiana multivariada
5. **DBSCAN**: Clustering baseado em densidade, identifica noise como outliers

**Validação**:
- Comparação com labels reais de ameaças
- Métricas: Accuracy, Precision, Recall, F1-Score
- Matrizes de confusão

## Como Executar

### Pré-requisitos
```bash
# Instalar dependências
pip install pandas numpy scikit-learn matplotlib seaborn

# Ou usar o script de instalação automática
python install_dependencies.py
```

### Executar Análises

#### Cyber Threat Detection
```bash
cd notebooks/cyber-outlier-detection
python cyber_threat_outlier_detection.py
```

#### Iris Dataset
```bash
cd notebooks/iris
python run_iris.py
```

#### Penguins Dataset
```bash
cd notebooks/penguin
python run_penguins.py
```

#### Weight-Height Dataset
```bash
cd notebooks/weight_height
python run_weight_height.py
```

### Usar Notebooks Jupyter
```bash
# Iniciar Jupyter Notebook
jupyter notebook

# Navegar até a pasta desejada e abrir o .ipynb
```

## Resultados Consolidados

### Comparação de Performance Entre Datasets

| Dataset | Amostras | Features | Melhor Modelo | Accuracy | Silhouette |
|---------|----------|----------|---------------|----------|------------|
| **Iris** | 150 | 4 | SVM | 93.33% | 0.4803 |
| **Penguins** | 333 | 6 | Random Forest | **100%** | 0.4548 |
| **Weight-Height** | 10,000 | 2 | SVM | 92.80% | 0.4970 |
| **Cyber Threats** | 15,000 | 17 | Elliptic Envelope | **99.52%** | N/A |

### Arquivos Gerados por Dataset

#### Cyber Outlier Detection
**Pasta**: `cyber-outlier-detection/output_images/`
- `01_exploratory_analysis.png`
- `02_comparative_metrics.png`
- `03_outliers_comparison.png`
- `04_confusion_matrices.png`
- `evaluation_metrics.csv`

#### Iris
**Pasta**: `iris/output-images/`
- `iris_exploratory.png`
- `iris_supervised_metrics.png`
- `iris_clustering_comparison.png`
- `iris_pca_analysis.png`
- `iris_supervised_results.csv`
- `iris_unsupervised_results.csv`

#### Penguins
**Pasta**: `penguin/output-images/`
- `penguins_exploratory.png`
- `penguins_supervised_metrics.png`
- `penguins_clustering_comparison.png`
- `penguins_pca_analysis.png`
- `penguins_supervised_results.csv`
- `penguins_unsupervised_results.csv`

#### Weight-Height
**Pasta**: `weight_height/output-images/`
- `weight_height_exploratory.png`
- `weight_height_supervised_metrics.png`
- `weight_height_clustering_comparison.png`
- `weight_height_supervised_results.csv`
- `weight_height_unsupervised_results.csv`

## Interpretação dos Resultados

### Métricas Principais:

**Accuracy**: Proporção total de classificações corretas
- Valores altos indicam boa performance geral
- Meta: >= 70%

**Precision**: Proporção de outliers detectados que são realmente maliciosos
- Alta precision reduz falsos positivos
- Importante para evitar alarmes falsos

**Recall**: Proporção de maliciosos que foram detectados como outliers
- Alta recall garante detecção de ameaças
- Crítico para segurança

**F1-Score**: Média harmônica entre Precision e Recall
- Balanceia ambas as métricas
- Meta: >= 70%

## Aplicação em Aprendizado Federado

### Relevância para o Projeto:

1. **Detecção de Ataques por Envenenamento**
   - Outliers podem representar updates maliciosos de modelos
   - Técnicas não supervisionadas não dependem de labels

2. **Validação de Participantes**
   - Identificar clientes maliciosos em federações
   - Proteger integridade do modelo global

3. **Robustez do Sistema**
   - Múltiplos métodos aumentam confiabilidade
   - Ensemble de técnicas melhora detecção

## Insights e Descobertas

### Por Dataset

#### Iris
- **SVM** demonstrou melhor performance em dados bem separáveis
- Clustering captura bem a estrutura natural de 3 espécies
- 2 componentes PCA suficientes para 95% da variância

#### Penguins
- **Classificação perfeita** (100%) alcançada por múltiplos modelos
- Features morfológicas + geográficas são altamente discriminativas
- Dataset bem estruturado, ideal para aprendizado

#### Weight-Height
- Separação clara entre gêneros baseada em características antropométricas
- **SVM** eficaz mesmo com apenas 2 features
- Grande volume de dados (10k) melhora performance

#### Cyber Threats
- **Elliptic Envelope** mais eficaz para detecção de anomalias (99.52%)
- Validação contra agentes maliciosos reais confirma eficácia
- Múltiplas técnicas necessárias para robustez

### Conclusões Gerais

1. **Modelos Ensemble** (Random Forest) consistentemente performam bem
2. **SVM** eficaz para separação de classes complexas
3. **Clustering Hierárquico** geralmente supera K-Means
4. **Detecção de Outliers** viável para identificar ameaças em FL
5. **PCA** eficiente para redução de dimensionalidade

### Recomendações para Aprendizado Federado

1. **Detecção de Envenenamento**:
   - Usar Elliptic Envelope ou Isolation Forest
   - Aplicar em gradientes/pesos dos clientes
   - Validar contra comportamento histórico

2. **Agregação Robusta**:
   - Filtrar outliers antes da agregação
   - Usar múltiplas técnicas (ensemble)
   - Monitorar métricas continuamente

3. **Validação**:
   - Manter dataset de validação rotulado
   - Testar periodicamente com ataques simulados
   - Ajustar thresholds conforme necessário

## Limitações

- Dataset sintético pode não capturar todas as nuances de ameaças reais
- Performance depende da qualidade das features
- Hiperparâmetros precisam de ajuste para cada caso
- Métodos não supervisionados têm limitações inerentes

## Trabalhos Futuros

### Datasets Adicionais
- [ ] MNIST/Fashion-MNIST para visão computacional
- [ ] Datasets de rede (network intrusion detection)
- [ ] Dados reais de aprendizado federado

### Técnicas Avançadas
- [ ] Deep Learning (Neural Networks)
- [ ] Ensemble learning avançado (Stacking, Boosting)
- [ ] AutoML para otimização de hiperparâmetros
- [ ] Explainable AI (SHAP, LIME)

### Integração com FL
- [ ] Simulação completa de aprendizado federado
- [ ] Ataques por envenenamento em ambiente controlado
- [ ] Byzantine-robust aggregation (Krum, Median, Trimmed Mean)
- [ ] Detecção em tempo real durante training

### Análise e Documentação
- [ ] Benchmarking sistemático de todas as técnicas
- [ ] Análise de custo computacional
- [ ] Guia de seleção de técnicas por contexto
- [ ] Publicação de resultados em conferência/journal

## Tecnologias Utilizadas

### Linguagens e Frameworks
- **Python 3.8+**: Linguagem principal
- **Jupyter Notebook**: Análises interativas
- **NumPy**: Computação numérica
- **Pandas**: Manipulação de dados

### Machine Learning
- **Scikit-learn**: Algoritmos de ML
  - Classification: DecisionTree, RandomForest, SVM, NaiveBayes, KNN, LogisticRegression
  - Clustering: KMeans, AgglomerativeClustering
  - Outlier Detection: IsolationForest, LOF, OneClassSVM, EllipticEnvelope
  - Dimensionality Reduction: PCA
  - Metrics: accuracy, precision, recall, f1, silhouette, davies_bouldin

### Visualização
- **Matplotlib**: Gráficos base
- **Seaborn**: Visualizações estatísticas
- **Plotly**: Gráficos interativos (opcional)

### Dados
- **Scikit-learn Datasets**: Iris
- **Seaborn Data**: Penguins
- **Synthetic Data**: Weight-Height
- **Kaggle**: Cyber Threat Detection

## Estrutura de Código

### Padrão dos Scripts Python

```python
# 1. Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# ... outros imports

# 2. Carregamento de Dados
df = load_data()

# 3. Exploração Visual
generate_exploratory_plots()

# 4. Preparação
X_train, X_test, y_train, y_test = prepare_data()

# 5. Treinamento Supervisionado
for model in models:
    train_and_evaluate(model)

# 6. Clustering
kmeans_results = run_clustering()

# 7. PCA
pca_analysis()

# 8. Salvamento
save_results()
```

### Padrão dos Notebooks

1. **Markdown**: Introdução e contexto
2. **Code**: Imports
3. **Markdown**: Seção de carregamento
4. **Code**: Load data + exploração
5. **Markdown**: Seção supervisionada
6. **Code**: Treinamento + métricas + plots
7. **Markdown**: Seção não supervisionada
8. **Code**: Clustering + PCA
9. **Markdown**: Resumo e conclusões

## Referências

### Algoritmos e Técnicas
1. Liu, F. T., et al. (2008). **Isolation Forest**. ICDM.
2. Breunig, M. M., et al. (2000). **LOF: Identifying Density-based Local Outliers**. SIGMOD.
3. Schölkopf, B., et al. (2001). **Estimating the Support of a High-Dimensional Distribution**. Neural Computation.
4. Chandola, V., Banerjee, A., & Kumar, V. (2009). **Anomaly Detection: A Survey**. ACM Computing Surveys.
5. Cortes, C., & Vapnik, V. (1995). **Support-Vector Networks**. Machine Learning.
6. Breiman, L. (2001). **Random Forests**. Machine Learning.

### Aprendizado Federado
7. McMahan, B., et al. (2017). **Communication-Efficient Learning of Deep Networks from Decentralized Data**. AISTATS.
8. Blanchard, P., et al. (2017). **Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent**. NeurIPS.
9. Bagdasaryan, E., et al. (2020). **Backdoor Attacks Against Learning Systems**. IEEE S&P.
10. Fung, C., et al. (2018). **Mitigating Sybils in Federated Learning Poisoning**. arXiv.

### Datasets
11. Fisher, R. A. (1936). **The Use of Multiple Measurements in Taxonomic Problems**. Annals of Eugenics.
12. Gorman, K. B., et al. (2014). **Ecological Sexual Dimorphism in Penguins**. PLOS ONE.

---

**Projeto**: Mitigação de Ataques por Envenenamento em Aprendizado Federado  
**Instituição**: Faculdade Impacta  
**Área**: Ciência da Computação / Cibersegurança / Machine Learning  
**Data**: Outubro 2025

---

## Contato e Contribuições

Para dúvidas, sugestões ou contribuições:
- Consulte a documentação em `docs/Projeto.md`
- Veja o README principal do projeto
- Analise os notebooks interativos para exemplos práticos

**Status**: ✅ Projeto completo e funcional
