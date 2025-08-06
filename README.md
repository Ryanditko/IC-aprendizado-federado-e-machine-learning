# 🧠 Projeto de Machine Learning - Aprendizado Supervisionado e Não Supervisionado

Este projeto implementa algoritmos de aprendizado de máquina (supervisionado e não supervisionado) utilizando os datasets **Iris** e **Penguin**. O projeto demonstra análises completas, comparações entre algoritmos e visualizações detalhadas.

## 📋 Índice

- [Estrutura do Projeto](#estrutura-do-projeto)
- [Datasets Utilizados](#datasets-utilizados)
- [Algoritmos Implementados](#algoritmos-implementados)
- [Instalação e Configuração](#instalação-e-configuração)
- [Como Executar](#como-executar)
- [Descrição dos Arquivos](#descrição-dos-arquivos)
- [Resultados e Métricas](#resultados-e-métricas)
- [Requisitos](#requisitos)

## 🗂️ Estrutura do Projeto

```
doc/
├── code/
│   ├── data/
│   │   ├── datasets-install.py          # Script para instalação de datasets
│   │   └── verificar-caminho.py         # Utilitário para verificação de caminhos
│   ├── iris-dataset/
│   │   ├── aprendizado-supervisionado.py      # ML Supervisionado - Iris
│   │   ├── aprendizado-nao-supervisionado.py  # ML Não Supervisionado - Iris
│   │   └── iris.csv                           # Dataset Iris (gerado automaticamente)
│   └── penguin-dataset/
│       ├── aprendizado-supervisionado.py      # ML Supervisionado - Penguin
│       ├── aprendizado-nao-supervisionado.py  # ML Não Supervisionado - Penguin
│       └── penguins.csv                       # Dataset Penguin (gerado automaticamente)
└── documentação-do-projeto/
    ├── Projeto.md
    ├── aprendizado-supervisionado/
    ├── aprendizado-não-supervisionado/
    ├── avaliações-de-modelos/
    └── desafios/
```

## 📊 Datasets Utilizados

### 1. Iris Dataset
- **Descrição**: Conjunto clássico de dados de flores Iris
- **Features**: 4 características (comprimento e largura de sépala e pétala)
- **Classes**: 3 espécies (Setosa, Versicolor, Virginica)
- **Amostras**: 150 observações

### 2. Penguin Dataset
- **Descrição**: Dados de pinguins Palmer
- **Features**: Características físicas e localização
- **Classes**: 3 espécies (Adelie, Chinstrap, Gentoo)
- **Amostras**: ~344 observações

## 🤖 Algoritmos Implementados

### Aprendizado Supervisionado
- **Decision Tree** (Árvore de Decisão)
- **Random Forest** (Floresta Aleatória)
- **SVM** (Support Vector Machine)
- **Naive Bayes**
- **K-Nearest Neighbors (KNN)**
- **Logistic Regression** (Regressão Logística)

### Aprendizado Não Supervisionado
- **K-Means Clustering**
- **Hierarchical Clustering** (Agglomerative)
- **DBSCAN** (Density-Based Clustering)
- **PCA** (Principal Component Analysis)

## 🛠️ Instalação e Configuração

### 1. Requisitos do Sistema
- Python 3.7 ou superior
- pip (gerenciador de pacotes)

### 2. Instalar Dependências

Execute o comando no terminal:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

### 3. Estrutura de Pastas
O projeto já possui a estrutura correta. Certifique-se de que você está na pasta raiz do projeto.

## ▶️ Como Executar

### Opção 1: Execução Individual

#### Aprendizado Supervisionado - Iris
```bash
cd "doc/code/iris-dataset"
python aprendizado-supervisionado.py
```

#### Aprendizado Não Supervisionado - Iris
```bash
cd "doc/code/iris-dataset"
python aprendizado-nao-supervisionado.py
```

#### Aprendizado Supervisionado - Penguin
```bash
cd "doc/code/penguin-dataset"
python aprendizado-supervisionado.py
```

#### Aprendizado Não Supervisionado - Penguin
```bash
cd "doc/code/penguin-dataset"
python aprendizado-nao-supervisionado.py
```

### Opção 2: Executar Tudo
Você pode executar todos os scripts em sequência visitando cada pasta e executando os arquivos.

## 📁 Descrição dos Arquivos

### Aprendizado Supervisionado
- **Carregamento automático de datasets** (local, seaborn, sklearn, URL)
- **Análise exploratória completa** (estatísticas, visualizações)
- **Pré-processamento** (limpeza, codificação, normalização)
- **Treinamento de 6 algoritmos diferentes**
- **Avaliação com múltiplas métricas** (accuracy, precision, recall, F1-score)
- **Visualizações comparativas** (confusion matrix, ROC curves, etc.)
- **Validação cruzada**
- **Hyperparameter tuning**

### Aprendizado Não Supervisionado
- **Análise de Componentes Principais (PCA)**
- **Clustering com K-Means** (método do cotovelo, silhouette analysis)
- **Clustering Hierárquico** (dendrograma, diferentes linkages)
- **DBSCAN** (detecção de outliers, otimização de parâmetros)
- **Comparação de algoritmos** (métricas de qualidade)
- **Visualizações 2D e 3D**
- **Análise de características dos clusters**

### Utilitários
- **datasets-install.py**: Script para instalação e verificação de datasets
- **verificar-caminho.py**: Template para configuração de caminhos

## 📈 Resultados e Métricas

### Métricas de Aprendizado Supervisionado
- **Accuracy**: Proporção de predições corretas
- **Precision**: Proporção de verdadeiros positivos
- **Recall**: Capacidade de encontrar todos os positivos
- **F1-Score**: Média harmônica entre precision e recall
- **ROC-AUC**: Área sob a curva ROC

### Métricas de Aprendizado Não Supervisionado
- **Silhouette Score**: Qualidade dos clusters (-1 a 1)
- **Adjusted Rand Index (ARI)**: Similaridade com classes verdadeiras
- **Normalized Mutual Information (NMI)**: Informação mútua normalizada
- **Inértia**: Soma das distâncias quadráticas aos centroides

## 📦 Requisitos

### Bibliotecas Python
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
scipy>=1.7.0
```

### Instalação Automática
Execute em qualquer terminal:
```bash
pip install -r requirements.txt
```

## 🚀 Funcionalidades Especiais

### 1. Carregamento Inteligente de Datasets
- Busca local primeiro
- Fallback para seaborn/sklearn
- Download automático via URL
- Salvamento local para reutilização

### 2. Caminhos Relativos Inteligentes
- Uso de `os.path.join()` para compatibilidade
- Caminhos baseados na localização do arquivo
- Funcionamento em qualquer sistema operacional

### 3. Análises Robustas
- Tratamento de valores faltantes
- Codificação automática de variáveis categóricas
- Normalização/padronização automática
- Validação cruzada

### 4. Visualizações Profissionais
- Gráficos de alta qualidade
- Comparações lado a lado
- Métricas em tabelas formatadas
- Cores e estilos consistentes

## 🛡️ Tratamento de Erros

- **Datasets não encontrados**: Download automático
- **Bibliotecas faltantes**: Mensagens de erro claras
- **Dados corrompidos**: Validação e limpeza automática
- **Problemas de caminho**: Resolução automática

## 💡 Dicas de Uso

1. **Primeira execução**: Pode demorar mais devido ao download dos datasets
2. **Execuções subsequentes**: Serão mais rápidas usando dados locais
3. **Personalização**: Modifique os parâmetros nos scripts para experimentar
4. **Visualizações**: Use `plt.show()` em ambientes que suportam
5. **Reprodutibilidade**: Seeds fixos garantem resultados consistentes

## 📚 Documentação Adicional

Consulte a pasta `documentação-do-projeto/` para:
- Explicações teóricas detalhadas
- Exemplos de uso avançado
- Desafios propostos
- Métricas de avaliação

## 🎯 Próximos Passos

1. Execute os scripts na ordem sugerida
2. Analise os resultados e gráficos gerados
3. Compare o desempenho entre algoritmos
4. Experimente com diferentes parâmetros
5. Consulte a documentação para aprofundamento teórico

---

**Autor**: Projeto de Iniciação Científica
**Data**: 2025
**Versão**: 1.0

Para dúvidas ou problemas, consulte a documentação ou verifique se todas as dependências estão instaladas corretamente.
