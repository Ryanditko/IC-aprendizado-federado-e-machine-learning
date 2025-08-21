# Machine Learning Aplicado a Classificação: Análise Comparativa

## 🎯 Descrição do Projeto

Este projeto faz parte da pesquisa de Iniciação Científica sobre **"Mitigação de Ataques por Envenenamento em Aprendizado Federado"** e tem como objetivo inicial aplicar diferentes técnicas de classificação em machine learning para análise comparativa de desempenho.

## 🎯 Objetivos

- Implementar 6 algoritmos de classificação diferentes
- Comparar desempenho usando 4 métricas principais  
- Alcançar meta de 70% de acurácia em todas as métricas
- Estabelecer base para estudos em aprendizado federado

## 📊 Datasets Utilizados

### 1. Iris Dataset
- **Features**: 4 numéricas (sepal_length, sepal_width, petal_length, petal_width)
- **Classes**: 3 espécies (Setosa, Versicolor, Virginica)
- **Tamanho**: 150 amostras

### 2. Penguin Dataset
- **Features**: Numéricas e categóricas (bill_length_mm, bill_depth_mm, sex, island)
- **Classes**: 3 espécies (Adelie, Chinstrap, Gentoo)
- **Tamanho**: Variável após limpeza

## 🤖 Algoritmos Implementados

1. **Decision Tree** - Árvore de Decisão
2. **Random Forest** - Floresta Aleatória
3. **Support Vector Machine (SVM)**
4. **Naive Bayes** - Classificador Bayesiano
5. **k-Nearest Neighbors (k-NN)**
6. **Logistic Regression** - Regressão Logística

## 📈 Métricas de Avaliação

- **Accuracy**: Proporção de predições corretas
- **Precision**: Capacidade de não rotular incorretamente
- **Recall**: Capacidade de encontrar todos os exemplos
- **F1-Score**: Média harmônica entre precision e recall

## 🚀 Como Executar

### Pré-requisitos
```bash
pip install pandas scikit-learn matplotlib seaborn numpy
```

### Execução
```bash
# Dataset Iris
cd doc/code/iris-dataset/
python modelagem.py

# Dataset Penguin
cd doc/code/penguin-dataset/
python modelagem.py
```

## 📁 Estrutura do Projeto

```
doc/
├── code/
│   ├── iris-dataset/
│   │   └── modelagem.py
│   └── penguin-dataset/
│       └── modelagem.py
└── documentação-do-projeto/
    ├── README.md
    ├── apresentacao-modelo.md
    └── desafios/
```

## 🔧 Metodologia

### Pré-processamento
- Remoção de valores faltantes
- Codificação de variáveis categóricas
- Normalização com StandardScaler
- Divisão treino/teste: 70%/30%

### Validação
- Random state fixo para reprodutibilidade
- Stratified split para balanceamento
- Métricas com average='weighted'

## 📊 Resultados Esperados

Os experimentos devem demonstrar:
- Performance superior a 70% em todas as métricas
- Comparação entre algoritmos
- Identificação do melhor modelo por dataset

## 🔗 Conexão com Projeto Principal

Esta etapa prepara para:
- Implementação de aprendizado federado
- Simulação de ataques por envenenamento
- Desenvolvimento de estratégias de mitigação
- Detecção de outliers e clientes maliciosos

## 📋 Status do Projeto

- ✅ **Concluído**: Implementação dos algoritmos
- 🔄 **Em andamento**: Análise de resultados
- ⏳ **Próximo**: Design de ambiente federado
- ⏳ **Futuro**: Simulação de ataques

## 👨‍💻 Autor - Ryanditko

Desenvolvido como parte da Iniciação Científica em Machine Learning e Aprendizado Federado.

## 📚 Referências

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Iris Dataset - Kaggle](https://www.kaggle.com/code/ash316/ml-from-scratch-with-iris)
- [Penguin Dataset - Kaggle](https://www.kaggle.com/code/parulpandey/penguin-dataset-the-new-iris)