# 🚀 GUIA RÁPIDO - PROJETO MACHINE LEARNING

## ⚡ Execução Rápida

### 1. Instalar Dependências
```bash
python install_dependencies.py
```

### 2. Executar Todas as Análises
```bash
python run_all_analyses.py
```

### 3. Executar Análises Individuais

#### Iris - Supervisionado
```bash
cd "doc/code/iris-dataset"
python aprendizado-supervisionado.py
```

#### Iris - Não Supervisionado
```bash
cd "doc/code/iris-dataset"
python aprendizado-nao-supervisionado.py
```

#### Penguin - Supervisionado
```bash
cd "doc/code/penguin-dataset"
python aprendizado-supervisionado.py
```

#### Penguin - Não Supervisionado
```bash
cd "doc/code/penguin-dataset"
python aprendizado-nao-supervisionado.py
```

## 📋 O que cada script faz

### Aprendizado Supervisionado
- Carrega e limpa os dados
- Treina 6 algoritmos diferentes
- Compara performance com métricas
- Gera gráficos de comparação
- Faz validação cruzada

### Aprendizado Não Supervisionado
- Análise de componentes principais (PCA)
- Clustering K-Means
- Clustering Hierárquico
- DBSCAN (detecção de outliers)
- Comparação entre algoritmos
- Visualizações 2D e 3D

## 🎯 Resultados Esperados

Cada script irá:
1. **Baixar dados automaticamente** (primeira execução)
2. **Mostrar estatísticas** dos datasets
3. **Exibir gráficos** de análise
4. **Comparar algoritmos** com métricas
5. **Salvar dados localmente** para próximas execuções

## 🔧 Resolução de Problemas

### Erro de Dependência
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

### Erro de Caminho
- Certifique-se de estar na pasta correta
- Use caminhos absolutos se necessário

### Datasets não carregam
- Verifique conexão com internet (primeira execução)
- Os dados serão salvos localmente após o primeiro download

## 📊 Métricas Importantes

### Supervisionado
- **Accuracy**: % de acertos
- **F1-Score**: Balanço precision/recall
- **ROC-AUC**: Qualidade da classificação

### Não Supervisionado
- **Silhouette**: Qualidade dos clusters
- **ARI**: Similaridade com classes reais
- **Inertia**: Compactação dos clusters

## ⏱️ Tempo de Execução

- **Primeira vez**: 5-10 minutos (download de dados)
- **Execuções seguintes**: 2-5 minutos
- **Por script**: 1-2 minutos cada

## 📁 Arquivos Gerados

Após a execução, você terá:
- `iris.csv` (dataset local)
- `penguins.csv` (dataset local)
- Gráficos exibidos na tela
- Resultados impressos no terminal

---

**Dica**: Execute `python run_all_analyses.py` para ver tudo de uma vez!
