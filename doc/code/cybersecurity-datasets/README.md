# Análise do Dataset Cybersecurity - Kaggle

Este projeto implementa uma análise completa do dataset "Text-based Cyber Threat Detection" da Kaggle, realizando tanto análise supervisionada quanto não supervisionada dos dados.

## 🎯 Objetivo

Realizar uma análise abrangente de dados de cybersecurity, incluindo:
- **Análise Exploratória**: Compreensão da estrutura e características dos dados
- **Aprendizado Supervisionado**: Classificação de ameaças usando Random Forest
- **Aprendizado Não Supervisionado**: Clustering e análise de padrões
- **Detecção de Anomalias**: Identificação de comportamentos anômalos

## 📋 Pré-requisitos

1. **Python 3.8+**
2. **Conta Kaggle** com API configurada
3. **Bibliotecas Python** (listadas em requirements.txt)

## 🚀 Configuração Rápida

### 1. Instalar Dependências
```bash
# Opção 1: Usar o script automatizado
python setup_cybersecurity_env.py

# Opção 2: Instalar manualmente
pip install -r requirements.txt
pip install kagglehub
```

### 2. Configurar Credenciais Kaggle

1. Acesse https://www.kaggle.com/settings/account
2. Role até a seção "API" e clique em "Create New Token"
3. Baixe o arquivo `kaggle.json`
4. **Windows**: Mova para `C:\Users\{seu_usuario}\.kaggle\kaggle.json`
5. **Linux/Mac**: Mova para `~/.kaggle/kaggle.json`

### 3. Testar Ambiente
```bash
python test_cybersecurity_env.py
```

### 4. Executar Análise Completa
```bash
python cybersecurity_analysis.py
```

## 📁 Estrutura dos Arquivos

```
cybersecurity-analysis/
├── cybersecurity_analysis.py      # Script principal de análise
├── setup_cybersecurity_env.py     # Configuração do ambiente
├── test_cybersecurity_env.py      # Teste do ambiente
├── avaliador_nao_supervisionado.py # Classe para análise não supervisionada
└── cybersecurity-datasets/        # Resultados da análise
    ├── cybersecurity_analysis_report.md
    └── cybersecurity_results.csv
```

## 🔍 O que o Script Faz

### 1. **Download Automático**
- Baixa o dataset "Text-based Cyber Threat Detection" da Kaggle
- Verifica integridade dos arquivos

### 2. **Análise Exploratória**
- Informações básicas do dataset (dimensões, tipos de dados)
- Identificação de valores faltantes
- Análise das variáveis categóricas e numéricas

### 3. **Pré-processamento**
- Codificação de variáveis categóricas
- Normalização de dados numéricos
- Separação de features e variável alvo

### 4. **Análise Supervisionada**
- Treinamento de modelo Random Forest
- Avaliação de performance (acurácia, matriz de confusão)
- Análise de importância das features

### 5. **Análise Não Supervisionada**
- Clustering K-means com diferentes valores de K
- Avaliação usando Silhouette Score
- Análise PCA para redução de dimensionalidade

### 6. **Detecção de Anomalias**
- Isolation Forest
- Local Outlier Factor (LOF)
- Análise de consenso entre métodos

### 7. **Geração de Relatórios**
- Relatório detalhado em Markdown
- Resultados formatados para planilhas
- Visualizações e estatísticas

## 📊 Resultados Esperados

Após a execução, você terá:

1. **Relatório Markdown** (`cybersecurity_analysis_report.md`):
   - Análise completa com estatísticas
   - Interpretação dos resultados
   - Recomendações

2. **Arquivo CSV** (`cybersecurity_results.csv`):
   - Resultados formatados para planilhas
   - Métricas de avaliação
   - Comparação entre técnicas

3. **Insights sobre**:
   - Qualidade dos dados de cybersecurity
   - Padrões de ameaças identificados
   - Eficácia de diferentes algoritmos
   - Anomalias nos dados

## 🛠️ Personalização

### Modificar Algoritmos
```python
# No arquivo cybersecurity_analysis.py, você pode alterar:

# Classificador supervisionado
rf_model = RandomForestClassifier(
    n_estimators=200,  # Aumentar árvores
    max_depth=10,      # Limitar profundidade
    random_state=42
)

# Parâmetros de clustering
k_range = range(2, 15)  # Testar mais clusters

# Detecção de anomalias
iso_forest = IsolationForest(
    contamination=0.05,  # Menos anomalias
    n_estimators=200
)
```

### Adicionar Novas Métricas
```python
# Adicionar no método supervised_analysis():
from sklearn.metrics import f1_score, precision_score, recall_score

f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
```

## ⚠️ Solução de Problemas

### Erro de Credenciais Kaggle
```
kaggle.api.kaggle_api_extended.KaggleApi.authenticate()
OSError: Could not find kaggle.json
```

**Solução**: Configure as credenciais conforme instruções acima.

### Erro de Importação kagglehub
```
ImportError: No module named 'kagglehub'
```

**Solução**: 
```bash
pip install kagglehub --upgrade
```

### Dataset não Encontrado
```
403 - Forbidden
```

**Solução**: 
1. Verifique se sua conta Kaggle pode acessar o dataset
2. Aceite os termos do dataset no site da Kaggle
3. Verifique se o nome do dataset está correto

### Problemas de Memória
Para datasets grandes:
```python
# Reduzir amostra para teste
sample_size = 10000
data_sample = data.sample(n=sample_size, random_state=42)
```

## 📈 Extensões Futuras

1. **Análise de Texto**: Processar colunas de texto com NLP
2. **Deep Learning**: Implementar redes neurais
3. **Visualizações**: Adicionar gráficos interativos
4. **API REST**: Criar endpoint para análise online
5. **Dashboard**: Interface web com Streamlit/Dash

## 🤝 Contribuição

Para contribuir:
1. Fork o repositório
2. Crie uma branch para sua feature
3. Faça commit das mudanças
4. Abra um Pull Request

## 📄 Licença

Este projeto é para fins educacionais e de pesquisa.

---

**Desenvolvido para**: Iniciação Científica - Faculdade Impacta  
**Foco**: Mitigação de Ataques por Envenenamento em Aprendizado Federado
