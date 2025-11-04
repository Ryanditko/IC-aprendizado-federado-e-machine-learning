# NSL-KDD Dataset - Detecção de Ataques de Cybersegurança

## 📊 Visão Geral

Este módulo implementa algoritmos de machine learning para detecção de ataques de cybersegurança utilizando o dataset **NSL-KDD** (Network Security Laboratory - Knowledge Discovery and Data Mining).

### 🎯 Objetivo

Avaliar a eficácia de diferentes algoritmos na detecção de ataques específicos, com foco em:
- **User-to-Root (U2R)** attacks (incluindo SQL injection-like attacks)
- Comparação de múltiplos modelos de ML
- Análise detalhada de métricas de performance

## 📁 Estrutura do Projeto

```
nsl-kdd/
├── data/                           # Dados do dataset NSL-KDD
│   ├── KDDTrain+.txt              # Dados de treino
│   ├── KDDTest+.txt               # Dados de teste
│   └── ...
├── scripts-datasets/
│   └── nsl-kdd/
│       └── deteccao-ataques-nsl-kdd.py  # Script principal
├── scripts-notebooks/
│   └── run_nsl_kdd.py             # Executor do notebook
├── notebooks/
│   └── nsl-kdd/
│       ├── nsl_kdd_evaluation.ipynb     # Notebook interativo
│       ├── output-images/               # Gráficos gerados
│       └── results/                     # Resultados das análises
└── downloads/
    └── download_nsl_kdd_dataset.py      # Download automático
```

## 🚀 Como Usar

### 1. Download do Dataset

```bash
cd downloads
python download_nsl_kdd_dataset.py
```

**Pré-requisitos:**
- Conta no Kaggle
- API Token configurado (`~/.kaggle/kaggle.json`)
- Biblioteca `kagglehub` instalada

### 2. Executar Análise

**Opção A - Script Direto:**
```bash
cd code/scripts-datasets/nsl-kdd
python deteccao-ataques-nsl-kdd.py
```

**Opção B - Via Notebook Executor:**
```bash
cd code/scripts-notebooks
python run_nsl_kdd.py
```

**Opção C - Jupyter Notebook:**
```bash
cd notebooks/nsl-kdd
jupyter notebook nsl_kdd_evaluation.ipynb
```

## 📊 Dataset NSL-KDD

### Características
- **Total de features:** 41 + target
- **Tipos de ataque:** Normal, DoS, Probe, R2L, U2R
- **Formato:** CSV com features numéricas e categóricas

### Tipos de Ataque Analisados

| Categoria | Descrição | Exemplos |
|-----------|-----------|----------|
| **Normal** | Tráfego legítimo | - |
| **DoS** | Denial of Service | neptune, smurf, pod |
| **Probe** | Reconnaissance | portsweep, nmap, satan |
| **R2L** | Remote to Local | warezclient, guess_passwd |
| **U2R** | User to Root | buffer_overflow, rootkit, **sqlattack** |

### Foco do Estudo: U2R Attacks

Os ataques **User-to-Root (U2R)** são o foco principal, incluindo:
- `buffer_overflow`: Exploração de buffer overflow
- `rootkit`: Instalação de rootkits
- `sqlattack`: **Ataques tipo SQL injection**
- `loadmodule`: Carregamento malicioso de módulos

## 🤖 Modelos Implementados

### 1. Random Forest
- **Vantagens:** Boa interpretabilidade, feature importance
- **Uso:** Baseline robusto para classificação

### 2. Logistic Regression
- **Vantagens:** Rápido, interpretável
- **Uso:** Modelo linear para comparação

### 3. Support Vector Machine (SVM)
- **Vantagens:** Eficaz em alta dimensionalidade
- **Uso:** Modelo não-linear sofisticado

## 📈 Métricas Avaliadas

### Métricas Principais
- **Accuracy:** Porcentagem de predições corretas
- **Precision:** TP / (TP + FP) - Reduz falsos alarmes
- **Recall:** TP / (TP + FN) - Detecta ataques reais
- **F1-Score:** Harmônica de precision e recall

### Visualizações Geradas
1. **Matriz de Confusão** - Distribuição de acertos/erros
2. **Comparação de Métricas** - Gráfico de barras comparativo
3. **Curvas ROC** - Capacidade discriminativa
4. **Distribuição de Ataques** - Pie chart dos tipos
5. **Feature Importance** - Variáveis mais relevantes

## 📊 Exemplo de Resultados

```
🏆 MELHOR MODELO: Random Forest
================================
F1-Score: 0.856
Accuracy: 0.934
Precision: 0.798
Recall: 0.924

🔍 MATRIZ DE CONFUSÃO:
True Positives:    42 - Ataques detectados
True Negatives:   534 - Tráfego normal
False Positives:   11 - Falsos alarmes
False Negatives:    3 - Ataques perdidos
```

## 🎯 Interpretação dos Resultados

### Para Cybersegurança:

**High Precision (>80%):**
- Poucos falsos alarmes
- Reduz fadiga de alertas
- Foco em ameaças reais

**High Recall (>90%):**
- Detecta maioria dos ataques
- Reduz riscos de segurança
- Cobertura abrangente

**F1-Score Balanceado:**
- Equilíbrio optimal
- Adequado para produção

## 📁 Arquivos Gerados

### Gráficos
- `nsl_kdd_attack_detection_analysis.png` - Dashboard principal
- `attack_distribution.png` - Distribuição dos ataques

### Resultados
- `attack_detection_results.txt` - Relatório detalhado
- `model_comparison.csv` - Tabela comparativa

## 🔧 Dependências

```python
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
kagglehub>=0.1.0
```

## 🚨 Considerações de Produção

### Vantagens do NSL-KDD:
- Dataset benchmark reconhecido
- Balanceamento melhorado vs KDD Cup 99
- Remoção de registros redundantes

### Limitações:
- Dataset de 1999 (pode não refletir ataques modernos)
- Características de rede podem estar desatualizadas
- Necessário validação com dados recentes

### Recomendações:
1. **Retreinamento periódico** com dados atuais
2. **Validação cruzada** com outros datasets
3. **Monitoramento contínuo** de performance
4. **Ajuste de thresholds** para produção

## 📚 Referências

- **NSL-KDD Dataset:** [University of New Brunswick](https://www.unb.ca/cic/datasets/nsl.html)
- **Paper Original:** Tavallaee, M., et al. "A detailed analysis of the KDD CUP 99 data set"
- **Kaggle Dataset:** [hassan06/nslkdd](https://www.kaggle.com/hassan06/nslkdd)

## 🤝 Contribuições

Para contribuir com melhorias:
1. Implementar novos algoritmos
2. Adicionar outras categorias de ataque
3. Otimizar performance dos modelos
4. Incluir técnicas de ensemble

---

**Autor:** Projeto de Iniciação Científica  
**Data:** Novembro 2025  
**Versão:** 1.0
