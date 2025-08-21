# 🎯 Projeto de Detecção de Outliers

## 📋 Visão Geral

Este projeto implementa e avalia múltiplas técnicas de detecção de outliers usando aprendizado não supervisionado. O objetivo é comparar diferentes algoritmos e determinar qual apresenta melhor desempenho para detecção de anomalias em dados de altura e peso.

## 🗂️ Estrutura do Projeto

```
outlier-detection/
├── main_analysis.py          # Script principal de análise
├── requirements.txt          # Dependências do projeto
├── data/
│   └── download_data.py      # Script para criação/download de dados
├── src/
│   ├── outlier_detector.py  # Implementação dos algoritmos de detecção
│   └── model_evaluator.py   # Avaliação e métricas dos modelos
├── results/                  # Resultados gerados (CSVs e gráficos)
└── notebooks/               # Jupyter notebooks (opcional)
```

## 🔧 Instalação e Configuração

### Pré-requisitos
- Python 3.8+
- pip (gerenciador de pacotes do Python)

### Instalação das Dependências

```bash
cd outlier-detection
pip install -r requirements.txt
```

### Dependências Principais
- `pandas` - Manipulação de dados
- `numpy` - Computação numérica
- `scikit-learn` - Algoritmos de machine learning
- `matplotlib` - Visualização de dados
- `seaborn` - Visualização estatística avançada
- `scipy` - Computação científica

## 🚀 Como Executar

### Execução Completa
```bash
python main_analysis.py
```

### Execução por Etapas

1. **Preparar dados:**
```bash
python data/download_data.py
```

2. **Executar análise específica:**
```python
from src.outlier_detector import OutlierDetector
detector = OutlierDetector(data)
results = detector.run_all_methods()
```

## 📊 Dados Utilizados

### Dataset: Altura e Peso
- **Registros**: 1.050 amostras
- **Variáveis**: 
  - `Height_cm`: Altura em centímetros
  - `Weight_kg`: Peso em quilogramas
  - `Is_Outlier`: Label verdadeiro (0=normal, 1=outlier)
- **Outliers**: 50 registros (4,76% dos dados)

### Características dos Dados
- Dados sintéticos gerados para demonstração
- Correlação natural entre altura e peso
- Outliers artificiais inseridos estrategicamente

## 🔍 Métodos de Detecção Implementados

### 1. Z-Score
- **Princípio**: Detecta pontos que estão a mais de N desvios padrão da média
- **Parâmetros**: `threshold=3`
- **Características**: Simples e eficaz para distribuições normais

### 2. IQR (Interquartile Range)
- **Princípio**: Identifica valores fora do intervalo [Q1-1.5*IQR, Q3+1.5*IQR]
- **Parâmetros**: `multiplier=1.5`
- **Características**: Robusto a outliers na própria detecção

### 3. Isolation Forest
- **Princípio**: Isola outliers usando árvores de decisão aleatórias
- **Parâmetros**: `contamination=0.1`
- **Características**: Eficiente para grandes datasets

### 4. Local Outlier Factor (LOF)
- **Princípio**: Compara densidade local com vizinhança
- **Parâmetros**: `n_neighbors=20, contamination=0.1`
- **Características**: Detecta outliers locais

### 5. DBSCAN
- **Princípio**: Clustering baseado em densidade
- **Parâmetros**: `eps=0.5, min_samples=5`
- **Características**: Identifica ruído como outliers

### 6. One-Class SVM
- **Princípio**: Cria fronteira de decisão para dados normais
- **Parâmetros**: `nu=0.1, gamma='scale'`
- **Características**: Flexível para distribuições complexas

### 7. Elliptic Envelope
- **Princípio**: Assume distribuição gaussiana multivariada
- **Parâmetros**: `contamination=0.1`
- **Características**: Bom para dados com distribuição elíptica

## 📈 Resultados Obtidos

### Desempenho dos Métodos

| Método | Acurácia | Precisão | Recall | F1-Score | Outliers Detectados |
|--------|----------|----------|--------|----------|-------------------|
| **Z-Score** | **99,8%** | **98,0%** | **98,0%** | **98,0%** | 50 |
| **IQR** | **99,2%** | **86,2%** | **100%** | **92,6%** | 58 |
| Isolation Forest | 94,8% | 47,6% | 100% | 64,5% | 105 |
| LOF | 85,2% | 0,0% | 0,0% | 0,0% | 105 |
| DBSCAN | 95,1% | 0,0% | 0,0% | 0,0% | 1 |
| One-Class SVM | 93,2% | 40,0% | 84,0% | 54,2% | 105 |
| Elliptic Envelope | 94,8% | 47,6% | 100% | 64,5% | 105 |

### 🏆 Melhores Métodos
1. **Z-Score**: Melhor método geral (alta precisão e recall)
2. **IQR**: Excelente recall com boa precisão
3. **Ensemble**: Combinação de métodos com F1-score de 81%

### 📊 Análise de Concordância
- **Maior concordância**: Z-Score e IQR (99,2%)
- **Menor concordância**: LOF e outros métodos (~85%)
- **Ensemble**: Voto majoritário de todos os métodos

## 📁 Arquivos de Saída

### CSVs Gerados
- `method_comparison.csv` - Comparação quantitativa dos métodos
- `evaluation_metrics.csv` - Métricas detalhadas de avaliação
- `method_agreement.csv` - Matriz de concordância entre métodos
- `detailed_predictions.csv` - Predições detalhadas por método

### Gráficos Gerados
- `exploratory_analysis.png` - Análise exploratória dos dados
- `outliers_[método].png` - Visualização de outliers por método
- `confusion_matrices.png` - Matrizes de confusão
- `method_agreement.png` - Heatmap de concordância
- `ensemble_analysis.png` - Análise do ensemble

## 🔬 Análise dos Resultados

### Principais Descobertas

1. **Métodos Estatísticos Simples São Eficazes**
   - Z-Score e IQR apresentaram os melhores resultados
   - Simplicidade nem sempre significa menor eficácia

2. **Problemas com Densidade**
   - LOF teve desempenho muito ruim neste dataset
   - DBSCAN foi excessivamente conservador

3. **Ensemble Melhora Recall**
   - Combinação de métodos aumentou sensibilidade
   - Trade-off entre precisão e recall

4. **Importância da Calibração**
   - Parâmetros de contaminação influenciam muito os resultados
   - Métodos com `contamination=0.1` detectaram muitos falsos positivos

### Recomendações

- **Para dados similares**: Use Z-Score ou IQR
- **Para exploração**: Combine múltiplos métodos
- **Para produção**: Valide com dados rotulados

## 🚧 Limitações e Trabalhos Futuros

### Limitações Atuais
- Dataset sintético pode não representar casos reais
- Parâmetros não foram otimizados sistematicamente
- Avaliação limitada a dados bivariados

### Melhorias Propostas
- [ ] Otimização automática de hiperparâmetros
- [ ] Teste com datasets reais diversos
- [ ] Implementação de métodos ensemble mais sofisticados
- [ ] Análise de explicabilidade dos outliers detectados
- [ ] Interface web para demonstração interativa

## 📚 Referências

1. Aggarwal, C. C. (2017). *Outlier Analysis*. Springer.
2. Liu, F. T., et al. (2008). Isolation Forest. ICDM.
3. Breunig, M. M., et al. (2000). LOF: Identifying Density-based Local Outliers. SIGMOD.
4. Scikit-learn Documentation: [Outlier Detection](https://scikit-learn.org/stable/modules/outlier_detection.html)

## 🤝 Contribuições

Este projeto faz parte da pesquisa de Iniciação Científica sobre "Mitigação de Ataques por Envenenamento em Aprendizado Federado - Avaliação de Abordagens Baseadas em Outliers".

Para contribuições ou dúvidas, consulte a documentação completa do projeto principal.

---

*Última atualização: 21 de agosto de 2025*
