# 📚 Índice da Documentação - Detecção de Outliers

## 🗂️ Estrutura da Documentação

### 📋 Documento Principal
- **[README.md](./README.md)** - Visão geral completa do projeto

### 🔍 Documentação Técnica Detalhada
- **[técnicas-detalhadas.md](./técnicas-detalhadas.md)** - Análise aprofundada de cada técnica
- **[metodologia-avaliação.md](./metodologia-avaliação.md)** - Metodologia de avaliação e métricas
- **[aplicação-aprendizado-federado.md](./aplicação-aprendizado-federado.md)** - Aplicação em FL

## 🎯 Navegação Rápida

### Por Interesse

#### 🔬 **Pesquisadores/Acadêmicos**
1. [README.md](./README.md) - Visão geral
2. [técnicas-detalhadas.md](./técnicas-detalhadas.md) - Fundamentos teóricos
3. [metodologia-avaliação.md](./metodologia-avaliação.md) - Metodologia científica
4. [aplicação-aprendizado-federado.md](./aplicação-aprendizado-federado.md) - Aplicações

#### 💻 **Desenvolvedores**
1. [README.md](./README.md) - Como executar
2. [técnicas-detalhadas.md](./técnicas-detalhadas.md) - Implementações
3. [aplicação-aprendizado-federado.md](./aplicação-aprendizado-federado.md) - Código prático

#### 📊 **Analistas de Dados**
1. [README.md](./README.md) - Resultados principais
2. [metodologia-avaliação.md](./metodologia-avaliação.md) - Métricas e análises
3. [técnicas-detalhadas.md](./técnicas-detalhadas.md) - Comparação de métodos

### Por Técnica

#### Z-Score
- [técnicas-detalhadas.md#1-z-score](./técnicas-detalhadas.md#1-z-score-pontuação-padrão)
- [aplicação-aprendizado-federado.md#z-score-recomendado](./aplicação-aprendizado-federado.md#1-z-score-recomendado)

#### IQR
- [técnicas-detalhadas.md#2-iqr](./técnicas-detalhadas.md#2-iqr-interquartile-range)
- [aplicação-aprendizado-federado.md#iqr-como-backup](./aplicação-aprendizado-federado.md#2-iqr-como-backup)

#### Isolation Forest
- [técnicas-detalhadas.md#3-isolation-forest](./técnicas-detalhadas.md#3-isolation-forest)

#### Local Outlier Factor (LOF)
- [técnicas-detalhadas.md#4-lof](./técnicas-detalhadas.md#4-local-outlier-factor-lof)

#### DBSCAN
- [técnicas-detalhadas.md#5-dbscan](./técnicas-detalhadas.md#5-dbscan)

#### One-Class SVM
- [técnicas-detalhadas.md#6-one-class-svm](./técnicas-detalhadas.md#6-one-class-svm)

#### Elliptic Envelope
- [técnicas-detalhadas.md#7-elliptic-envelope](./técnicas-detalhadas.md#7-elliptic-envelope)

### Por Métrica

#### Acurácia
- [metodologia-avaliação.md#acurácia](./metodologia-avaliação.md#acurácia)

#### Precisão
- [metodologia-avaliação.md#precisão](./metodologia-avaliação.md#precisão)

#### Recall
- [metodologia-avaliação.md#recall-sensibilidade](./metodologia-avaliação.md#recall-sensibilidade)

#### F1-Score
- [metodologia-avaliação.md#f1-score](./metodologia-avaliação.md#f1-score)

## 📈 Principais Resultados

### 🏆 Melhores Métodos
1. **Z-Score**: 99,8% acurácia, 98% precisão/recall
2. **IQR**: 99,2% acurácia, 100% recall
3. **Ensemble**: 97,8% acurácia, 100% recall

### 💡 Recomendações
- **FL Production**: Z-Score (alta precisão, baixo overhead)
- **Exploração**: IQR (máximo recall)
- **Robustez**: Ensemble (combinação de métodos)

## 🔗 Links Externos

### Código Fonte
- **Projeto Principal**: `/outlier-detection/main_analysis.py`
- **Implementações**: `/outlier-detection/src/`
- **Resultados**: `/outlier-detection/results/`

### Documentação do Projeto Geral
- **[Projeto Principal](../Projeto.md)** - Pesquisa de Iniciação Científica
- **[Aprendizado Supervisionado](../aprendizado-supervisionado/)** - Outros experimentos
- **[Aprendizado Não-Supervisionado](../aprendizado-não-supervisionado/)** - Técnicas relacionadas

## 📅 Cronologia

- **21/08/2025**: Implementação completa do projeto
- **21/08/2025**: Documentação técnica criada
- **21/08/2025**: Análise de resultados finalizada
- **21/08/2025**: Integração com pesquisa principal

## 🤝 Como Contribuir

### Para Pesquisa
1. Leia a documentação completa
2. Execute o projeto localmente
3. Analise os resultados
4. Sugira melhorias ou extensões

### Para Desenvolvimento
1. Clone o repositório
2. Instale as dependências
3. Execute os testes
4. Implemente novas técnicas

### Para Aplicação em FL
1. Estude a seção específica de FL
2. Adapte o código para seu caso
3. Valide com seus dados
4. Compartilhe resultados

---

*Índice atualizado: 21 de agosto de 2025*

**📞 Contato**: Consulte a documentação principal do projeto para informações de contato.
