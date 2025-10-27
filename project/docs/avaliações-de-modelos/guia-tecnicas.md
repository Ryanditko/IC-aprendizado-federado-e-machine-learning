# Guia Completo: Técnicas de Avaliação em Aprendizado Não Supervisionado

## Introdução

O aprendizado não supervisionado é uma categoria de machine learning onde trabalhamos com dados que não possuem rótulos ou respostas conhecidas. Por isso, avaliar a qualidade dos modelos se torna um desafio único, exigindo métricas específicas para cada tipo de técnica.

## 1. Agrupamento Particional (Partitional Clustering)

### O que é?
O agrupamento particional divide os dados em grupos (clusters) mutuamente exclusivos, onde cada ponto pertence a exatamente um cluster. O exemplo mais comum é o algoritmo K-Means.

### Como funciona?
- Define um número fixo de clusters (k)
- Inicializa centroides aleatoriamente
- Atribui cada ponto ao centroide mais próximo
- Recalcula os centroides baseado nos pontos atribuídos
- Repete até convergir

### Métricas de Avaliação:

#### 1.1 Coesão Intracluster (Within-cluster Sum of Squares - WCSS)
- **O que mede**: Compacidade dos pontos dentro de cada cluster
- **Como funciona**: Soma das distâncias quadráticas entre cada ponto e o centroide do seu cluster
- **Interpretação**: Valores menores indicam clusters mais compactos
- **Fórmula**: WCSS = Σ(distância(ponto, centroide)²)

#### 1.2 Separação Intercluster
- **O que mede**: Distância entre clusters diferentes
- **Como funciona**: Calcula a distância entre centroides de diferentes clusters
- **Interpretação**: Valores maiores indicam clusters bem separados

#### 1.3 Coeficiente de Silhueta
- **O que mede**: Combinação de coesão e separação
- **Como funciona**: Para cada ponto, calcula:
  - a = distância média aos pontos do mesmo cluster
  - b = distância média aos pontos do cluster mais próximo
  - silhueta = (b - a) / max(a, b)
- **Interpretação**: Varia de -1 a 1, onde:
  - Próximo de 1: ponto está bem agrupado
  - Próximo de 0: ponto está na fronteira entre clusters
  - Próximo de -1: ponto pode estar no cluster errado

#### 1.4 Índice Davies-Bouldin
- **O que mede**: Razão entre distâncias intra e inter-cluster
- **Interpretação**: Valores menores indicam melhor agrupamento

## 2. Agrupamento Hierárquico (Hierarchical Clustering)

### O que é?
Cria uma hierarquia de clusters representada por um dendrograma (árvore). Pode ser aglomerativo (bottom-up) ou divisivo (top-down).

### Como funciona (Aglomerativo)?
- Inicia com cada ponto como um cluster individual
- Repetidamente une os dois clusters mais próximos
- Para quando todos os pontos estão em um único cluster
- Produz um dendrograma mostrando a estrutura hierárquica

### Métricas de Avaliação:

#### 2.1 Coeficiente de Correlação Cofenética
- **O que mede**: Quão bem o dendrograma preserva as distâncias originais
- **Como funciona**: Correlaciona as distâncias originais com as distâncias cofenéticas (do dendrograma)
- **Interpretação**: Valores próximos de 1 indicam que o dendrograma representa bem os dados originais

#### 2.2 Avaliação Visual
- **Dendrograma**: Permite identificar visualmente o número ideal de clusters
- **Observar**: Grandes "saltos" na altura indicam pontos naturais de corte

## 3. Redução de Dimensionalidade

### O que é?
Técnicas que reduzem o número de características (dimensões) dos dados, mantendo a informação mais importante.

### Principais Técnicas:

#### 3.1 PCA (Principal Component Analysis)
- **Como funciona**: Encontra direções de máxima variância nos dados
- **Objetivo**: Projetar dados em espaço de menor dimensão preservando variância

#### 3.2 t-SNE (t-Distributed Stochastic Neighbor Embedding)
- **Como funciona**: Preserva relações de vizinhança local
- **Objetivo**: Visualização de dados de alta dimensão

#### 3.3 UMAP (Uniform Manifold Approximation and Projection)
- **Como funciona**: Preserva estrutura global e local
- **Objetivo**: Alternativa mais rápida ao t-SNE

### Métricas de Avaliação:

#### 3.1 Variância Explicada (PCA)
- **O que mede**: Porcentagem da variância original preservada
- **Como usar**: Escolher número de componentes que expliquem 80-95% da variância
- **Interpretação**: Maior variância explicada = melhor preservação da informação

#### 3.2 Erro de Reconstrução
- **O que mede**: Diferença entre dados originais e reconstruídos
- **Como calcular**: Norma da diferença entre X original e X reconstruído

#### 3.3 Trustworthiness e Continuity
- **Trustworthiness**: Vizinhos no espaço reduzido eram vizinhos no original?
- **Continuity**: Vizinhos no espaço original continuam vizinhos no reduzido?

## 4. Detecção de Anomalias

### O que é?
Identificação de pontos que diferem significativamente do padrão normal dos dados.

### Principais Técnicas:

#### 4.1 Isolation Forest
- **Como funciona**: Isola anomalias usando árvores aleatórias
- **Princípio**: Anomalias são mais fáceis de isolar

#### 4.2 One-Class SVM
- **Como funciona**: Aprende a fronteira dos dados "normais"
- **Princípio**: Pontos fora da fronteira são anomalias

#### 4.3 Local Outlier Factor (LOF)
- **Como funciona**: Compara densidade local de cada ponto com seus vizinhos
- **Princípio**: Anomalias têm densidade menor que seus vizinhos

### Métricas de Avaliação:

#### 4.1 Contamination Rate
- **O que mede**: Porcentagem de pontos classificados como anomalias
- **Como usar**: Deve ser ajustado baseado no conhecimento do domínio

#### 4.2 Scores de Anomalia
- **O que são**: Valores que indicam "quão anômalo" é cada ponto
- **Interpretação**: Valores mais altos = maior probabilidade de ser anomalia

## 5. Boas Práticas para Validação

### 5.1 Validação Cruzada em Clustering
- Usar diferentes inicializações aleatórias
- Testar diferentes números de clusters
- Comparar múltiplas métricas

### 5.2 Interpretação de Resultados
- Não confiar em uma única métrica
- Considerar o contexto do problema
- Validar com conhecimento do domínio

### 5.3 Visualização
- Usar gráficos de dispersão para clusters 2D/3D
- Plotar métricas vs. número de clusters
- Dendrogramas para clustering hierárquico

## Conclusão

A avaliação de modelos não supervisionados requer uma abordagem multifacetada, combinando:
- Múltiplas métricas quantitativas
- Análise visual
- Conhecimento do domínio
- Validação experimental

Cada técnica tem suas próprias métricas específicas, e a escolha da métrica adequada depende do objetivo da análise e das características dos dados.
