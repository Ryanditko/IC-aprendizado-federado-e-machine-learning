ALGORITMO APRIORI: MINERAÇÃO DE REGRAS DE ASSOCIAÇÃO

Resumo:
Este documento detalha o algoritmo Apriori aplicado em aprendizado não supervisionado para descoberta de padrões e geração de regras de associação em dados transacionais. A técnica é fundamental para análise de comportamento de compra, sistemas de recomendação e identificação de relações frequentes entre itens.

1. Fundamentos e Objetivos

1.1 Propósito:
- Identificação de conjuntos de itens que ocorrem frequentemente juntos
- Geração de regras associativas do tipo: "Se X, então Y com probabilidade P%"
- Descoberta de padrões não óbvios em grandes volumes de dados transacionais

1.2 Estrutura de Dados:
- Base transacional: Matriz binária (1 = item presente, 0 = ausente)
- Cada linha representa uma transação
- Cada coluna representa um item

2. Conceitos Fundamentais

2.1 Métricas Básicas:

Suporte (Support):
- Definição: Frequência relativa de um conjunto de itens
- Fórmula: suporte(X) = |transações contendo X| / |total de transações|
- Interpretação: Proporção de transações que contêm o itemset

Confiança (Confidence):
- Definição: Probabilidade condicional P(Y|X)
- Fórmula: confiança(X → Y) = suporte(X ∪ Y) / suporte(X)
- Interpretação: Dado X, qual a probabilidade de Y ocorrer

Lift:
- Definição: Razão entre confiança observada e esperada
- Fórmula: lift(X → Y) = confiança(X → Y) / suporte(Y)
- Interpretação: lift > 1 indica correlação positiva

2.2 Propriedade Apriori (Anti-monotonicidade):
Se um conjunto de itens é infrequente, todos os seus superconjuntos também são infrequentes.

Implicação: Permite poda eficiente do espaço de busca, eliminando candidatos sem necessidade de cálculo explícito.

3. Metodologia do Algoritmo

3.1 Processo Iterativo:

Etapa 1 - Geração de Candidatos (Ck):
- C₁: Itemsets de tamanho 1 (itens individuais)
- C₂: Itemsets de tamanho 2 (pares)
- Cₖ: Itemsets de tamanho k

Etapa 2 - Filtragem por Suporte Mínimo (Lk):
- Cálculo do suporte de cada candidato
- Retenção apenas de itemsets com suporte ≥ minsup
- L₁, L₂, ..., Lₖ: Conjuntos de itemsets frequentes

Etapa 3 - Junção (Join):
- Combinação de itemsets de Lₖ₋₁ para formar candidatos Cₖ
- União de itemsets que diferem em apenas um item

Etapa 4 - Poda (Prune):
- Eliminação de candidatos cujos subconjuntos não são frequentes
- Aplicação da propriedade Apriori

Etapa 5 - Geração de Regras:
- Para cada itemset frequente, geração de regras do tipo X → Y
- Cálculo de confiança para cada regra
- Retenção de regras com confiança ≥ minconf

3.2 Pseudocódigo:
```
Apriori(T, minsup):
    L₁ = {itemsets frequentes de tamanho 1}
    k = 2
    while Lₖ₋₁ ≠ ∅:
        Cₖ = gerar_candidatos(Lₖ₋₁)  # Join
        for cada transação t ∈ T:
            Cₜ = subconjunto(Cₖ, t)
            for cada candidato c ∈ Cₜ:
                c.count++
        Lₖ = {c ∈ Cₖ | c.count ≥ minsup}  # Prune
        k++
    return ⋃ₖ Lₖ
```

4. Implementação com Python

4.1 Biblioteca mlxtend:
```python
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Preparação dos dados (formato transacional)
# DataFrame com valores booleanos (True/False)
transactions = pd.DataFrame({
    'pão': [True, True, False, True],
    'leite': [True, True, True, False],
    'manteiga': [True, False, True, True]
})

# Aplicação do Apriori
frequent_itemsets = apriori(
    transactions,
    min_support=0.5,
    use_colnames=True
)

# Geração de regras
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.7
)

# Análise dos resultados
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
```

4.2 Interpretação dos Resultados:
```python
# Exemplo de regra encontrada:
# {pão} → {manteiga}
# support: 0.60 (60% das transações contêm ambos)
# confidence: 0.75 (75% das transações com pão contêm manteiga)
# lift: 1.25 (correlação positiva)
```

5. Eficiência Computacional

5.1 Complexidade:
- Pior caso: Exponencial no número de itens
- Prática: Drasticamente reduzido pela poda
- Número de candidatos gerados << 2ⁿ (n = número de itens)

5.2 Otimizações:
- Estruturas de dados eficientes (hash trees)
- Amostragem para grandes datasets
- Processamento paralelo
- Técnicas de compressão de transações

6. Aplicações Práticas
- Análise de cesta de compras em varejo
- Sistemas de recomendação de produtos
- Descoberta de padrões em registros médicos eletrônicos
- Análise de clickstream em websites
- Detecção de fraudes através de padrões suspeitos
- Análise de co-ocorrência em textos e documentos
- Identificação de eventos correlacionados em segurança cibernética

7. Algoritmos Alternativos

7.1 FP-Growth (Frequent Pattern Growth):
Características:
- Mais eficiente que Apriori para grandes datasets
- Evita geração explícita de candidatos
- Utiliza estrutura FP-Tree (Frequent Pattern Tree)
- Crescimento de padrões através de projeções recursivas

Vantagens:
- Apenas duas passagens sobre o dataset
- Compressão eficiente via FP-Tree
- Melhor desempenho em datasets densos

Desvantagens:
- Maior uso de memória
- Implementação mais complexa

7.2 Eclat (Equivalence Class Transformation):
- Abordagem baseada em intersecção de conjuntos
- Utiliza representação vertical (TID-sets)
- Eficiente para datasets com transações longas

8. Considerações Práticas

8.1 Seleção de Parâmetros:
- Suporte mínimo: Trade-off entre completude e eficiência
  * Muito baixo: Muitos itemsets, alto custo computacional
  * Muito alto: Perda de padrões interessantes
- Confiança mínima: Depende da aplicação
  * Recomendações: 60-80%
  * Regras críticas: >90%

8.2 Validação de Regras:
- Análise de lift para evitar correlações espúrias
- Validação de significância estatística
- Consideração do contexto de negócio
- Testes A/B para regras aplicadas em produção

9. Considerações Finais
O algoritmo Apriori representa marco fundamental na mineração de dados, mantendo relevância contemporânea pela sua simplicidade conceitual e eficácia em descobrir padrões associativos. Embora existam alternativas mais eficientes (FP-Growth), o Apriori continua sendo método didático essencial e base para compreensão de técnicas avançadas de mineração de regras de associação. A implementação via mlxtend em Python democratiza o acesso prático a esta poderosa ferramenta analítica.

Referências:
[1] AGRAWAL, R.; SRIKANT, R. (1994) - Fast Algorithms for Mining Association Rules
[2] HAN, J. et al. (2000) - Mining Frequent Patterns without Candidate Generation (FP-Growth)
[3] Biblioteca mlxtend - Implementação Python de algoritmos de mineração

Contribuições Esperadas:
- Metodologia para descoberta automática de padrões associativos
- Técnicas para sistemas de recomendação
- Base para análise de comportamento em grandes volumes de dados