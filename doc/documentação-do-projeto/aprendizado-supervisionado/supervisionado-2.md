Resumo: Aprendizado Supervisionado - Classificação

Técnicas abordadas:
1. Naive Bayes
- Baseado no Teorema de Bayes
- Assume independência entre atributos
- Tipos: Gaussiano e Bernoulli
- Implementação scikit-learn: GaussianNB()

2. Regressão Logística
- Técnica probabilística para classificação binária
- Usa função logística para mapear valores entre 0 e 1
- Hiperplano de decisão em P(Y=1) > 0.5
- Implementação scikit-learn: LogisticRegression()

3. K-Vizinhos Mais Próximos (KNN)
- Classificador baseado em distância
- "Votação" dos k vizinhos mais próximos
- Métricas de distância: Euclidiana, Manhattan, etc.
- Implementação scikit-learn: KNeighborsClassifier()

Conceitos-chave:
- Probabilidade condicional e Teorema de Bayes
- Função logística (σ(z) = 1/(1 + e⁻ᶻ))
- Métricas de distância entre pontos
- Separação treino/teste
- Hiperparâmetros (valor de k no KNN)

Implementação Prática:
- Biblioteca scikit-learn
- Métodos principais:
  * fit(): treinamento do modelo
  * predict(): previsão em novos dados
- Exemplos com arrays NumPy

Aplicações:
- Classificação binária e multiclasse
- Problemas com variáveis categóricas
- Cenários onde relações probabilísticas são relevantes

Referências:
- Faceli et al. (2021) - Fundamentos de ML
- James et al. (2013) - Statistical Learning
- Gonzalez-Lopez (2010) - Probabilidade Condicional

Próximos passos:
- Explorar redes bayesianas
- Aprofundar métricas de avaliação
- Testar com conjuntos de dados reais