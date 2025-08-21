Resumo: Aprendizado Supervisionado - Regressão

Técnicas abordadas:
1. Regressão Linear
- Modelo: Y = a₁X₁ + ... + aₙXₙ + c
- Tipos:
  * Mínimos Quadrados Ordinários
  * Ridge (regularização L2)
  * Lasso (regularização L1)
- Implementação: LinearRegression(), Ridge(), Lasso()

2. Regressão Não Linear
- Modelos adaptados de técnicas de classificação:
  * KNN para Regressão (KNeighborsRegressor)
  * SVM para Regressão (SVR)
  * Árvore de Regressão (DecisionTreeRegressor)
  * Floresta Aleatória para Regressão (RandomForestRegressor)

Conceitos-chave:
- Espaço de atributos e funções de regressão
- Resíduos e funções de custo (RSS, RSE)
- Regularização (evitar overfitting)
- Variância como medida de pureza em regressão
- Adaptação de técnicas de classificação para regressão

Aplicações Práticas:
- Previsão de valores contínuos
- Modelagem de relações complexas
- Análise de tendências (ex: previsão de doenças)
- Problemas com padrões não lineares

Implementação:
- Biblioteca scikit-learn
- Pré-processamento necessário:
  * Codificação de variáveis qualitativas
  * Normalização para algumas técnicas
- Métricas de avaliação:
  * Erro quadrático médio (MSE)
  * Coeficiente de determinação (R²)

Referências:
- Faceli et al. (2021) - Fundamentos de ML
- James et al. (2013) - Statistical Learning

Próximos passos:
- Explorar outros modelos não lineares
- Aprofundar técnicas de regularização
- Testar com conjuntos de dados reais
- Comparar desempenho entre abordagens