Resumo: Aprendizado Supervisionado - Classificação (Continuação)

Técnicas abordadas:
1. Máquina de Vetores de Suporte (SVM)
- Encontra hiperplano ótimo para separar classes
- Maximiza a margem entre classes
- Tipos: Linear e não-linear (kernel trick)
- Implementação scikit-learn: SVC()

2. Árvore de Decisão
- Estrutura hierárquica de nós de decisão
- Métricas de impureza: Índice Gini e Entropia
- Divisão recursiva até critérios de parada
- Implementação scikit-learn: DecisionTreeClassifier()

3. Floresta Aleatória
- Ensemble de múltiplas Árvores de Decisão
- Técnicas: Bootstrap sampling e Feature bagging
- Votação majoritária para classificação final
- Implementação scikit-learn: RandomForestClassifier()

Conceitos-chave:
- Espaço de atributos e hiperplanos
- Margens suaves em SVM
- Impureza e homogeneidade em nós
- Aleatoriedade controlada (bootstrap/bagging)
- Interpretabilidade vs performance

Aplicações Práticas:
- Análise de sentimentos (SVM)
- Classificação de dados complexos
- Problemas com alta dimensionalidade
- Cenários que exigem robustez

Implementação:
- Biblioteca scikit-learn
- Pré-processamento necessário
- Parâmetros ajustáveis:
  * kernel (SVM)
  * profundidade máxima (Árvores)
  * número de estimadores (Florestas)

Referências:
- Faceli et al. (2021) - Fundamentos de ML
- James et al. (2013) - Statistical Learning

Próximos passos:
- Aprofundar parâmetros e tuning
- Explorar outros kernels para SVM
- Testar com conjuntos de dados reais
- Comparar desempenho entre técnicas