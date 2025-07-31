Resumo: Avaliação de modelos em aprendizado não supervisionado

Este texto aborda como avaliar modelos em aprendizado não supervisionado, focando em três áreas principais: agrupamento particional, agrupamento hierárquico e redução de dimensionalidade.

1. Agrupamento Particional:
   - Métricas:
     • Coesão intracluster (compacidade): proximidade entre pontos de um mesmo cluster.
     • Separação intercluster: distância entre clusters diferentes.
     • Coeficiente de silhueta: combina coesão e separação; varia entre -1 e 1, onde valores altos indicam bom agrupamento.
   - Pode-se usar métricas de classificação (acurácia, precisão etc.) caso se tenha dados rotulados.

2. Agrupamento Hierárquico:
   - Difere do particional por produzir uma estrutura em árvore (dendrograma).
   - Métrica principal: Coeficiente de correlação cofenética (rc) — mede o quanto a estrutura do dendrograma preserva as distâncias originais dos dados.
   - Avaliação visual também é útil para entender os agrupamentos formados.

3. Redução de Dimensionalidade:
   - Exemplo com PCA (Análise de Componentes Principais).
   - Métrica: Variância explicada — representa a quantidade de informação preservada pelos componentes principais escolhidos.
   - A escolha do número de componentes depende do percentual acumulado da variância explicada (e.g., 95%).

Conclusão:
Avaliar modelos em aprendizado não supervisionado exige o uso criterioso de métricas específicas para cada técnica. O uso de ferramentas práticas como Python permite verificar, ajustar e melhorar a performance dos modelos baseando-se em métricas internas confiáveis.


