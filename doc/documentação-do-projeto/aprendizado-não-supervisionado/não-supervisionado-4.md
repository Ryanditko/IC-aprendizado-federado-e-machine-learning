Não Supervisionado 4

🧠 Resumo — Algoritmo Apriori para Regras de Associação
Este conteúdo detalha o funcionamento do algoritmo Apriori, usado em aprendizado não supervisionado para descobrir padrões e gerar regras de associação a partir de dados transacionais — tipo carrinho de compras de supermercado.

🧩 Objetivo
- Encontrar conjuntos de itens frequentes que aparecem juntos com certa regularidade.
- Gerar regras do tipo: "Se um cliente compra pão, há X% de chance de comprar manteiga também".

📦 Como funciona
- Os dados são estruturados como uma base transacional binária (1 = presente, 0 = ausente).
- O algoritmo trabalha com:
  * Suporte: frequência de um conjunto de itens no total de transações.
  * Confiança: probabilidade condicional de um item ocorrer dado que outro ocorreu.

🔁 Etapas do Apriori
1. Gerar subconjuntos de itens (Ck):
   - Começa com itens individuais (C1), depois pares (C2), trios (C3), etc.
2. Filtrar por suporte mínimo (Lk):
   - Mantém apenas subconjuntos com frequência suficiente.
3. Iteração com junção + poda:
   - Junção: combina itens de Lk-1 para formar novos candidatos Ck.
   - Poda: remove subconjuntos que não satisfazem o suporte mínimo.
4. Geração de regras de associação:
   - Calcula a confiança de regras do tipo antecedente → consequente.

🧠 Propriedade Apriori
- Se um conjunto de itens não é frequente, qualquer superconjunto dele também não será.
- Isso reduz drasticamente o número de combinações, tornando o algoritmo eficiente.

💻 Implementação em Python (mlxtend)
from mlxtend.frequent_patterns import apriori, association_rules
# 1. Criar DataFrame com True/False para itens
# 2. Usar apriori() para subconjuntos frequentes
# 3. Usar association_rules() com confiança mínima

🧪 Aplicações
- Recomendação de produtos
- Análise de comportamento de clientes
- Descoberta de padrões em registros médicos
- Detecção de eventos correlacionados em segurança

🧠 Alternativa: FP-Growth
- Mais eficiente que Apriori
- Evita gerar todos os subconjuntos candidatos
- Usa estrutura FP-Tree
- Analogia: Se Apriori é um detetive com caderninho, FP-Growth chega com mapa pronto.

🧾 Conclusão
- Apriori é um clássico da mineração de dados
- Simples, poderoso e ainda relevante
- Prático para extrair valor de dados não rotulados em alto volume
- Implementação facilitada pela biblioteca mlxtend em Python