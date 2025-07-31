 Não Supervisionado 4

🧠 Resumo — Algoritmo Apriori para Regras de Associação
Este conteúdo detalha o funcionamento do algoritmo Apriori, usado em aprendizado não supervisionado para descobrir padrões e gerar regras de associação a partir de dados transacionais — tipo carrinho de compras de supermercado.

🧩 Objetivo
Encontrar conjuntos de itens frequentes que aparecem juntos com certa regularidade.

Gerar regras do tipo: “Se um cliente compra pão, há X% de chance de comprar manteiga também”.

📦 Como funciona
Os dados são estruturados como uma base transacional binária, onde cada transação é um vetor com 1 (presente) ou 0 (ausente) para cada item.

O algoritmo trabalha com:

Suporte: frequência de um conjunto de itens em relação ao total de transações.

Confiança: probabilidade condicional de um item ocorrer dado que outro ocorreu.

🔁 Etapas do Apriori
Gerar subconjuntos de itens (Ck): começa com itens individuais (C1), depois pares (C2), trios (C3), e assim por diante.

Filtrar por suporte mínimo (Lk): mantém apenas os subconjuntos com frequência suficiente.

Iteração com junção + poda:

Junção: combina itens de Lk-1 para formar novos candidatos Ck.

Poda: remove subconjuntos que não satisfazem o suporte mínimo.

Geração de regras de associação: baseando-se nos subconjuntos frequentes, calcula-se a confiança de regras do tipo antecedente → consequente.

🧠 Truque esperto: Propriedade Apriori
Se um conjunto de itens não é frequente, qualquer superconjunto dele também não será.

Isso reduz drasticamente o número de combinações, tornando o algoritmo viável mesmo para bases grandes.

💻 Implementação em Python
A biblioteca mlxtend facilita a aplicação prática. O código:

Cria o DataFrame com True/False para indicar presença/ausência.

Usa apriori() para extrair subconjuntos frequentes.

Usa association_rules() para gerar as regras com confiança mínima.

python
Copiar
Editar
from mlxtend.frequent_patterns import apriori, association_rules
🧪 Aplicações
Recomendação de produtos

Análise de comportamento de clientes

Descoberta de padrões em registros médicos

Segurança e detecção de eventos correlacionados

🧠 Bônus: FP-Growth
Como alternativa ao Apriori, o FP-Growth resolve o mesmo problema de forma mais eficiente, evitando gerar todos os subconjuntos candidatos. Ele usa uma estrutura chamada FP-Tree.

Resumindo: se o Apriori é o detetive que faz anotações no caderninho, o FP-Growth já chega com o mapa na mão.

🧾 Conclusão
O Apriori é um clássico da mineração de dados: simples, poderoso e ainda relevante. Sua aplicação em Python com mlxtend o torna uma ferramenta prática para qualquer analista ou cientista de dados que deseje extrair valor de dados não rotulados com alto volume.


