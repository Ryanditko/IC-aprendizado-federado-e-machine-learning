Não Supervisionado 2

🧠 Resumo — Agrupamento no Aprendizado Não Supervisionado
O texto explora as técnicas de agrupamento (clustering) em aprendizado não supervisionado, com foco em K-Means (particional) e AGNES e DIANA (hierárquico), além de conceitos fundamentais como funções de distância e centróides.

⚙️ Conceitos-Chave
- Aprendizado não supervisionado: identifica padrões e estruturas nos dados sem rótulos pré-definidos.
- Cluster: grupo de dados semelhantes entre si.
- Funções de distância: calculam quão "parecidos" ou "diferentes" os pontos são. Exemplos:
  * Euclidiana
  * Manhattan
  * Minkowski
  * Chebyshev
- Centróide: ponto central representativo de um cluster.

📌 Técnicas de Agrupamento

🔷 1. Agrupamento Particional — K-Means
- Divide os dados em K clusters pré-definidos.
- Processo:
  1. Escolhe K centróides aleatórios.
  2. Atribui cada ponto ao centróide mais próximo.
  3. Recalcula os centróides com base na média dos pontos do cluster.
  4. Repete até estabilizar.
- Critérios de parada: convergência dos centróides, número máximo de iterações, etc.
- Aplicação prática com scikit-learn usando KMeans + make_blobs.

🧪 Exemplo K-Means:
from sklearn.cluster import KMeans
model = KMeans(n_clusters=4)
model.fit(X)

🔷 2. Agrupamento Hierárquico — AGNES e DIANA
- Organiza os dados em uma estrutura em árvore (dendrograma).
- Útil para analisar em diferentes níveis de granularidade.

✅ AGNES (Aglomerativo):
- Começa com cada ponto como um cluster individual.
- Junta os clusters mais semelhantes iterativamente.
- Usa linkage para medir a distância entre clusters:
  * Linkage completo
  * Linkage simples
  * Linkage médio
  * Linkage centróide

✅ DIANA (Divisivo):
- Começa com todos os pontos em um único cluster.
- Vai dividindo os grupos conforme a maior dissimilaridade entre pontos.
- Mais raro e sem implementação direta no scikit-learn.

🧪 Exemplo AGNES:
from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
model.fit(X)

📊 Visualização:
- Dendrogramas são usados para visualizar hierarquias de similaridade.
- Ajudam a interpretar como os clusters se formam ou se dividem.

🚀 Aplicações Práticas
- Segmentação de mercado
- Análise genômica
- Reconhecimento de padrões
- Exploração de dados médicos, imagens, etc.

🧰 Resumo Final
- O agrupamento no aprendizado não supervisionado é uma ferramenta poderosa para descobrir padrões naturais em dados não rotulados.
- K-Means é direto, eficiente e muito usado.
- AGNES e DIANA oferecem visão hierárquica, sendo ideais quando a estrutura dos dados não é clara.
- Com o apoio de bibliotecas como o scikit-learn, essas técnicas se tornam acessíveis e práticas para resolver problemas reais.