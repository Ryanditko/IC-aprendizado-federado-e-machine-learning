Não Supervisionado 3

🧠 Resumo — Redução de Dimensionalidade com PCA
Este conteúdo explora a técnica de redução de dimensionalidade em aprendizado não supervisionado, com foco em PCA (Principal Component Analysis). A ideia central é simplificar a representação de dados complexos, preservando ao máximo a variabilidade original. Tudo isso com exemplos práticos em Python com scikit-learn.

⚙️ Conceitos-Chave
- Redução de dimensionalidade: criar uma nova representação dos dados com menos variáveis, mas sem perder a essência.
- Componentes principais:
  * São combinações lineares das variáveis originais.
  * Cada componente representa uma direção de máxima variação dos dados.
  * O 1º componente capta a maior variância, o 2º é ortogonal ao primeiro e capta a segunda maior variância, e assim por diante.
- Objetivo: facilitar visualização, reduzir ruído, e melhorar desempenho de algoritmos.

🧪 Etapas do PCA
1. Normalização dos dados (zero média e desvio padrão 1).
2. Cálculo da matriz de covariância ou correlação.
3. Extração de autovalores e autovetores da matriz.
   - Autovalores: quanta variância cada componente captura.
   - Autovetores: direções dos novos eixos (componentes principais).
4. Ordenação dos componentes com base na variância explicada.
5. Projeção dos dados nos novos eixos.

📉 Quando aplicar PCA?
- Quando há muitos atributos (alta dimensionalidade).
- Quando há correlação entre variáveis.
- Antes de aplicar algoritmos como SVM, KNN, clustering etc.

💻 Exemplo com scikit-learn
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)  # Após normalização
X_reduzido = pca.transform(X)

📈 Visualização
- A visualização dos componentes principais mostra a estrutura dos dados em um espaço mais simples e mais interpretável.

🧠 PCA vs LDA
Técnica      Tipo de aprendizado  Usa rótulos?  Objetivo
PCA          Não supervisionado   ❌            Preservar variância
LDA          Supervisionado       ✅            Separar classes

🚀 Conclusão
PCA é uma ferramenta essencial para:
- Pré-processamento
- Visualização
- Melhoria de performance
- Compressão de dados

Dica: Compare o desempenho do modelo antes e depois do PCA usando datasets como o Iris para ver o impacto na prática.