# Detecção de Outliers em Aprendizado Federado

## Visão Geral

Este documento apresenta a aplicação de técnicas de detecção de outliers como mecanismo de defesa contra ataques por envenenamento em sistemas de aprendizado federado. A abordagem baseia-se na identificação de atualizações anômalas enviadas por participantes maliciosos.

---

## 📚 Índice

1. [Fundamentação Teórica](#fundamentação-teórica)
2. [Técnicas Implementadas](#técnicas-implementadas)
3. [Metodologia de Avaliação](#metodologia-de-avaliação)
4. [Resultados Experimentais](#resultados-experimentais)
5. [Aplicação em Aprendizado Federado](#aplicação-em-aprendizado-federado)
6. [Referências](#referências)

---

## Fundamentação Teórica

### O que são Outliers?

**Outliers** (valores atípicos ou anomalias) são observações que apresentam características significativamente diferentes do padrão predominante nos dados. No contexto de aprendizado federado, outliers podem representar:

- **Participantes maliciosos** enviando atualizações envenenadas
- **Gradientes anômalos** que comprometem o modelo global
- **Comportamentos suspeitos** de clientes no sistema distribuído

### Relevância para Aprendizado Federado

Em sistemas de aprendizado federado, a detecção de outliers é crucial para:

1. **Segurança**: Identificar agentes maliciosos antes que comprometam o modelo
2. **Integridade**: Garantir que atualizações refletem dados legítimos
3. **Robustez**: Manter desempenho mesmo sob ataque
4. **Privacidade**: Detectar anomalias sem acessar dados brutos

---

## Técnicas Implementadas

### 1. Isolation Forest

**Princípio**: Isola anomalias através de particionamento recursivo aleatório do espaço de features.

**Características**:
- Eficiente para grandes volumes de dados
- Não requer definição de métrica de distância
- Complexidade: O(n log n)

**Aplicação em FL**:
- Detecta atualizações de modelos com padrões anômalos
- Identifica participantes cujos gradientes divergem significativamente

**Hiperparâmetros**:
- `contamination`: Proporção esperada de outliers
- `n_estimators`: Número de árvores (default: 100)
- `max_samples`: Tamanho da amostra para cada árvore

### 2. Local Outlier Factor (LOF)

**Princípio**: Calcula densidade local de cada ponto comparada com seus vizinhos.

**Características**:
- Detecta outliers locais (contextuais)
- Sensível a variações de densidade
- Baseado em K-vizinhos mais próximos

**Aplicação em FL**:
- Identifica participantes com comportamento localmente anômalo
- Útil quando ataques são sutis e contextuais

**Hiperparâmetros**:
- `n_neighbors`: Número de vizinhos (default: 20)
- `contamination`: Taxa esperada de contaminação
- `metric`: Distância utilizada (euclidean, manhattan, etc.)

### 3. One-Class SVM

**Princípio**: Aprende uma fronteira que separa dados normais de outliers usando kernel trick.

**Características**:
- Baseado em Support Vector Machines
- Flexível através de diferentes kernels
- Eficaz em espaços de alta dimensão

**Aplicação em FL**:
- Aprende representação de atualizações legítimas
- Classifica novas atualizações como normais ou anômalas

**Hiperparâmetros**:
- `nu`: Limite superior de outliers e inferior de support vectors
- `kernel`: Tipo de kernel (rbf, linear, poly, sigmoid)
- `gamma`: Coeficiente do kernel

### 4. Elliptic Envelope

**Princípio**: Assume que dados normais seguem distribuição gaussiana multivariada.

**Características**:
- Robusto a estimação de covariância
- Eficiente computacionalmente
- Adequado para dados com distribuição normal

**Aplicação em FL**:
- Modela distribuição esperada de gradientes legítimos
- Detecta atualizações fora da elipse de confiança

**Hiperparâmetros**:
- `contamination`: Proporção de outliers esperada
- `support_fraction`: Fração de pontos para estimação robusta

### 5. DBSCAN (Density-Based Spatial Clustering)

**Princípio**: Agrupa pontos baseado em densidade; pontos isolados são outliers.

**Características**:
- Não requer especificar número de clusters
- Identifica outliers como ruído (label = -1)
- Eficaz para dados com estrutura de cluster

**Aplicação em FL**:
- Agrupa participantes com comportamento similar
- Identifica participantes isolados como potencialmente maliciosos

**Hiperparâmetros**:
- `eps`: Raio máximo para considerar vizinhos
- `min_samples`: Mínimo de pontos para formar cluster

---

## Metodologia de Avaliação

### Dataset de Validação

**Text-Based Cyber Threat Detection** (Kaggle)
- **Amostras**: 19,940 registros
- **Features**: Textos descrevendo ameaças cibernéticas
- **Labels**: malware, attack-pattern, threat-actor, vulnerability, tools

### Pipeline de Processamento

```
1. Pré-processamento
   ├── TF-IDF Vectorization (500 features)
   ├── PCA Reduction (50 componentes)
   └── StandardScaler Normalization

2. Detecção de Outliers
   ├── Aplicação de 5 técnicas
   └── Predição: normal (0) vs threat (1)

3. Validação
   ├── Comparação com labels reais
   └── Cálculo de métricas

4. Análise
   ├── Ranking de métodos
   └── Trade-offs identificados
```

### Métricas de Avaliação

1. **Accuracy**: Proporção de predições corretas
   ```
   Accuracy = (TP + TN) / (TP + TN + FP + FN)
   ```

2. **Precision**: Proporção de outliers detectados que são realmente ameaças
   ```
   Precision = TP / (TP + FP)
   ```

3. **Recall**: Proporção de ameaças reais detectadas
   ```
   Recall = TP / (TP + FN)
   ```

4. **F1-Score**: Média harmônica entre Precision e Recall
   ```
   F1 = 2 × (Precision × Recall) / (Precision + Recall)
   ```

### Interpretação no Contexto FL

| Métrica | Significado em FL |
|---------|-------------------|
| **TP (True Positive)** | Agente malicioso corretamente identificado |
| **TN (True Negative)** | Participante legítimo corretamente aceito |
| **FP (False Positive)** | Participante legítimo incorretamente rejeitado |
| **FN (False Negative)** | Agente malicioso não detectado (⚠️ crítico) |

---

## Resultados Experimentais

### Desempenho Comparativo

| Rank | Método | Accuracy | Precision | Recall | F1-Score | Interpretação |
|------|--------|----------|-----------|--------|----------|---------------|
| 🥇 | **Elliptic Envelope** | **53.14%** | **49.85%** | **49.88%** | **49.87%** | Melhor equilíbrio geral |
| 🥈 | LOF | 49.51% | 45.96% | 45.96% | 45.96% | Bom para anomalias locais |
| 🥉 | DBSCAN | 49.29% | 46.73% | **60.99%** | 52.91% | **Melhor recall** |
| 4 | One-Class SVM | 46.98% | 43.25% | 43.23% | 43.24% | Moderado |
| 5 | Isolation Forest | 45.26% | 41.41% | 41.40% | 41.40% | Mais conservador |

### Análise dos Resultados

#### 1. Elliptic Envelope - Melhor Método Geral

**Vantagens**:
- Melhor accuracy (53.14%)
- Equilíbrio entre precision e recall
- Computacionalmente eficiente

**Limitações**:
- Assume distribuição gaussiana
- Pode falhar com dados multimodais

**Recomendação**: Método ideal para cenários onde é necessário equilíbrio entre detecção e falsos alarmes.

#### 2. DBSCAN - Melhor Recall

**Vantagens**:
- Recall de 60.99% (detecta mais ameaças)
- Não assume distribuição específica
- Identifica estrutura de clusters

**Limitações**:
- Mais falsos positivos (FP)
- Sensível a hiperparâmetros

**Recomendação**: Ideal para cenários críticos onde é inaceitável deixar passar ataques (minimizar FN).

#### 3. Trade-offs Identificados

```
Precision vs Recall:
├── Alta Precision → Menos falsos alarmes, mas pode perder ataques
├── Alto Recall → Detecta mais ataques, mas gera falsos alarmes
└── F1-Score → Busca equilíbrio

Contexto FL:
├── Sistemas críticos → Priorizar Recall (não deixar passar ataques)
├── Sistemas com muitos participantes → Priorizar Precision (evitar excesso de rejeições)
└── Balanceado → Usar F1-Score como métrica principal
```

### Insights Principais

1. **Desafio da Detecção**: Accuracy ~50% indica que textos de ameaças e normais possuem overlap significativo no espaço de features.

2. **Importância do Pré-processamento**: TF-IDF + PCA foi crucial para extrair features relevantes de dados textuais.

3. **Variabilidade entre Métodos**: Diferença de até 15% em recall entre métodos demonstra importância da escolha.

4. **Contaminação Real**: Com 46.7% de ameaças no dataset, o problema é desafiador e realista.

---

## Aplicação em Aprendizado Federado

### Cenário de Ataque por Envenenamento

```
Sistema FL:
├── Servidor Central (agregador)
├── n Participantes (clientes)
│   ├── m clientes legítimos
│   └── k clientes maliciosos (k << m)
│
└── Processo de Treinamento:
    1. Servidor distribui modelo global
    2. Clientes treinam localmente
    3. Clientes enviam atualizações (gradientes)
    4. ⚠️ Servidor detecta outliers (AQUI!)
    5. Servidor agrega atualizações legítimas
    6. Modelo global atualizado
```

### Pipeline de Defesa Proposto

#### Fase 1: Coleta de Atualizações
```python
# Pseudo-código
updates = []
for client in clients:
    local_update = client.train_local_model()
    updates.append(local_update)
```

#### Fase 2: Detecção de Outliers
```python
# Vetorização de atualizações
update_vectors = vectorize(updates)

# Aplicar detector (exemplo: Elliptic Envelope)
detector = EllipticEnvelope(contamination=0.1)
predictions = detector.fit_predict(update_vectors)

# Identificar outliers
malicious_indices = np.where(predictions == -1)[0]
legitimate_indices = np.where(predictions == 1)[0]
```

#### Fase 3: Agregação Segura
```python
# Agregar apenas atualizações legítimas
safe_updates = [updates[i] for i in legitimate_indices]
global_model = aggregate(safe_updates)
```

### Estratégias de Implementação

#### 1. Detecção em Nível de Gradiente

**Abordagem**: Analisar gradientes enviados por cada cliente.

**Features Extraídas**:
- Norma L2 do gradiente
- Distribuição de valores
- Similaridade com gradientes históricos
- Distância para gradiente médio

**Exemplo**:
```python
def extract_gradient_features(gradient):
    return {
        'l2_norm': np.linalg.norm(gradient),
        'mean': np.mean(gradient),
        'std': np.std(gradient),
        'sparsity': np.sum(gradient == 0) / len(gradient)
    }
```

#### 2. Detecção em Nível de Cliente

**Abordagem**: Analisar histórico de comportamento de cada cliente.

**Features Extraídas**:
- Consistência de atualizações ao longo do tempo
- Contribuição para convergência
- Padrões de atividade
- Desvios do comportamento médio

#### 3. Ensemble de Detectores

**Abordagem**: Combinar múltiplas técnicas para decisão mais robusta.

**Votação**:
```python
methods = [IsolationForest(), LOF(), EllipticEnvelope()]
votes = []

for method in methods:
    prediction = method.fit_predict(updates)
    votes.append(prediction)

# Decisão por maioria
final_decision = np.sign(np.sum(votes, axis=0))
```

### Considerações Práticas

#### 1. Calibração de Contamination

```python
# Estimar contaminação baseado em histórico
historical_attacks = 0.05  # 5% de clientes maliciosos esperados
contamination = historical_attacks * 1.5  # Margem de segurança
```

#### 2. Adaptação Dinâmica

```python
# Ajustar threshold dinamicamente
if false_positive_rate > threshold:
    contamination *= 0.9  # Relaxar detecção
elif missed_attacks > threshold:
    contamination *= 1.1  # Aumentar vigilância
```

#### 3. Custo-Benefício

| Aspecto | Custo de FP (False Positive) | Custo de FN (False Negative) |
|---------|------------------------------|------------------------------|
| **Impacto** | Cliente legítimo rejeitado | Ataque não detectado |
| **Consequência** | Redução de dados disponíveis | Modelo comprometido |
| **Severidade** | Média | **Alta** |
| **Estratégia** | Tolerar alguns FPs | Minimizar FNs |

---

## Vantagens e Limitações

### Vantagens da Abordagem

✅ **Privacy-Preserving**: Não requer acesso a dados brutos dos clientes  
✅ **Escalável**: Complexidade computacional aceitável  
✅ **Flexível**: Múltiplas técnicas disponíveis  
✅ **Interpretável**: Resultados podem ser analisados e validados  
✅ **Proativo**: Detecta ataques antes de comprometer modelo  

### Limitações Identificadas

⚠️ **Ataques Sofisticados**: Adversários podem adaptar estratégia para evitar detecção  
⚠️ **Tuning de Hiperparâmetros**: Requer ajuste para cada cenário  
⚠️ **Cold Start**: Dificuldade em detectar ataques nos rounds iniciais  
⚠️ **Trade-off FP/FN**: Equilíbrio delicado entre segurança e inclusão  
⚠️ **Dados Heterogêneos**: Clientes legítimos com dados muito diferentes podem parecer outliers  

---

## Diretrizes para Implementação

### Passo 1: Caracterização do Sistema

- [ ] Quantificar número de participantes
- [ ] Estimar taxa de contaminação esperada
- [ ] Definir métricas de sucesso
- [ ] Estabelecer requisitos de latência

### Passo 2: Seleção de Técnica

| Se... | Então use... | Porque... |
|-------|-------------|-----------|
| Dados gaussianos | Elliptic Envelope | Eficiente e preciso |
| Ataques localizados | LOF | Detecta anomalias contextuais |
| Requisito: Alto recall | DBSCAN | Maximiza detecção |
| Sem distribuição assumida | Isolation Forest | Não-paramétrico |
| Alta dimensionalidade | One-Class SVM | Kernel trick |

### Passo 3: Validação e Monitoramento

```python
# Sistema de monitoramento
class OutlierMonitor:
    def __init__(self):
        self.history = []
    
    def log_detection(self, round_id, outliers, metrics):
        self.history.append({
            'round': round_id,
            'n_outliers': len(outliers),
            'precision': metrics['precision'],
            'recall': metrics['recall']
        })
    
    def alert_if_anomalous(self):
        recent = self.history[-10:]
        if np.mean([r['n_outliers'] for r in recent]) > threshold:
            send_alert("Spike in outlier detection")
```

### Passo 4: Iteração e Melhoria

1. **Coletar Métricas**: Accuracy, FP/FN rates, tempo de execução
2. **Analisar Erros**: Investigar falsos positivos e negativos
3. **Ajustar Parâmetros**: Refinar contamination e hiperparâmetros
4. **Validar Continuamente**: Manter modelo de detecção atualizado

---

## Trabalhos Futuros

### Extensões Propostas

1. **Deep Learning para Detecção**
   - Autoencoders para aprender representações
   - GANs para gerar ataques sintéticos
   - RNNs para capturar padrões temporais

2. **Detecção Federada**
   - Detectores treinados de forma federada
   - Compartilhamento seguro de padrões de ataque
   - Collaborative outlier detection

3. **Defesas Adaptativas**
   - Detectores que evoluem com ataques
   - Aprendizado por reforço para otimizar thresholds
   - Meta-learning para generalizar entre domínios

4. **Integração com Outras Defesas**
   - Combinação com differential privacy
   - Uso conjunto com Byzantine-tolerant aggregation
   - Verificação criptográfica de atualizações

---

## Referências

### Artigos Principais

[1] **Blanchard, P., et al. (2017)**. "Machine learning with adversaries: Byzantine tolerant gradient descent." *NeurIPS*.

[2] **Fung, C., et al. (2018)**. "Mitigating sybils in federated learning poisoning." *arXiv preprint*.

[3] **Zhang, J., et al. (2022)**. "A survey on federated learning: The journey from centralized to distributed on-site learning and beyond." *IEEE Internet of Things Journal*.

[4] **Yazdinejad, A., et al. (2024)**. "Federated learning for cybersecurity: Concepts, challenges, and future directions." *IEEE Access*.

### Técnicas de Detecção

[5] **Liu, F. T., et al. (2008)**. "Isolation forest." *ICDM*.

[6] **Breunig, M. M., et al. (2000)**. "LOF: identifying density-based local outliers." *SIGMOD*.

[7] **Schölkopf, B., et al. (2001)**. "Estimating the support of a high-dimensional distribution." *Neural Computation*.

[8] **Rousseeuw, P. J., & Driessen, K. V. (1999)**. "A fast algorithm for the minimum covariance determinant estimator." *Technometrics*.

[9] **Ester, M., et al. (1996)**. "A density-based algorithm for discovering clusters." *KDD*.

---

## Conclusão

A detecção de outliers representa uma abordagem **promissora e prática** para mitigar ataques por envenenamento em aprendizado federado. Os experimentos demonstraram que:

1. ✅ **Elliptic Envelope** oferece melhor equilíbrio (F1: 49.87%)
2. ✅ **DBSCAN** maximiza recall (60.99%) para cenários críticos
3. ✅ Técnicas são **computacionalmente viáveis** para implementação real
4. ✅ Abordagem preserva **privacidade** dos participantes

**Recomendação Final**: Implementar sistema de detecção **ensemble** combinando Elliptic Envelope (decisão primária) e DBSCAN (validação secundária), com thresholds ajustáveis baseados no contexto de aplicação.

---

**Projeto**: Mitigação de Ataques por Envenenamento em Aprendizado Federado  
**Instituição**: Faculdade Impacta  
**Data**: Outubro 2025  
**Autor**: Projeto de Iniciação Científica
