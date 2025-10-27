# MITIGAÇÃO DE ATAQUES POR ENVENENAMENTO EM APRENDIZADO FEDERADO

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange.svg)
![License](https://img.shields.io/badge/License-Academic-green.svg)

## 📋 Sobre o Projeto

Este projeto de Iniciação Científica investiga estratégias de **detecção de outliers** para mitigar **ataques por envenenamento** em sistemas de **Aprendizado Federado** (Federated Learning - FL). 

O aprendizado federado é um paradigma emergente de machine learning distribuído que permite treinamento colaborativo de modelos preservando a privacidade dos dados. No entanto, sua natureza descentralizada o torna vulnerável a **ataques maliciosos de envenenamento**, onde agentes adversários manipulam o processo de treinamento enviando atualizações corrompidas.

Este estudo combina **revisão integrativa da literatura** com **simulações computacionais** para avaliar técnicas defensivas baseadas em **detecção de anomalias**, contribuindo para o fortalecimento da segurança em sistemas de ML distribuído.

---

## 🎯 Objetivos

### Objetivo Geral

Investigar e avaliar estratégias de prevenção e mitigação de ataques de envenenamento em sistemas de aprendizado federado.

### Objetivos Específicos

1. Analisar vulnerabilidades do FL a ataques por envenenamento
2. Investigar métodos baseados em outliers para detecção de agentes maliciosos
3. Simular cenários controlados de ataque e defesa
4. Validar abordagens defensivas através de métricas quantitativas

---

## 🗂️ Estrutura do Repositório

```plaintext
.
├── README.md                         # Documentação principal
├── requirements.txt                  # Dependências Python
│
├── code/                             # Código-fonte dos experimentos
│   ├── iris-dataset/
│   ├── penguin-dataset/
│   └── weight_height/
│
├── data/                             # Datasets utilizados
│
├── notebooks/                        # Jupyter Notebooks e análises
│   ├── cyber-threat-detection/
│   ├── iris/
│   ├── penguin/
│   └── weight_height/
│
└── docs/                             # Documentação técnica completa
    ├── Projeto.md
    ├── aprendizado-supervisionado/
    ├── aprendizado-não-supervisionado/
    ├── avaliações-de-modelos/
    ├── desafios/
    └── detecção-de-outliers/
```

---

## � Metodologia

O projeto adota uma abordagem mista combinando pesquisa teórica e experimental:

### 1. Revisão Integrativa da Literatura

Revisão sistemática sobre ataques de envenenamento em aprendizado federado e técnicas de mitigação baseadas em detecção de outliers.

### 2. Simulações Computacionais

Implementação de experimentos controlados utilizando múltiplos datasets para avaliar técnicas de detecção de anomalias em diferentes contextos:

- **Datasets de Benchmark**: Iris, Penguins, Weight-Height
- **Dataset de Cybersecurity**: Detecção de ameaças cibernéticas
- **Técnicas de ML**: Supervisionado, Não-supervisionado, Detecção de Outliers
- **Métricas**: Accuracy, Precision, Recall, F1-Score, Silhouette, Davies-Bouldin

### 3. Validação Experimental

Verificação da eficácia das técnicas através da comparação entre outliers detectados e agentes maliciosos conhecidos.

---

## � Principais Resultados

Este projeto demonstra que **técnicas de detecção de outliers** são eficazes para identificar **agentes maliciosos** em sistemas de aprendizado federado.

### Técnicas de Detecção Avaliadas

- **Isolation Forest**: Detecção baseada em isolamento aleatório
- **Local Outlier Factor (LOF)**: Análise de densidade local
- **One-Class SVM**: Fronteira de decisão em alta dimensão
- **Elliptic Envelope**: Modelo gaussiano multivariado
- **DBSCAN**: Clustering baseado em densidade

### Performance em Detecção de Ameaças

| Técnica | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| **Elliptic Envelope** | **99.52%** | **97.62%** | **99.52%** | **98.56%** |
| Isolation Forest | 97.14% | 85.71% | 97.14% | 91.09% |
| LOF | 95.24% | 80.00% | 95.24% | 86.96% |
| One-Class SVM | 90.48% | 65.52% | 90.48% | 76.00% |
| DBSCAN | 85.71% | 55.56% | 85.71% | 67.57% |

**Conclusão**: As técnicas avaliadas demonstraram alta eficácia (85-99% de acurácia) na identificação de agentes maliciosos, validando a aplicabilidade de detecção de outliers como mecanismo de defesa em aprendizado federado.

---

## 🛠️ Tecnologias e Ferramentas

- **Linguagem**: Python 3.8+
- **Machine Learning**: scikit-learn, pandas, numpy
- **Visualização**: matplotlib, seaborn
- **Ambiente**: Jupyter Notebook, VS Code
- **Controle de Versão**: Git/GitHub

## 🚀 Como Executar

### Pré-requisitos

```bash
Python 3.8+
pip
```

### Instalação

```powershell
# Clone o repositório
git clone https://github.com/Ryanditko/IC-aprendizado-federado-e-machine-learning-em-cybersecurity.git
cd IC-aprendizado-federado-e-machine-learning-em-cybersecurity

# Instale as dependências
pip install -r requirements.txt
```

### Execução

**Notebooks Jupyter:**

```powershell
jupyter notebook
```

**Scripts Python:**

```powershell
# Exemplo: análise supervisionada do Iris
python code/iris-dataset/aprendizado-supervisionado.py

# Exemplo: detecção de outliers
python notebooks/cyber-threat-detection/cyber_threat_outlier_detection.py
```

---

## 📚 Documentação Completa

Para documentação técnica detalhada, consulte a pasta `docs/`:

- **[Projeto.md](docs/Projeto.md)**: Documento oficial do projeto
- **[Detecção de Outliers](docs/detecção-de-outliers/)**: Fundamentação teórica e aplicação em FL
- **[Avaliação de Modelos](docs/avaliações-de-modelos/)**: Metodologias e resultados experimentais

## 🔐 Aplicação em Aprendizado Federado

### Problema

Em sistemas de aprendizado federado, **clientes maliciosos** podem enviar **atualizações corrompidas** que comprometem o modelo global, caracterizando um **ataque de envenenamento**.

### Solução Proposta

**Pipeline de Defesa:**

1. Coleta de atualizações dos clientes participantes
2. **Detecção de outliers** nas atualizações recebidas
3. Filtragem de agentes suspeitos
4. Agregação robusta dos gradientes legítimos

### Técnicas Validadas

- **Elliptic Envelope**: 99.52% de acurácia na detecção
- **Isolation Forest**: 97.14% de acurácia
- **Byzantine-Robust Aggregation**: Complemento às técnicas de outliers

## 🤝 Contribuições Científicas

Este projeto contribui para o avanço do conhecimento em:

- **Segurança em Aprendizado Federado**: Mapeamento de técnicas de mitigação
- **Detecção de Anomalias**: Avaliação comparativa de métodos
- **Cibersegurança**: Diretrizes para implementação segura de ML distribuído

---

## 👤 Informações do Projeto

**Tipo**: Iniciação Científica  
**Instituição**: Faculdade Impacta  
**Área**: Ciência da Computação / Cibersegurança / Machine Learning

## 📄 Licença

Este projeto é de natureza acadêmica e destinado a fins educacionais e de pesquisa científica.

---

**Temas**: Segurança | Machine Learning | Federated Learning | Data Science

Projeto desenvolvido no âmbito do programa de Iniciação Científica
