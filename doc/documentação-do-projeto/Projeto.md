MITIGAÇÃO DE ATAQUES POR ENVENENAMENTO EM APRENDIZADO FEDERADO: AVALIAÇÃO DE ABORDAGENS BASEADAS EM OUTLIERS

Resumo:
O aprendizado federado é um paradigma distribuído de aprendizado de máquina que preserva a privacidade ao permitir o treinamento colaborativo de modelos sem compartilhamento de dados brutos. Este estudo avalia abordagens baseadas em detecção de outliers para mitigar ataques por envenenamento, combinando revisão integrativa da literatura com simulações computacionais. Os resultados visam identificar estratégias eficazes para fortalecer a segurança em aplicações sensíveis à privacidade.

1. Introdução
- Contexto: Aprendizado Federado (FL) como abordagem descentralizada que preserva privacidade
- Desafio: Vulnerabilidade a ataques por envenenamento (dados e modelos)
- Solução proposta: Detecção de outliers em atualizações de modelos
- Relevância: Aplicações em ambientes regulados (LGPD, GDPR)

2. Objetivos
2.1 Objetivo Geral:
Investigar estratégias de prevenção/mitigação de ataques de envenenamento em FL

2.2 Objetivos Específicos:
- Analisar vulnerabilidades do FL a ataques por envenenamento
- Investigar métodos baseados em outliers
- Simular cenários de ataque e validação
- Validar abordagens de defesa

3. Metodologia
Abordagem mista:
3.1 Revisão Integrativa da Literatura (Whittemore e Knafl, 2005):
- Identificação do problema
- Busca sistemática
- Triagem e avaliação
- Síntese crítica

3.2 Pesquisa Experimental (Runeson e Höst, 2009):
- Projeto do estudo de caso
- Configuração de ambiente de simulação
- Execução de ataques controlados
- Aplicação de estratégias defensivas
- Análise estatística

4. Cronograma (12 meses)
Mês 1-3: Revisão Integrativa
Mês 4-6: Design Experimental
Mês 7-9: Simulações e Coleta de Dados
Mês 10-11: Análise e Validação
Mês 12: Redação e Submissão

Referências:
[1] ANITHA, G.; JEGATHEESAN, A. (2023) - Privacidade em FL e GDPR
[2] BHAGOJI, A. N. et al. (2019) - Análise de FL sob perspectiva adversária
[3] BLANCHARD, P. et al. (2017) - Gradiente tolerante a bizantinos
[4] ZHANG, J. et al. (2022) - Ameaças à segurança e privacidade em FL
[5] YAZDINEJAD, A et al. (2024) - Modelo robusto contra envenenamento

5. Implementação Prática
5.1 Projeto de Detecção de Outliers:
Como parte prática da pesquisa, foi desenvolvido um sistema completo de detecção de outliers para avaliar diferentes técnicas de identificação de anomalias. Este projeto serve como base experimental para as estratégias de defesa em aprendizado federado.

📁 Localização: `/outlier-detection/`

📊 Técnicas Implementadas:
- Z-Score: Detecção baseada em desvios padrão
- IQR: Método dos quartis interquartis  
- Isolation Forest: Isolamento de anomalias
- Local Outlier Factor (LOF): Densidade local
- DBSCAN: Clustering baseado em densidade
- One-Class SVM: Máquina de vetor de suporte
- Elliptic Envelope: Envoltória elíptica

🎯 Resultados Principais:
- Z-Score: 99,8% de acurácia, 98% de precisão e recall
- IQR: 99,2% de acurácia, 100% de recall
- Ensemble: Combinou múltiplos métodos com F1-score de 81%

📖 Documentação Completa:
- `/doc/documentação-do-projeto/detecção-de-outliers/README.md`
- `/doc/documentação-do-projeto/detecção-de-outliers/técnicas-detalhadas.md`
- `/doc/documentação-do-projeto/detecção-de-outliers/metodologia-avaliação.md`
- `/doc/documentação-do-projeto/detecção-de-outliers/aplicação-aprendizado-federado.md`

🔗 Aplicação em FL:
Os resultados indicam que Z-Score e IQR são os métodos mais adequados para detecção de atualizações maliciosas em ambientes de aprendizado federado, oferecendo alta precisão com baixo overhead computacional.

Contribuições Esperadas:
- Mapeamento de técnicas de mitigação
- Avaliação comparativa de abordagens baseadas em outliers ✅
- Diretrizes para implementação segura
- Fortalecimento da cibersegurança em FL
- Sistema prático de detecção de anomalias ✅