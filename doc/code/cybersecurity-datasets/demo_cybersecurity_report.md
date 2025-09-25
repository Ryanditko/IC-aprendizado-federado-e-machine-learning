# Relatório Demo - Análise Cybersecurity

**ESTE É UM RELATÓRIO DE DEMONSTRAÇÃO COM DADOS SIMULADOS**

Data da análise: 2025-09-24 21:29:46

## 📊 Informações do Dataset (Simulado)

- **Dimensões**: 1,000 linhas × 9 colunas
- **Features**: packet_size, connection_duration, bytes_sent, bytes_received, num_connections, port_scan_attempts, failed_logins, protocol_violations
- **Classes**: Normal (0), Suspicious (1), Malicious (2)

## 🎯 Análise Supervisionada

- **Modelo**: RandomForestClassifier
- **Acurácia**: 0.9500

### Features Mais Importantes:
1. packet_size: 0.4673
2. failed_logins: 0.1039
3. bytes_sent: 0.0898
4. bytes_received: 0.0897
5. connection_duration: 0.0881

## 🔄 Análise Não Supervisionada

- **Melhor K (clusters)**: 8
- **Silhouette Score**: 0.1356

## 🚨 Detecção de Anomalias

- **Isolation Forest**: 10.0% anomalias
- **LOF**: 10.0% anomalias

## 💡 Próximos Passos

1. Configure as credenciais Kaggle para usar dados reais
2. Execute `python cybersecurity_analysis.py` para análise completa
3. Compare os resultados com dados reais vs simulados
