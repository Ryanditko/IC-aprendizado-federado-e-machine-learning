"""
Script para verificar compatibilidade do dataset cyber com os resultados existentes
"""

import pandas as pd
import numpy as np
import os

print("="*80)
print("VERIFICAÇÃO: DATASET CYBER VS RESULTADOS EXISTENTES")
print("="*80)

# 1. Verificar resultados existentes
print("\n[1/4] RESULTADOS EXISTENTES (notebooks/cyber-outlier-detection/results/)")
print("-"*80)

results_file = "notebooks/cyber-outlier-detection/results/evaluation_metrics.csv"
if os.path.exists(results_file):
    df_results = pd.read_csv(results_file)
    print("\n✓ Arquivo de métricas encontrado:")
    print(df_results.to_string(index=False))
    print(f"\nMelhor método: {df_results.loc[df_results['Accuracy'].idxmax(), 'Método']}")
    print(f"Melhor accuracy: {df_results['Accuracy'].max():.4f}")
else:
    print("❌ Arquivo de métricas não encontrado!")

# 2. Verificar dataset instalado
print("\n[2/4] DATASET INSTALADO (data/)")
print("-"*80)

data_files = [f for f in os.listdir('data') if 'cyber' in f.lower() and f.endswith('.csv')]
print(f"\n✓ Encontrados {len(data_files)} arquivos:")
for f in data_files:
    size_mb = os.path.getsize(f'data/{f}') / (1024 * 1024)
    print(f"  • {f} ({size_mb:.2f} MB)")

# 3. Analisar dataset principal
print("\n[3/4] ANÁLISE DO DATASET PRINCIPAL")
print("-"*80)

df_cyber = pd.read_csv('data/cyber-threat-intelligence_all.csv')
print(f"\n✓ Dataset carregado: {df_cyber.shape[0]} linhas x {df_cyber.shape[1]} colunas")

print("\nColunas disponíveis:")
for i, col in enumerate(df_cyber.columns, 1):
    non_null = df_cyber[col].notna().sum()
    null_pct = (df_cyber[col].isna().sum() / len(df_cyber)) * 100
    print(f"  {i}. {col:20s} - {df_cyber[col].dtype:10s} ({non_null:5d} não-nulos, {null_pct:5.1f}% nulos)")

# 4. Verificar compatibilidade
print("\n[4/4] ANÁLISE DE COMPATIBILIDADE")
print("-"*80)

issues = []

# Verificar label column
if 'label' in df_cyber.columns:
    print("\n✓ Coluna 'label' encontrada")
    print(f"  Labels únicos: {df_cyber['label'].nunique()}")
    print(f"  Labels não-nulos: {df_cyber['label'].notna().sum()}")
    print("\n  Top 10 labels:")
    print(df_cyber['label'].value_counts().head(10).to_string())
else:
    issues.append("❌ Coluna 'label' não encontrada (necessária para validação)")

# Verificar coluna text
if 'text' in df_cyber.columns:
    print("\n✓ Coluna 'text' encontrada")
    print(f"  Textos únicos: {df_cyber['text'].nunique()}")
    print(f"  Textos não-nulos: {df_cyber['text'].notna().sum()}")
    avg_length = df_cyber['text'].str.len().mean()
    print(f"  Comprimento médio: {avg_length:.0f} caracteres")
else:
    issues.append("❌ Coluna 'text' não encontrada (necessária para feature extraction)")

# Verificar entities
if 'entities' in df_cyber.columns:
    print("\n✓ Coluna 'entities' encontrada")
    non_null = df_cyber['entities'].notna().sum()
    print(f"  Registros com entities: {non_null} ({(non_null/len(df_cyber))*100:.1f}%)")
else:
    issues.append("⚠ Coluna 'entities' não encontrada")

# Resumo
print("\n" + "="*80)
print("RESUMO DA VERIFICAÇÃO")
print("="*80)

if not issues:
    print("\n✅ DATASET COMPATÍVEL!")
    print("\nO dataset baixado contém todas as colunas necessárias para:")
    print("  1. Feature extraction (coluna 'text')")
    print("  2. Validação de outliers (coluna 'label')")
    print("  3. Análise de entidades (coluna 'entities')")
    print("\n⚠ ATENÇÃO:")
    print("  - O notebook atual está INCOMPLETO (só tem setup)")
    print("  - Os resultados em 'results/' foram gerados com dataset DIFERENTE")
    print("  - É necessário ATUALIZAR o notebook para usar o novo dataset")
else:
    print("\n⚠ PROBLEMAS ENCONTRADOS:")
    for issue in issues:
        print(f"  {issue}")

print("\n" + "="*80)
print("RECOMENDAÇÕES")
print("="*80)
print("\n1. ATUALIZAR NOTEBOOK:")
print("   - Completar células de carregamento de dados")
print("   - Adaptar para colunas: text, label, entities")
print("   - Usar TF-IDF ou embeddings para processar 'text'")
print("\n2. RE-EXECUTAR ANÁLISE:")
print("   - Aplicar os 5 métodos: IsolationForest, LOF, OneClassSVM, EllipticEnvelope, DBSCAN")
print("   - Validar outliers contra coluna 'label'")
print("   - Gerar novas visualizações e métricas")
print("\n3. COMPARAR RESULTADOS:")
print("   - Resultados antigos: Elliptic Envelope (99.52% accuracy)")
print("   - Verificar se o novo dataset mantém performance similar")

print("\n" + "="*80)
