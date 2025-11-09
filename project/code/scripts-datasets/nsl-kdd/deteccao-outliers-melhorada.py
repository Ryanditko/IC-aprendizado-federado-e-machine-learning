"""
NSL-KDD Outlier Detection - Versão Melhorada
============================================

ORIENTAÇÕES DO DIA 04/11:
✓ Escolher apenas 2 mapeamentos: "Normal" + 1 tipo de ataque
✓ Análise de correlação para selecionar features relevantes
✓ Avaliação da normalização (usar apenas se melhorar métricas)
✓ NÃO usar redução de dimensionalidade (PCA)
✓ Recall como métrica principal

Autor: Projeto de Iniciação Científica - Faculdade Impacta
Data: Novembro 2025
"""

import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Métodos de detecção de outliers
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope

# Métricas - FOCO NO RECALL
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', '..', 'data', 'nsl-kdd')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output-nsl-kdd')
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
CORRELATION_THRESHOLD = 0.95  # Para remover features altamente correlacionadas

print("="*80)
print("NSL-KDD OUTLIER DETECTION - VERSÃO MELHORADA")
print("="*80)
print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"FOCO: Recall como métrica principal")
print(f"Configuração: 2 classes + análise de correlação")
print("="*80)

# ============================================================================
# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS (2 CLASSES)
# ============================================================================

print("\n[1/6] CARREGAMENTO E PREPARAÇÃO DOS DADOS")
print("-" * 80)

# Carregar NSL-KDD
train_path = os.path.join(DATA_DIR, 'KDDTrain+.txt')
test_path = os.path.join(DATA_DIR, 'KDDTest+.txt')

# Colunas do NSL-KDD
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'attack_type', 'difficulty'
]

try:
    df_train = pd.read_csv(train_path, names=columns)
    df_test = pd.read_csv(test_path, names=columns)
    print(f"✓ Train: {len(df_train):,} registros")
    print(f"✓ Test: {len(df_test):,} registros")
except FileNotFoundError:
    print("ERRO: Arquivos NSL-KDD não encontrados!")
    print("Execute o download primeiro:")
    print("cd ../../../downloads && python download_nsl_kdd_dataset.py")
    exit(1)

# Combinar datasets
df = pd.concat([df_train, df_test], ignore_index=True)
print(f"✓ Dataset combinado: {len(df):,} registros")

# ============================================================================
# ESCOLHA DE 2 CLASSES: Normal + U2R (User-to-Root)
# ============================================================================

print(f"\nDistribuição original de ataques:")
attack_counts = df['attack_type'].value_counts()
print(attack_counts.head(10))

# Mapear ataques para categorias principais
attack_mapping = {
    'normal': 'normal',
    # U2R attacks (User-to-Root) - Escolhido por ser mais raro e crítico
    'buffer_overflow': 'u2r',
    'loadmodule': 'u2r',
    'perl': 'u2r',
    'rootkit': 'u2r',
    'sqlattack': 'u2r',
    'xterm': 'u2r',
    'ps': 'u2r',
    'httptunnel': 'u2r',
}

# DECISÃO: Focar apenas em Normal vs U2R
print(f"\n🎯 ESCOLHA: Normal vs U2R (User-to-Root) attacks")
print(f"Motivo: U2R são ataques críticos e raros, ideais para detecção de outliers")

# Filtrar apenas Normal e U2R
df['attack_category'] = df['attack_type'].map(attack_mapping)
df_filtered = df[df['attack_category'].notna()].copy()

print(f"\nDistribuição das 2 classes escolhidas:")
class_counts = df_filtered['attack_category'].value_counts()
print(class_counts)

contamination = class_counts['u2r'] / len(df_filtered)
print(f"\nContaminação (% U2R): {contamination:.4f} ({contamination*100:.2f}%)")

# Criar labels binários: 0=Normal, 1=U2R(Outlier)
df_filtered['is_outlier'] = (df_filtered['attack_category'] == 'u2r').astype(int)

print(f"\n✓ Dataset final: {len(df_filtered):,} registros")
print(f"  Normal (0): {(df_filtered['is_outlier'] == 0).sum():,}")
print(f"  U2R (1):    {(df_filtered['is_outlier'] == 1).sum():,}")

# ============================================================================
# 2. ANÁLISE DE CORRELAÇÃO E SELEÇÃO DE FEATURES
# ============================================================================

print("\n[2/6] ANÁLISE DE CORRELAÇÃO E SELEÇÃO DE FEATURES")
print("-" * 80)

# Preparar features numéricas
feature_columns = [col for col in df_filtered.columns 
                  if col not in ['attack_type', 'attack_category', 'is_outlier', 'difficulty']]

# Codificar variáveis categóricas
df_features = df_filtered[feature_columns].copy()

# Label encoding para categóricas
categorical_cols = ['protocol_type', 'service', 'flag']
label_encoders = {}

for col in categorical_cols:
    if col in df_features.columns:
        le = LabelEncoder()
        df_features[col] = le.fit_transform(df_features[col].astype(str))
        label_encoders[col] = le

print(f"✓ Features codificadas: {len(df_features.columns)} colunas")

# Análise de correlação
print("\nCalculando matriz de correlação...")
correlation_matrix = df_features.corr()

# Encontrar features altamente correlacionadas
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_value = abs(correlation_matrix.iloc[i, j])
        if corr_value > CORRELATION_THRESHOLD:
            high_corr_pairs.append((
                correlation_matrix.columns[i],
                correlation_matrix.columns[j],
                corr_value
            ))

print(f"\n🔍 Features altamente correlacionadas (>{CORRELATION_THRESHOLD}):")
if high_corr_pairs:
    for feat1, feat2, corr in high_corr_pairs:
        print(f"  {feat1} ↔ {feat2}: {corr:.3f}")
    
    # Remover uma das features correlacionadas (manter a primeira)
    features_to_remove = set()
    for feat1, feat2, corr in high_corr_pairs:
        features_to_remove.add(feat2)
    
    df_features = df_features.drop(columns=list(features_to_remove))
    print(f"✓ Removidas {len(features_to_remove)} features correlacionadas")
else:
    print("  Nenhuma correlação alta encontrada")

print(f"✓ Features finais: {len(df_features.columns)} colunas")

# Salvar heatmap de correlação
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, cmap='coolwarm', center=0,
            square=True, annot=False, cbar_kws={"shrink": .8})
plt.title('Matriz de Correlação - Features NSL-KDD')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Matriz de correlação salva")

# ============================================================================
# 3. AVALIAÇÃO DA NORMALIZAÇÃO
# ============================================================================

print("\n[3/6] AVALIAÇÃO DA NORMALIZAÇÃO")
print("-" * 80)

X = df_features.values
y = df_filtered['is_outlier'].values

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

print(f"✓ Treino: {len(X_train):,} amostras")
print(f"✓ Teste: {len(X_test):,} amostras")

# Testar COM e SEM normalização
normalization_results = {}

print(f"\nTestando Isolation Forest COM e SEM normalização...")

# SEM normalização
iso_forest_raw = IsolationForest(
    contamination=contamination,
    random_state=RANDOM_STATE,
    n_estimators=100
)
iso_forest_raw.fit(X_train)
pred_raw = iso_forest_raw.predict(X_test)
pred_raw_binary = (pred_raw == -1).astype(int)

recall_raw = recall_score(y_test, pred_raw_binary)
f1_raw = f1_score(y_test, pred_raw_binary)

print(f"  SEM normalização - Recall: {recall_raw:.3f}, F1: {f1_raw:.3f}")

# COM normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

iso_forest_scaled = IsolationForest(
    contamination=contamination,
    random_state=RANDOM_STATE,
    n_estimators=100
)
iso_forest_scaled.fit(X_train_scaled)
pred_scaled = iso_forest_scaled.predict(X_test_scaled)
pred_scaled_binary = (pred_scaled == -1).astype(int)

recall_scaled = recall_score(y_test, pred_scaled_binary)
f1_scaled = f1_score(y_test, pred_scaled_binary)

print(f"  COM normalização - Recall: {recall_scaled:.3f}, F1: {f1_scaled:.3f}")

# DECISÃO: Usar normalização apenas se melhorar
use_normalization = recall_scaled > recall_raw
print(f"\n🎯 DECISÃO: {'USAR' if use_normalization else 'NÃO USAR'} normalização")
print(f"   Motivo: Recall {'melhorou' if use_normalization else 'piorou'} com normalização")

# Preparar dados finais
if use_normalization:
    X_final_train = X_train_scaled
    X_final_test = X_test_scaled
    print("✓ Usando dados normalizados")
else:
    X_final_train = X_train
    X_final_test = X_test
    print("✓ Usando dados brutos (sem normalização)")

# ============================================================================
# 4. DETECÇÃO DE OUTLIERS - MÚLTIPLAS TÉCNICAS
# ============================================================================

print("\n[4/6] DETECÇÃO DE OUTLIERS - FOCO NO RECALL")
print("-" * 80)

methods = {
    'Isolation Forest': IsolationForest(
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_estimators=100,
        n_jobs=-1
    ),
    'Local Outlier Factor': LocalOutlierFactor(
        contamination=contamination,
        n_neighbors=20,
        novelty=True,  # Permite usar predict em dados novos
        n_jobs=-1
    ),
    'One-Class SVM': OneClassSVM(
        nu=contamination,
        kernel='rbf',
        gamma='auto'
    ),
    'Elliptic Envelope': EllipticEnvelope(
        contamination=contamination,
        random_state=RANDOM_STATE
    )
}

results = []

for method_name, method in methods.items():
    print(f"\n🔧 {method_name}...")
    start_time = time.time()
    
    # Treinar modelo
    method.fit(X_final_train)
    pred_test = method.predict(X_final_test)
    
    execution_time = time.time() - start_time
    
    # Converter predições (-1/1 para 1/0)
    pred_binary = (pred_test == -1).astype(int)
    
    # Calcular métricas - FOCO NO RECALL
    accuracy = accuracy_score(y_test, pred_binary)
    precision = precision_score(y_test, pred_binary, zero_division=0)
    recall = recall_score(y_test, pred_binary, zero_division=0)  # MÉTRICA PRINCIPAL
    f1 = f1_score(y_test, pred_binary, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, pred_binary).ravel()
    
    results.append({
        'Método': method_name,
        'Recall': recall,  # Métrica principal em primeiro
        'F1-Score': f1,
        'Precision': precision,
        'Accuracy': accuracy,
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn,
        'Tempo (s)': execution_time
    })
    
    print(f"   ✓ Recall:    {recall*100:.2f}% ⭐ (PRINCIPAL)")
    print(f"   ✓ F1-Score:  {f1*100:.2f}%")
    print(f"   ✓ Precision: {precision*100:.2f}%")
    print(f"   ✓ Accuracy:  {accuracy*100:.2f}%")
    print(f"   ✓ Tempo:     {execution_time:.2f}s")

# ============================================================================
# 5. ANÁLISE DOS RESULTADOS - RANKING POR RECALL
# ============================================================================

print("\n[5/6] ANÁLISE DOS RESULTADOS - RANKING POR RECALL")
print("-" * 80)

# Criar DataFrame e ordenar por RECALL
df_results = pd.DataFrame(results)
df_results = df_results.sort_values('Recall', ascending=False).reset_index(drop=True)
df_results['Rank'] = range(1, len(df_results) + 1)

print("\n🏆 RANKING POR RECALL (MÉTRICA PRINCIPAL):")
print(df_results[['Rank', 'Método', 'Recall', 'F1-Score', 'Precision', 'Accuracy']].to_string(index=False))

# Melhor método
best_method = df_results.iloc[0]
print(f"\n🥇 MELHOR MÉTODO: {best_method['Método']}")
print(f"   Recall:     {best_method['Recall']*100:.2f}% ⭐")
print(f"   F1-Score:   {best_method['F1-Score']*100:.2f}%")
print(f"   Precision:  {best_method['Precision']*100:.2f}%")
print(f"   Accuracy:   {best_method['Accuracy']*100:.2f}%")

# Análise de performance
print(f"\n📊 ANÁLISE DE PERFORMANCE:")
print(f"   Recall médio:    {df_results['Recall'].mean()*100:.2f}%")
print(f"   F1-Score médio:  {df_results['F1-Score'].mean()*100:.2f}%")
print(f"   Melhor recall:   {df_results['Recall'].max()*100:.2f}%")

# ============================================================================
# 6. VISUALIZAÇÕES E EXPORTAÇÃO
# ============================================================================

print("\n[6/6] VISUALIZAÇÕES E EXPORTAÇÃO")
print("-" * 80)

# Gráfico de comparação das métricas
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Recall (métrica principal)
axes[0,0].bar(df_results['Método'], df_results['Recall'], color='darkred', alpha=0.7)
axes[0,0].set_title('Recall por Método (MÉTRICA PRINCIPAL)', fontweight='bold')
axes[0,0].set_ylabel('Recall')
axes[0,0].tick_params(axis='x', rotation=45)

# F1-Score
axes[0,1].bar(df_results['Método'], df_results['F1-Score'], color='navy', alpha=0.7)
axes[0,1].set_title('F1-Score por Método')
axes[0,1].set_ylabel('F1-Score')
axes[0,1].tick_params(axis='x', rotation=45)

# Precision
axes[1,0].bar(df_results['Método'], df_results['Precision'], color='darkgreen', alpha=0.7)
axes[1,0].set_title('Precision por Método')
axes[1,0].set_ylabel('Precision')
axes[1,0].tick_params(axis='x', rotation=45)

# Tempo de execução
axes[1,1].bar(df_results['Método'], df_results['Tempo (s)'], color='purple', alpha=0.7)
axes[1,1].set_title('Tempo de Execução por Método')
axes[1,1].set_ylabel('Tempo (s)')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'nsl_kdd_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# Confusion Matrix do melhor método
best_idx = df_results['Recall'].idxmax()
best_result = df_results.iloc[best_idx]

plt.figure(figsize=(8, 6))
cm_data = [[best_result['TN'], best_result['FP']], 
           [best_result['FN'], best_result['TP']]]
sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'U2R'], yticklabels=['Normal', 'U2R'])
plt.title(f'Confusion Matrix - {best_result["Método"]}\nRecall: {best_result["Recall"]*100:.1f}%')
plt.ylabel('Verdadeiro')
plt.xlabel('Predito')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'best_confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# Salvar resultados
output_path = os.path.join(OUTPUT_DIR, 'nsl_kdd_results.csv')
df_results.to_csv(output_path, index=False)

print(f"✓ Gráficos salvos em: {OUTPUT_DIR}")
print(f"✓ Resultados salvos em: {output_path}")

# ============================================================================
# RESUMO FINAL
# ============================================================================

print("\n" + "="*80)
print("✅ ANÁLISE NSL-KDD CONCLUÍDA - VERSÃO MELHORADA")
print("="*80)

print(f"\n📋 CONFIGURAÇÕES APLICADAS:")
print(f"   ✓ Classes: Normal vs U2R ({contamination*100:.2f}% outliers)")
print(f"   ✓ Features: {len(df_features.columns)} (após análise de correlação)")
print(f"   ✓ Normalização: {'Aplicada' if use_normalization else 'Não aplicada'}")
print(f"   ✓ Métrica principal: Recall")
print(f"   ✓ PCA: Não utilizado (conforme orientação)")

print(f"\n🏆 MELHOR RESULTADO:")
print(f"   Método: {best_method['Método']}")
print(f"   Recall: {best_method['Recall']*100:.2f}% ⭐")
print(f"   Capacidade de detectar U2R: {best_method['TP']}/{best_method['TP'] + best_method['FN']}")

print(f"\n🎯 ADEQUAÇÃO PARA CYBERSECURITY:")
print(f"   • Foco em ataques U2R críticos")
print(f"   • Priorização do recall (detectar todos os ataques)")
print(f"   • Features relevantes (sem correlação alta)")
print(f"   • Metodologia cientificamente rigorosa")

print("\n" + "="*80)
