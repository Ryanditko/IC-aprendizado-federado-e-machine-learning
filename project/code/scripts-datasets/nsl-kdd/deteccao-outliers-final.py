"""
NSL-KDD Outlier Detection - Versão Final
========================================

ORIENTAÇÕES DO DIA 04/11 IMPLEMENTADAS:
✓ Apenas 2 classes: Normal vs U2R
✓ Análise de correlação para seleção de features
✓ Avaliação de normalização (só usar se melhorar)
✓ SEM redução de dimensionalidade (PCA)
✓ Recall como métrica principal
"""

import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', '..', 'data', 'nsl-kdd')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output-nsl-kdd')
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
CORRELATION_THRESHOLD = 0.95

print("="*80)
print("NSL-KDD OUTLIER DETECTION - VERSÃO FINAL")
print("="*80)
print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"FOCO: Recall como métrica principal")
print(f"Configuração: 2 classes + análise de correlação")
print("="*80)

# 1. CARREGAMENTO DOS DADOS
print("\n[1/6] CARREGAMENTO DOS DADOS")
print("-" * 80)

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

train_path = os.path.join(DATA_DIR, 'KDDTrain+.txt')
test_path = os.path.join(DATA_DIR, 'KDDTest+.txt')

df_train = pd.read_csv(train_path, names=columns)
df_test = pd.read_csv(test_path, names=columns)
df = pd.concat([df_train, df_test], ignore_index=True)

print(f"✓ Dataset combinado: {len(df):,} registros")

# 2. FILTRAR APENAS NORMAL vs U2R
attack_mapping = {
    'normal': 'normal',
    'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'perl': 'u2r',
    'rootkit': 'u2r', 'sqlattack': 'u2r', 'xterm': 'u2r', 'ps': 'u2r'
}

df['attack_category'] = df['attack_type'].map(attack_mapping)
df_filtered = df[df['attack_category'].notna()].copy()
df_filtered['is_outlier'] = (df_filtered['attack_category'] == 'u2r').astype(int)

class_counts = df_filtered['attack_category'].value_counts()
contamination = class_counts['u2r'] / len(df_filtered)

print(f"🎯 ESCOLHA: Normal vs U2R")
print(f"  Normal: {class_counts['normal']:,}")
print(f"  U2R: {class_counts['u2r']:,}")
print(f"  Contaminação: {contamination:.4f} ({contamination*100:.2f}%)")

# 3. ANÁLISE DE CORRELAÇÃO
print("\n[2/6] ANÁLISE DE CORRELAÇÃO")
print("-" * 80)

feature_columns = [col for col in df_filtered.columns 
                  if col not in ['attack_type', 'attack_category', 'is_outlier', 'difficulty']]

df_features = df_filtered[feature_columns].copy()

# Codificar categóricas
categorical_cols = ['protocol_type', 'service', 'flag']
for col in categorical_cols:
    if col in df_features.columns:
        le = LabelEncoder()
        df_features[col] = le.fit_transform(df_features[col].astype(str))

print(f"✓ Features iniciais: {len(df_features.columns)}")

# Matriz de correlação
correlation_matrix = df_features.corr()

# Remover features altamente correlacionadas
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_value = abs(correlation_matrix.iloc[i, j])
        if corr_value > CORRELATION_THRESHOLD:
            high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], corr_value))

if high_corr_pairs:
    features_to_remove = set()
    for feat1, feat2, corr in high_corr_pairs:
        features_to_remove.add(feat2)
        print(f"  Removendo {feat2} (correlação com {feat1}: {corr:.3f})")
    
    df_features = df_features.drop(columns=list(features_to_remove))

print(f"✓ Features finais: {len(df_features.columns)}")

# Salvar heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df_features.corr(), cmap='coolwarm', center=0, square=True, 
            annot=False, cbar_kws={"shrink": .8})
plt.title('Matriz de Correlação - Features NSL-KDD (Filtradas)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Matriz de correlação salva")

# 4. AVALIAÇÃO DA NORMALIZAÇÃO
print("\n[3/6] AVALIAÇÃO DA NORMALIZAÇÃO")
print("-" * 80)

X = df_features.values
y = df_filtered['is_outlier'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

print(f"✓ Treino: {len(X_train):,} | Teste: {len(X_test):,}")

# Testar normalização
iso_raw = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
iso_raw.fit(X_train)
pred_raw = (iso_raw.predict(X_test) == -1).astype(int)
recall_raw = recall_score(y_test, pred_raw)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

iso_scaled = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
iso_scaled.fit(X_train_scaled)
pred_scaled = (iso_scaled.predict(X_test_scaled) == -1).astype(int)
recall_scaled = recall_score(y_test, pred_scaled)

use_normalization = recall_scaled > recall_raw
print(f"  SEM normalização - Recall: {recall_raw:.3f}")
print(f"  COM normalização - Recall: {recall_scaled:.3f}")
print(f"🎯 DECISÃO: {'USAR' if use_normalization else 'NÃO USAR'} normalização")

if use_normalization:
    X_final_train, X_final_test = X_train_scaled, X_test_scaled
else:
    X_final_train, X_final_test = X_train, X_test

# 5. DETECÇÃO DE OUTLIERS
print("\n[4/6] DETECÇÃO DE OUTLIERS - FOCO NO RECALL")
print("-" * 80)

methods = {
    'Isolation Forest': IsolationForest(contamination=contamination, random_state=RANDOM_STATE, n_estimators=100),
    'Local Outlier Factor': LocalOutlierFactor(contamination=contamination, n_neighbors=20, novelty=True),
    'One-Class SVM': OneClassSVM(nu=contamination, kernel='rbf', gamma='auto'),
    'Elliptic Envelope': EllipticEnvelope(contamination=contamination, random_state=RANDOM_STATE)
}

results = []

for method_name, method in methods.items():
    print(f"\n🔧 {method_name}...")
    start_time = time.time()
    
    method.fit(X_final_train)
    pred_test = method.predict(X_final_test)
    pred_binary = (pred_test == -1).astype(int)
    
    execution_time = time.time() - start_time
    
    accuracy = accuracy_score(y_test, pred_binary)
    precision = precision_score(y_test, pred_binary, zero_division=0)
    recall = recall_score(y_test, pred_binary, zero_division=0)
    f1 = f1_score(y_test, pred_binary, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_test, pred_binary).ravel()
    
    results.append({
        'Método': method_name, 'Recall': recall, 'F1-Score': f1,
        'Precision': precision, 'Accuracy': accuracy,
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'Tempo (s)': execution_time
    })
    
    print(f"   ✓ Recall: {recall*100:.2f}% ⭐ | F1: {f1*100:.2f}% | Precision: {precision*100:.2f}%")

# 6. ANÁLISE DOS RESULTADOS
print("\n[5/6] ANÁLISE DOS RESULTADOS")
print("-" * 80)

df_results = pd.DataFrame(results)
df_results = df_results.sort_values('Recall', ascending=False).reset_index(drop=True)
df_results['Rank'] = range(1, len(df_results) + 1)

print("\n🏆 RANKING POR RECALL:")
print(df_results[['Rank', 'Método', 'Recall', 'F1-Score', 'Precision']].to_string(index=False))

best_method = df_results.iloc[0]
print(f"\n🥇 MELHOR: {best_method['Método']} - Recall: {best_method['Recall']*100:.2f}%")

# 7. GRÁFICOS
print("\n[6/6] GERANDO GRÁFICOS")
print("-" * 80)

# Gráfico comparativo
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Recall (principal)
axes[0,0].bar(df_results['Método'], df_results['Recall'], color='darkred', alpha=0.7)
axes[0,0].set_title('Recall por Método (MÉTRICA PRINCIPAL)', fontweight='bold', fontsize=14)
axes[0,0].set_ylabel('Recall')
axes[0,0].tick_params(axis='x', rotation=45)
for i, v in enumerate(df_results['Recall']):
    axes[0,0].text(i, v + 0.01, f'{v*100:.1f}%', ha='center', fontweight='bold')

# F1-Score
axes[0,1].bar(df_results['Método'], df_results['F1-Score'], color='navy', alpha=0.7)
axes[0,1].set_title('F1-Score por Método', fontsize=14)
axes[0,1].set_ylabel('F1-Score')
axes[0,1].tick_params(axis='x', rotation=45)
for i, v in enumerate(df_results['F1-Score']):
    axes[0,1].text(i, v + 0.01, f'{v*100:.1f}%', ha='center')

# Precision
axes[1,0].bar(df_results['Método'], df_results['Precision'], color='darkgreen', alpha=0.7)
axes[1,0].set_title('Precision por Método', fontsize=14)
axes[1,0].set_ylabel('Precision')
axes[1,0].tick_params(axis='x', rotation=45)
for i, v in enumerate(df_results['Precision']):
    axes[1,0].text(i, v + 0.01, f'{v*100:.1f}%', ha='center')

# Tempo
axes[1,1].bar(df_results['Método'], df_results['Tempo (s)'], color='purple', alpha=0.7)
axes[1,1].set_title('Tempo de Execução', fontsize=14)
axes[1,1].set_ylabel('Tempo (s)')
axes[1,1].tick_params(axis='x', rotation=45)
for i, v in enumerate(df_results['Tempo (s)']):
    axes[1,1].text(i, v + 0.1, f'{v:.1f}s', ha='center')

plt.suptitle('NSL-KDD: Comparação de Métodos de Detecção de Outliers', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'nsl_kdd_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# Confusion Matrix do melhor
plt.figure(figsize=(8, 6))
best_result = df_results.iloc[0]
cm_data = [[best_result['TN'], best_result['FP']], 
           [best_result['FN'], best_result['TP']]]

sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Quantidade'},
            xticklabels=['Normal', 'U2R'], yticklabels=['Normal', 'U2R'])
plt.title(f'Confusion Matrix - {best_result["Método"]}\nRecall: {best_result["Recall"]*100:.1f}% | F1-Score: {best_result["F1-Score"]*100:.1f}%', 
          fontsize=14, fontweight='bold')
plt.ylabel('Verdadeiro', fontsize=12)
plt.xlabel('Predito', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'best_confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# Gráfico de comparação Recall vs F1-Score
plt.figure(figsize=(10, 6))
x = np.arange(len(df_results))
width = 0.35

plt.bar(x - width/2, df_results['Recall'], width, label='Recall ⭐', color='darkred', alpha=0.7)
plt.bar(x + width/2, df_results['F1-Score'], width, label='F1-Score', color='navy', alpha=0.7)

plt.xlabel('Métodos')
plt.ylabel('Score')
plt.title('Comparação: Recall vs F1-Score\n(Recall = Métrica Principal)', fontweight='bold', fontsize=14)
plt.xticks(x, df_results['Método'], rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# Adicionar valores nas barras
for i, (recall, f1) in enumerate(zip(df_results['Recall'], df_results['F1-Score'])):
    plt.text(i - width/2, recall + 0.01, f'{recall*100:.1f}%', ha='center', fontweight='bold')
    plt.text(i + width/2, f1 + 0.01, f'{f1*100:.1f}%', ha='center')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'recall_vs_f1_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# Salvar resultados
output_path = os.path.join(OUTPUT_DIR, 'nsl_kdd_results.csv')
df_results.to_csv(output_path, index=False)

print(f"✓ Gráficos salvos em: {OUTPUT_DIR}")
print(f"✓ Resultados salvos em: {output_path}")

# RESUMO FINAL
print("\n" + "="*80)
print("✅ ANÁLISE NSL-KDD CONCLUÍDA - VERSÃO FINAL")
print("="*80)
print(f"📋 ORIENTAÇÕES 04/11 IMPLEMENTADAS:")
print(f"   ✓ 2 classes: Normal vs U2R ({contamination*100:.2f}% outliers)")
print(f"   ✓ {len(df_features.columns)} features (após correlação)")
print(f"   ✓ Normalização: {'Aplicada' if use_normalization else 'Não aplicada'}")
print(f"   ✓ Recall como métrica principal")
print(f"   ✓ SEM PCA (conforme orientação)")

print(f"\n🏆 MELHOR RESULTADO:")
print(f"   Método: {best_method['Método']}")
print(f"   Recall: {best_method['Recall']*100:.2f}% ⭐")
print(f"   F1-Score: {best_method['F1-Score']*100:.2f}%")
print(f"   U2R detectados: {best_method['TP']}/{best_method['TP'] + best_method['FN']}")

print(f"\n📊 GRÁFICOS GERADOS:")
print(f"   • nsl_kdd_comparison.png - Comparação completa")
print(f"   • best_confusion_matrix.png - Matriz do melhor método")
print(f"   • recall_vs_f1_comparison.png - Recall vs F1-Score")
print(f"   • correlation_matrix.png - Análise de correlação")

print("\n" + "="*80)
