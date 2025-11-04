"""
GRÁFICOS CORRIGIDOS - NSL-KDD CYBERSECURITY
===========================================

Script para gerar gráficos limpos e legíveis das métricas de cybersegurança.
Corrige problemas de sobreposição de texto e formatação.

Autor: Projeto de Iniciação Científica
Data: Novembro 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, classification_report, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score)
import warnings
warnings.filterwarnings('ignore')

# Configuração para corrigir problemas de renderização
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'text.usetex': False,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

plt.style.use('default')  # Usar estilo padrão para evitar conflitos

print("="*80)
print("GERANDO GRÁFICOS CORRIGIDOS - NSL-KDD")
print("="*80)

# Carregar dados rapidamente (versão otimizada)
DATA_DIR = '../../../data/nsl-kdd'
OUTPUT_DIR = '../../../notebooks/nsl-kdd/output-images'

print("📊 Carregando e processando dados...")

# Carregar apenas uma amostra para análise rápida
df_train = pd.read_csv(f'{DATA_DIR}/KDDTrain+_20Percent.txt', header=None)
df_test = pd.read_csv(f'{DATA_DIR}/KDDTest-21.txt', header=None)

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
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty'
]

df_train.columns = columns
df_test.columns = columns
df = pd.concat([df_train, df_test], ignore_index=True)

# Mapear ataques
attack_mapping = {
    'normal': 'normal',
    'neptune': 'dos', 'smurf': 'dos', 'pod': 'dos', 'teardrop': 'dos',
    'land': 'dos', 'back': 'dos', 'apache2': 'dos', 'processtable': 'dos',
    'buffer_overflow': 'u2r', 'rootkit': 'u2r', 'loadmodule': 'u2r',
    'perl': 'u2r', 'httptunnel': 'u2r', 'ps': 'u2r', 'sqlattack': 'u2r',
    'xterm': 'u2r'
}

df['attack_category'] = df['attack_type'].map(attack_mapping)
df['attack_category'] = df['attack_category'].fillna('other')
df['is_target_attack'] = (df['attack_category'] == 'u2r').astype(int)

target_count = df['is_target_attack'].sum()

# Preparar features simplificado
numeric_features = ['duration', 'src_bytes', 'dst_bytes', 'hot', 'num_failed_logins',
                   'logged_in', 'num_compromised', 'count', 'srv_count']
                   
df_processed = df[numeric_features].copy()

# Encoder simples
le_protocol = LabelEncoder()
le_service = LabelEncoder()
le_flag = LabelEncoder()

df_processed['protocol_type'] = le_protocol.fit_transform(df['protocol_type'].astype(str))
df_processed['service'] = le_service.fit_transform(df['service'].astype(str))
df_processed['flag'] = le_flag.fit_transform(df['flag'].astype(str))

X = df_processed
y = df['is_target_attack']

# Balanceamento
attack_indices = df[df['is_target_attack'] == 1].index
normal_indices = df[df['is_target_attack'] == 0].sample(n=len(attack_indices) * 3, random_state=42).index
selected_indices = list(attack_indices) + list(normal_indices)

X = X.loc[selected_indices]
y = y.loc[selected_indices]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Treinar Random Forest
print("🤖 Treinando Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Calcular métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"✅ Métricas: Acc={accuracy:.3f}, Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}")

# GRÁFICO 1: MATRIZ DE CONFUSÃO LIMPA
print("📊 Gerando Matriz de Confusão...")

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
plt.clf()  # Limpar figura

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Criar heatmap simples
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')

# Adicionar colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Número de Casos', rotation=270, labelpad=20)

# Labels
tick_marks = np.arange(2)
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(['Normal/Other', 'U2R Attack'])
ax.set_yticklabels(['Normal/Other', 'U2R Attack'])

# Título e labels
ax.set_title('MATRIZ DE CONFUSÃO\\nDetecção de Ataques U2R - Random Forest', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Predição do Modelo', fontsize=14, fontweight='bold')
ax.set_ylabel('Valor Real', fontsize=14, fontweight='bold')

# Adicionar valores nas células
thresh = cm.max() / 2.
for i in range(2):
    for j in range(2):
        value = cm[i, j]
        ax.text(j, i, f'{value}', ha="center", va="center",
                color="white" if value > thresh else "black",
                fontsize=20, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/matriz_confusao_limpa.png', dpi=300, bbox_inches='tight')
plt.close()

# GRÁFICO 2: MÉTRICAS DE PERFORMANCE
print("📊 Gerando Gráfico de Métricas...")

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
plt.clf()

metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_values = [accuracy, precision, recall, f1]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

bars = ax.bar(metrics_names, metrics_values, color=colors, alpha=0.8, 
              edgecolor='black', linewidth=2)

# Adicionar valores nas barras
for bar, value in zip(bars, metrics_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{value:.3f}\\n({value*100:.1f}%)',
             ha='center', va='bottom', fontweight='bold', fontsize=12)

ax.set_title('MÉTRICAS DE PERFORMANCE\\nDetecção de Ataques U2R - NSL-KDD Dataset', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('Score', fontweight='bold', fontsize=14)
ax.set_ylim(0, 1.1)
ax.grid(alpha=0.3, axis='y')

# Adicionar linha de referência
ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='90% Benchmark')
ax.legend()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/metricas_performance_limpa.png', dpi=300, bbox_inches='tight')
plt.close()

# GRÁFICO 3: DASHBOARD SIMPLIFICADO
print("📊 Gerando Dashboard Simplificado...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
plt.clf()

# Subplot 1: Matriz de Confusão
im1 = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
ax1.set_title('Matriz de Confusão', fontweight='bold')
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(['Normal', 'U2R'])
ax1.set_yticklabels(['Normal', 'U2R'])

for i in range(2):
    for j in range(2):
        ax1.text(j, i, cm[i, j], ha="center", va="center", 
                color="white" if cm[i, j] > cm.max()/2 else "black",
                fontsize=14, fontweight='bold')

# Subplot 2: Métricas
bars2 = ax2.bar(metrics_names, metrics_values, color=colors, alpha=0.8)
ax2.set_title('Métricas de Performance', fontweight='bold')
ax2.set_ylim(0, 1.1)
ax2.grid(alpha=0.3, axis='y')

for bar, value in zip(bars2, metrics_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

# Subplot 3: Distribuição de Ataques
attack_dist = df['attack_category'].value_counts()
ax3.pie(attack_dist.values, labels=attack_dist.index, autopct='%1.1f%%', startangle=90)
ax3.set_title('Distribuição dos Tipos de Ataque', fontweight='bold')

# Subplot 4: Estatísticas
stats_text = f'''RESULTADOS FINAIS

Dataset: NSL-KDD
Total: {len(df):,} registros
Ataques U2R: {target_count} ({target_count/len(df)*100:.2f}%)

MATRIZ DE CONFUSÃO:
• True Positives: {tp}
• True Negatives: {tn} 
• False Positives: {fp}
• False Negatives: {fn}

MÉTRICAS:
• Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)
• Precision: {precision:.3f} ({precision*100:.1f}%)
• Recall: {recall:.3f} ({recall*100:.1f}%)
• F1-Score: {f1:.3f} ({f1*100:.1f}%)'''

ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')
ax4.set_title('Resumo dos Resultados', fontweight='bold')

plt.suptitle('ANÁLISE DE CYBERSEGURANÇA - NSL-KDD DATASET', 
             fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig(f'{OUTPUT_DIR}/dashboard_limpo.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Gráficos corrigidos salvos:")
print(f"  • {OUTPUT_DIR}/matriz_confusao_limpa.png")
print(f"  • {OUTPUT_DIR}/metricas_performance_limpa.png") 
print(f"  • {OUTPUT_DIR}/dashboard_limpo.png")

print("\\n" + "="*80)
print("✅ GRÁFICOS LIMPOS E LEGÍVEIS GERADOS!")
print("="*80)

print(f"\\n📊 RESULTADOS FINAIS:")
print(f"• Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
print(f"• Precision: {precision:.3f} ({precision*100:.1f}%)")
print(f"• Recall:    {recall:.3f} ({recall*100:.1f}%)")
print(f"• F1-Score:  {f1:.3f} ({f1*100:.1f}%)")

print(f"\\n📈 Matriz de Confusão:")
print(f"  TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")

print("\\n💡 Os novos gráficos devem estar limpos e legíveis!")
print("\\n")
