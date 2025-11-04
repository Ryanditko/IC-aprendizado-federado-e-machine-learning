"""
GRÁFICOS ESPECÍFICOS - MÉTRICAS NSL-KDD
=======================================

Script para gerar gráficos específicos das métricas solicitadas:
- Matriz de Confusão
- Acurácia, Precisão e Recall

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

# Configuração de visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("GRÁFICOS ESPECÍFICOS - MÉTRICAS NSL-KDD")
print("="*80)

# Carregar e processar dados (versão simplificada)
DATA_DIR = '../../../data/nsl-kdd'
OUTPUT_DIR = '../../../notebooks/nsl-kdd/output-images'

# Carregar dados
df_train = pd.read_csv(f'{DATA_DIR}/KDDTrain+_20Percent.txt', header=None)
df_test = pd.read_csv(f'{DATA_DIR}/KDDTest-21.txt', header=None)

# Definir colunas
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

print(f"✅ Dataset carregado: {len(df):,} registros")

# Mapear ataques
attack_mapping = {
    'normal': 'normal',
    'neptune': 'dos', 'smurf': 'dos', 'pod': 'dos', 'teardrop': 'dos',
    'land': 'dos', 'back': 'dos', 'apache2': 'dos', 'processtable': 'dos',
    'mailbomb': 'dos', 'udpstorm': 'dos',
    'ipsweep': 'probe', 'portsweep': 'probe', 'nmap': 'probe', 'satan': 'probe',
    'saint': 'probe', 'mscan': 'probe',
    'warezclient': 'r2l', 'warezmaster': 'r2l', 'ftpwrite': 'r2l',
    'guess_passwd': 'r2l', 'imap': 'r2l', 'multihop': 'r2l', 'phf': 'r2l',
    'spy': 'r2l', 'sendmail': 'r2l', 'named': 'r2l', 'snmpgetattack': 'r2l',
    'snmpguess': 'r2l', 'xlock': 'r2l', 'xsnoop': 'r2l', 'worm': 'r2l',
    'buffer_overflow': 'u2r', 'rootkit': 'u2r', 'loadmodule': 'u2r',
    'perl': 'u2r', 'httptunnel': 'u2r', 'ps': 'u2r', 'sqlattack': 'u2r',
    'xterm': 'u2r'
}

df['attack_category'] = df['attack_type'].map(attack_mapping)
df['attack_category'] = df['attack_category'].fillna('other')

# Foco em U2R attacks
TARGET_ATTACK = 'u2r'
df['is_target_attack'] = (df['attack_category'] == TARGET_ATTACK).astype(int)

target_count = df['is_target_attack'].sum()
print(f"✅ Ataques {TARGET_ATTACK.upper()}: {target_count} ({target_count/len(df)*100:.2f}%)")

# Preparar features
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features = [col for col in numeric_features if col not in ['is_target_attack', 'difficulty']]

categorical_features = ['protocol_type', 'service', 'flag']
encoders = {}

df_processed = df[numeric_features].copy()

for col in categorical_features:
    if col in df.columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

X = df_processed
y = df['is_target_attack']

# Balanceamento
if target_count < 1000:
    attack_indices = df[df['is_target_attack'] == 1].index
    normal_indices = df[df['is_target_attack'] == 0].sample(
        n=min(len(attack_indices) * 3, len(df) - target_count), random_state=42
    ).index
    
    selected_indices = list(attack_indices) + list(normal_indices)
    X = X.loc[selected_indices]
    y = y.loc[selected_indices]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Treinar modelos
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM': SVC(random_state=42, probability=True)
}

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}

for name, model in models.items():
    print(f"🔄 Treinando {name}...")
    
    if 'SVM' in name:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    results[name] = {
        'y_pred': y_pred,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Selecionar melhor modelo
best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
best_result = results[best_model_name]

print(f"🏆 Melhor modelo: {best_model_name}")

# GERAR GRÁFICOS ESPECÍFICOS
print("📊 Gerando gráficos específicos...")

# Criar figura com 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'ANÁLISE DE CYBERSEGURANÇA - DETECÇÃO DE ATAQUES U2R\\nDataset NSL-KDD - Melhor Modelo: {best_model_name}', 
             fontsize=16, fontweight='bold', y=0.98)

# 1. MATRIZ DE CONFUSÃO (Principal)
ax1 = axes[0, 0]
cm = confusion_matrix(y_test, best_result['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar_kws={'shrink': 0.8},
            xticklabels=['Normal/Other', 'U2R Attack'],
            yticklabels=['Normal/Other', 'U2R Attack'])
ax1.set_title('MATRIZ DE CONFUSÃO', fontsize=14, fontweight='bold', pad=20)
ax1.set_xlabel('Predição', fontweight='bold', fontsize=12)
ax1.set_ylabel('Valor Real', fontweight='bold', fontsize=12)

# Adicionar percentuais na matriz
tn, fp, fn, tp = cm.ravel()
total = tn + fp + fn + tp
annotations = [
    [f'{tn}\\n({tn/total*100:.1f}%)', f'{fp}\\n({fp/total*100:.1f}%)'],
    [f'{fn}\\n({fn/total*100:.1f}%)', f'{tp}\\n({tp/total*100:.1f}%)']
]

for i in range(2):
    for j in range(2):
        ax1.text(j+0.5, i+0.5, annotations[i][j], ha='center', va='center',
                fontsize=11, fontweight='bold', color='white' if cm[i,j] > cm.max()/2 else 'black')

# 2. COMPARAÇÃO DE MÉTRICAS
ax2 = axes[0, 1]
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_values = [
    best_result['accuracy'],
    best_result['precision'], 
    best_result['recall'],
    best_result['f1']
]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = ax2.bar(metrics_names, metrics_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

# Adicionar valores nas barras
for bar, value in zip(bars, metrics_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{value:.3f}\\n({value*100:.1f}%)',
             ha='center', va='bottom', fontweight='bold', fontsize=11)

ax2.set_title(f'MÉTRICAS DE PERFORMANCE\\n{best_model_name}', fontsize=14, fontweight='bold', pad=20)
ax2.set_ylabel('Score', fontweight='bold', fontsize=12)
ax2.set_ylim(0, 1.1)
ax2.grid(alpha=0.3, axis='y')

# 3. COMPARAÇÃO ENTRE TODOS OS MODELOS
ax3 = axes[1, 0]
model_names = list(results.keys())
x_pos = np.arange(len(model_names))
width = 0.2

metrics_comparison = {
    'Accuracy': [results[m]['accuracy'] for m in model_names],
    'Precision': [results[m]['precision'] for m in model_names],
    'Recall': [results[m]['recall'] for m in model_names],
    'F1-Score': [results[m]['f1'] for m in model_names]
}

colors_comp = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
for i, (metric, values) in enumerate(metrics_comparison.items()):
    bars = ax3.bar(x_pos + i*width, values, width, label=metric, 
                   color=colors_comp[i], alpha=0.8, edgecolor='black', linewidth=0.5)

ax3.set_xlabel('Modelos', fontweight='bold', fontsize=12)
ax3.set_ylabel('Score', fontweight='bold', fontsize=12)
ax3.set_title('COMPARAÇÃO ENTRE MODELOS', fontsize=14, fontweight='bold', pad=20)
ax3.set_xticks(x_pos + width * 1.5)
ax3.set_xticklabels(model_names, rotation=15)
ax3.legend(loc='upper right')
ax3.grid(alpha=0.3, axis='y')
ax3.set_ylim(0, 1.1)

# 4. ESTATÍSTICAS DETALHADAS
ax4 = axes[1, 1]
ax4.axis('off')

# Calcular estatísticas adicionais
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # = recall
npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate

stats_text = f'''ESTATÍSTICAS DETALHADAS

📊 DATASET:
• Total de registros: {len(df):,}
• Ataques U2R: {target_count:,} ({target_count/len(df)*100:.2f}%)
• Features utilizadas: {len(X.columns)}

🎯 MATRIZ DE CONFUSÃO:
• True Positives (TP): {tp}
• True Negatives (TN): {tn}
• False Positives (FP): {fp}
• False Negatives (FN): {fn}

📈 MÉTRICAS PRINCIPAIS:
• Accuracy: {best_result['accuracy']:.3f} ({best_result['accuracy']*100:.1f}%)
• Precision: {best_result['precision']:.3f} ({best_result['precision']*100:.1f}%)
• Recall: {best_result['recall']:.3f} ({best_result['recall']*100:.1f}%)
• F1-Score: {best_result['f1']:.3f} ({best_result['f1']*100:.1f}%)

📊 MÉTRICAS ADICIONAIS:
• Specificity: {specificity:.3f} ({specificity*100:.1f}%)
• NPV: {npv:.3f} ({npv*100:.1f}%)
• FPR: {fpr:.3f} ({fpr*100:.1f}%)
• FNR: {fnr:.3f} ({fnr*100:.1f}%)

💡 INTERPRETAÇÃO:
• {best_result['precision']*100:.1f}% dos ataques preditos são reais
• {best_result['recall']*100:.1f}% dos ataques reais foram detectados
• {(1-fpr)*100:.1f}% do tráfego normal foi corretamente identificado'''

ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig(f'{OUTPUT_DIR}/metricas_especificas_nsl_kdd.png', dpi=300, bbox_inches='tight')
plt.close()

# GRÁFICO ADICIONAL: MATRIZ DE CONFUSÃO ISOLADA (GRANDE)
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Matriz de confusão grande e detalhada
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax, cbar_kws={'shrink': 0.8},
            xticklabels=['Normal/Other Traffic', 'U2R Attack'],
            yticklabels=['Normal/Other Traffic', 'U2R Attack'])

# Adicionar anotações customizadas
labels = [['True Negatives\\n(TN)', 'False Positives\\n(FP)'],
          ['False Negatives\\n(FN)', 'True Positives\\n(TP)']]

values = [[tn, fp], [fn, tp]]
percentages = [[f'{tn/total*100:.1f}%', f'{fp/total*100:.1f}%'],
               [f'{fn/total*100:.1f}%', f'{tp/total*100:.1f}%']]

for i in range(2):
    for j in range(2):
        text = f'{labels[i][j]}\\n\\n{values[i][j]}\\n({percentages[i][j]})'
        ax.text(j+0.5, i+0.5, text, ha='center', va='center',
                fontsize=14, fontweight='bold', 
                color='white' if cm[i,j] > cm.max()/2 else 'black')

ax.set_title(f'MATRIZ DE CONFUSÃO DETALHADA\\nDetecção de Ataques U2R - {best_model_name}', 
             fontsize=16, fontweight='bold', pad=30)
ax.set_xlabel('PREDIÇÃO DO MODELO', fontweight='bold', fontsize=14)
ax.set_ylabel('VALOR REAL', fontweight='bold', fontsize=14)

# Adicionar legenda explicativa
legend_text = f'''
INTERPRETAÇÃO:
• True Positives (TP): Ataques corretamente detectados
• True Negatives (TN): Tráfego normal corretamente identificado  
• False Positives (FP): Falsos alarmes (normal classificado como ataque)
• False Negatives (FN): Ataques não detectados (CRÍTICO para segurança!)

MÉTRICAS CALCULADAS:
• Accuracy = (TP + TN) / Total = {best_result['accuracy']:.3f}
• Precision = TP / (TP + FP) = {best_result['precision']:.3f}
• Recall = TP / (TP + FN) = {best_result['recall']:.3f}
'''

ax.text(1.05, 0.5, legend_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='center', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/matriz_confusao_detalhada_nsl_kdd.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Gráficos específicos salvos:")
print(f"  • {OUTPUT_DIR}/metricas_especificas_nsl_kdd.png")
print(f"  • {OUTPUT_DIR}/matriz_confusao_detalhada_nsl_kdd.png")

print("\\n" + "="*80)
print("✅ GRÁFICOS ESPECÍFICOS GERADOS COM SUCESSO!")
print("="*80)

print(f"\\n🎯 RESUMO DOS RESULTADOS ({best_model_name}):")
print("="*50)
print(f"📊 Accuracy:  {best_result['accuracy']:.3f} ({best_result['accuracy']*100:.1f}%)")
print(f"🎯 Precision: {best_result['precision']:.3f} ({best_result['precision']*100:.1f}%)")
print(f"🔍 Recall:    {best_result['recall']:.3f} ({best_result['recall']*100:.1f}%)")
print(f"⚖️  F1-Score:  {best_result['f1']:.3f} ({best_result['f1']*100:.1f}%)")

print(f"\\n📈 Matriz de Confusão:")
print(f"  • True Positives: {tp} ataques detectados")
print(f"  • True Negatives: {tn} tráfego normal identificado")
print(f"  • False Positives: {fp} falsos alarmes")
print(f"  • False Negatives: {fn} ataques perdidos")

print("\\n💡 SIGNIFICADO PARA CYBERSEGURANÇA:")
print(f"  • Dos ataques preditos pelo modelo, {best_result['precision']*100:.1f}% são realmente maliciosos")
print(f"  • O modelo detecta {best_result['recall']*100:.1f}% de todos os ataques reais")
print(f"  • Apenas {fp} falsos alarmes em {tn+fp} casos normais ({fp/(tn+fp)*100:.2f}%)")
print(f"  • Apenas {fn} ataques não detectados em {tp+fn} casos maliciosos ({fn/(tp+fn)*100:.2f}%)")
print("\\n")
