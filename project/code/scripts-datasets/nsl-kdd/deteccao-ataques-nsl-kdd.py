"""
DETECÇÃO DE ATAQUES NSL-KDD - ANÁLISE DE SQL INJECTION
======================================================

Este script analisa o dataset NSL-KDD focando na detecção de ataques específicos,
incluindo métricas de performance e visualizações.

Dataset: NSL-KDD (Network Security Laboratory)
Foco: Detecção de ataques específicos vs tráfego normal

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
                           f1_score, roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

# Configuração de visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

print("="*80)
print("DETECÇÃO DE ATAQUES NSL-KDD - ANÁLISE ESPECÍFICA")
print("="*80)

# 1. CONFIGURAÇÕES
DATA_DIR = '../../../data/nsl-kdd'
OUTPUT_DIR = '../../../notebooks/nsl-kdd/output-images'
RESULTS_DIR = '../../../notebooks/nsl-kdd/results'

# Criar diretórios se não existirem
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 2. CARREGAR DADOS
print("\n[1/8] Carregando dataset NSL-KDD...")

# NSL-KDD geralmente tem os seguintes arquivos principais
try:
    # Tentar carregar diferentes variações do arquivo
    possible_files = ['KDDTrain+.txt', 'KDDTest+.txt', 'nsl-kdd.csv', 'train.csv', 'test.csv']
    
    train_file = None
    test_file = None
    
    for file in os.listdir(DATA_DIR):
        if 'train' in file.lower() and file.endswith(('.csv', '.txt')):
            train_file = file
        elif 'test' in file.lower() and file.endswith(('.csv', '.txt')):
            test_file = file
    
    if train_file:
        df_train = pd.read_csv(f'{DATA_DIR}/{train_file}', header=None)
        print(f"  ✓ Arquivo de treino: {train_file} ({len(df_train):,} registros)")
    
    if test_file:
        df_test = pd.read_csv(f'{DATA_DIR}/{test_file}', header=None)
        print(f"  ✓ Arquivo de teste: {test_file} ({len(df_test):,} registros)")
    
    # Se não encontrar arquivos separados, usar o primeiro arquivo disponível
    if not train_file and not test_file:
        files = [f for f in os.listdir(DATA_DIR) if f.endswith(('.csv', '.txt'))]
        if files:
            df_train = pd.read_csv(f'{DATA_DIR}/{files[0]}', header=None)
            print(f"  ✓ Arquivo único: {files[0]} ({len(df_train):,} registros)")
        else:
            raise FileNotFoundError("Nenhum arquivo de dados encontrado!")
    
except Exception as e:
    print(f"  ❌ Erro ao carregar: {e}")
    print("  💡 Execute primeiro o script de download do dataset")
    exit(1)

# 3. DEFINIR COLUNAS NSL-KDD
print("\n[2/8] Definindo estrutura do dataset...")

# Colunas padrão do NSL-KDD
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
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type'
]

# Se o dataset tem 42 colunas (incluindo difficulty), adicionar
if len(df_train.columns) >= 42:
    columns.append('difficulty')

df_train.columns = columns[:len(df_train.columns)]

if 'df_test' in locals():
    df_test.columns = columns[:len(df_test.columns)]
    df = pd.concat([df_train, df_test], ignore_index=True)
else:
    df = df_train.copy()

print(f"  ✓ Dataset combinado: {len(df):,} registros, {len(df.columns)} colunas")

# 4. ANÁLISE EXPLORATÓRIA
print("\n[3/8] Análise exploratória dos ataques...")

# Mapear tipos de ataque
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

# Aplicar mapeamento
df['attack_category'] = df['attack_type'].map(attack_mapping)
df['attack_category'] = df['attack_category'].fillna('other')

# Estatísticas dos ataques
attack_counts = df['attack_category'].value_counts()
print(f"  ✓ Distribuição dos ataques:")
for attack, count in attack_counts.items():
    percentage = count / len(df) * 100
    print(f"    • {attack}: {count:,} ({percentage:.2f}%)")

# 5. FOCAR EM UM TIPO ESPECÍFICO DE ATAQUE
print("\n[4/8] Preparando dados para detecção específica...")

# Escolher tipo de ataque específico para análise binária
TARGET_ATTACK = 'u2r'  # User-to-Root attacks (inclui SQL injection-like attacks)

# Criar target binário: ataque específico vs resto
df['is_target_attack'] = (df['attack_category'] == TARGET_ATTACK).astype(int)

target_count = df['is_target_attack'].sum()
normal_count = len(df) - target_count

print(f"  ✓ Foco do estudo: {TARGET_ATTACK.upper()} attacks")
print(f"  ✓ Ataques {TARGET_ATTACK}: {target_count:,} ({target_count/len(df)*100:.2f}%)")
print(f"  ✓ Outros/Normal: {normal_count:,} ({normal_count/len(df)*100:.2f}%)")

# 6. PREPARAÇÃO DOS DADOS
print("\n[5/8] Preparando features para machine learning...")

# Selecionar features numéricas
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features = [col for col in numeric_features if col not in ['is_target_attack', 'difficulty']]

# Encoder para features categóricas
categorical_features = ['protocol_type', 'service', 'flag']
encoders = {}

df_processed = df[numeric_features].copy()

for col in categorical_features:
    if col in df.columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

print(f"  ✓ Features numéricas: {len(numeric_features)}")
print(f"  ✓ Features categóricas: {len(categorical_features)}")
print(f"  ✓ Total de features: {len(df_processed.columns)}")

# Preparar X e y
X = df_processed
y = df['is_target_attack']

# Balanceamento da amostra para visualização (se muito desbalanceado)
if target_count < 1000:
    # Se há poucos ataques, usar todos + amostra equivalente de normais
    attack_indices = df[df['is_target_attack'] == 1].index
    normal_indices = df[df['is_target_attack'] == 0].sample(
        n=min(len(attack_indices) * 3, normal_count), random_state=42
    ).index
    
    selected_indices = list(attack_indices) + list(normal_indices)
    X = X.loc[selected_indices]
    y = y.loc[selected_indices]
    
    print(f"  ✓ Amostra balanceada: {len(selected_indices):,} registros")

# Split dos dados
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  ✓ Treino: {len(X_train):,} | Teste: {len(X_test):,}")

# 7. TREINAMENTO DE MODELOS
print("\n[6/8] Treinando modelos de machine learning...")

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM': SVC(random_state=42, probability=True)
}

results = {}

for name, model in models.items():
    print(f"  • Treinando {name}...")
    
    # Treinar modelo
    if 'SVM' in name:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except:
        auc = 0
    
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }
    
    print(f"    ✓ Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}")

# 8. VISUALIZAÇÕES
print("\n[7/8] Gerando visualizações...")

# Selecionar melhor modelo baseado no F1-score
best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
best_result = results[best_model_name]

print(f"  ✓ Melhor modelo: {best_model_name} (F1: {best_result['f1']:.3f})")

# Criar figura com subplots
fig = plt.figure(figsize=(20, 15))

# 1. Matriz de Confusão
ax1 = plt.subplot(2, 3, 1)
cm = confusion_matrix(y_test, best_result['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['Normal/Other', f'{TARGET_ATTACK.upper()} Attack'],
            yticklabels=['Normal/Other', f'{TARGET_ATTACK.upper()} Attack'])
ax1.set_title(f'Matriz de Confusão - {best_model_name}', fontsize=14, fontweight='bold')
ax1.set_xlabel('Predito', fontweight='bold')
ax1.set_ylabel('Real', fontweight='bold')

# 2. Métricas Comparativas
ax2 = plt.subplot(2, 3, 2)
metrics_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'Precision': [results[m]['precision'] for m in results.keys()],
    'Recall': [results[m]['recall'] for m in results.keys()],
    'F1-Score': [results[m]['f1'] for m in results.keys()]
})

x = np.arange(len(metrics_df))
width = 0.2

ax2.bar(x - 1.5*width, metrics_df['Accuracy'], width, label='Accuracy', alpha=0.8)
ax2.bar(x - 0.5*width, metrics_df['Precision'], width, label='Precision', alpha=0.8)
ax2.bar(x + 0.5*width, metrics_df['Recall'], width, label='Recall', alpha=0.8)
ax2.bar(x + 1.5*width, metrics_df['F1-Score'], width, label='F1-Score', alpha=0.8)

ax2.set_xlabel('Modelos', fontweight='bold')
ax2.set_ylabel('Score', fontweight='bold')
ax2.set_title('Comparação de Métricas', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(metrics_df['Model'], rotation=45)
ax2.legend()
ax2.grid(alpha=0.3)

# 3. Curva ROC
ax3 = plt.subplot(2, 3, 3)
for name, result in results.items():
    if result['auc'] > 0:
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        ax3.plot(fpr, tpr, label=f'{name} (AUC={result["auc"]:.3f})', linewidth=2)

ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax3.set_xlabel('Taxa de Falsos Positivos', fontweight='bold')
ax3.set_ylabel('Taxa de Verdadeiros Positivos', fontweight='bold')
ax3.set_title('Curvas ROC', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Distribuição dos Ataques
ax4 = plt.subplot(2, 3, 4)
attack_dist = df['attack_category'].value_counts()
colors = plt.cm.Set3(np.linspace(0, 1, len(attack_dist)))
wedges, texts, autotexts = ax4.pie(attack_dist.values, labels=attack_dist.index, 
                                   autopct='%1.1f%%', colors=colors, startangle=90)
ax4.set_title('Distribuição dos Tipos de Ataque', fontsize=14, fontweight='bold')

# 5. Feature Importance (se Random Forest for o melhor)
ax5 = plt.subplot(2, 3, 5)
if 'Random Forest' in best_model_name:
    feature_importance = best_result['model'].feature_importances_
    feature_names = X.columns
    
    # Top 10 features mais importantes
    indices = np.argsort(feature_importance)[-10:]
    
    ax5.barh(range(len(indices)), feature_importance[indices], color='skyblue', alpha=0.8)
    ax5.set_yticks(range(len(indices)))
    ax5.set_yticklabels([feature_names[i] for i in indices])
    ax5.set_xlabel('Importância', fontweight='bold')
    ax5.set_title('Top 10 Features Mais Importantes', fontsize=14, fontweight='bold')
    ax5.grid(alpha=0.3)
else:
    # Se não for Random Forest, mostrar estatísticas gerais
    stats_text = f"""
ESTATÍSTICAS GERAIS

Dataset: NSL-KDD
Total de registros: {len(df):,}
Total de features: {len(X.columns)}

Foco do estudo: {TARGET_ATTACK.upper()}
Ataques detectados: {target_count:,}
Taxa de ataque: {target_count/len(df)*100:.2f}%

Melhor modelo: {best_model_name}
Accuracy: {best_result['accuracy']:.3f}
Precision: {best_result['precision']:.3f}
Recall: {best_result['recall']:.3f}
F1-Score: {best_result['f1']:.3f}
"""
    ax5.text(0.1, 0.5, stats_text, transform=ax5.transAxes, fontsize=11,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax5.set_xticks([])
    ax5.set_yticks([])
    ax5.set_title('Resumo da Análise', fontsize=14, fontweight='bold')

# 6. Métricas Detalhadas
ax6 = plt.subplot(2, 3, 6)
tn, fp, fn, tp = cm.ravel()

metrics_detailed = {
    'True Positives': tp,
    'True Negatives': tn,
    'False Positives': fp,
    'False Negatives': fn,
    'Accuracy': f"{best_result['accuracy']:.3f}",
    'Precision': f"{best_result['precision']:.3f}",
    'Recall': f"{best_result['recall']:.3f}",
    'F1-Score': f"{best_result['f1']:.3f}"
}

metrics_text = '\n'.join([f'{k}: {v}' for k, v in metrics_detailed.items()])

ax6.text(0.1, 0.5, f'MÉTRICAS DETALHADAS\n\n{metrics_text}', 
         transform=ax6.transAxes, fontsize=12, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax6.set_xticks([])
ax6.set_yticks([])
ax6.set_title(f'Resultados - {best_model_name}', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/nsl_kdd_attack_detection_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Salvar métricas em arquivo
with open(f'{RESULTS_DIR}/attack_detection_results.txt', 'w') as f:
    f.write("RESULTADOS DA DETECÇÃO DE ATAQUES NSL-KDD\n")
    f.write("="*50 + "\n\n")
    f.write(f"Dataset: NSL-KDD\n")
    f.write(f"Foco: {TARGET_ATTACK.upper()} attacks vs others\n")
    f.write(f"Total de registros: {len(df):,}\n")
    f.write(f"Ataques {TARGET_ATTACK}: {target_count:,} ({target_count/len(df)*100:.2f}%)\n\n")
    
    f.write("RESULTADOS POR MODELO:\n")
    f.write("-" * 30 + "\n")
    for name, result in results.items():
        f.write(f"\n{name}:\n")
        f.write(f"  Accuracy:  {result['accuracy']:.4f}\n")
        f.write(f"  Precision: {result['precision']:.4f}\n")
        f.write(f"  Recall:    {result['recall']:.4f}\n")
        f.write(f"  F1-Score:  {result['f1']:.4f}\n")
        f.write(f"  AUC:       {result['auc']:.4f}\n")
    
    f.write(f"\nMELHOR MODELO: {best_model_name}\n")
    f.write("=" * 50 + "\n")

print(f"  ✓ Gráfico salvo: {OUTPUT_DIR}/nsl_kdd_attack_detection_analysis.png")
print(f"  ✓ Resultados salvos: {RESULTS_DIR}/attack_detection_results.txt")

# 9. RESUMO FINAL
print("\n[8/8] Resumo da análise...")
print("\n" + "="*80)
print("RESUMO DA DETECÇÃO DE ATAQUES NSL-KDD")
print("="*80)

print(f"\n📊 Dataset:")
print(f"  • Total de registros: {len(df):,}")
print(f"  • Foco do estudo: {TARGET_ATTACK.upper()} attacks")
print(f"  • Ataques detectados: {target_count:,} ({target_count/len(df)*100:.2f}%)")
print(f"  • Features utilizadas: {len(X.columns)}")

print(f"\n🎯 Melhor Modelo: {best_model_name}")
print(f"  • Accuracy: {best_result['accuracy']:.3f}")
print(f"  • Precision: {best_result['precision']:.3f}")
print(f"  • Recall: {best_result['recall']:.3f}")
print(f"  • F1-Score: {best_result['f1']:.3f}")

print(f"\n📈 Matriz de Confusão:")
print(f"  • True Positives: {tp}")
print(f"  • True Negatives: {tn}")
print(f"  • False Positives: {fp}")
print(f"  • False Negatives: {fn}")

print("\n" + "="*80)
print("✅ ANÁLISE DE CYBERSEGURANÇA CONCLUÍDA!")
print("="*80)

print(f"\n💡 INTERPRETAÇÃO:")
print(f"  • O modelo {best_model_name} conseguiu detectar {tp} ataques {TARGET_ATTACK}")
print(f"  • Precision de {best_result['precision']:.1%}: dos ataques preditos, {best_result['precision']:.1%} eram reais")
print(f"  • Recall de {best_result['recall']:.1%}: dos ataques reais, {best_result['recall']:.1%} foram detectados")
print(f"  • F1-Score de {best_result['f1']:.3f} indica o equilíbrio entre precision e recall")

print(f"\n📁 Arquivos gerados:")
print(f"  • Gráfico: {OUTPUT_DIR}/nsl_kdd_attack_detection_analysis.png")
print(f"  • Resultados: {RESULTS_DIR}/attack_detection_results.txt")
print("\n")
