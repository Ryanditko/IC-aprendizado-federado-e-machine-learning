"""
GERADOR DE GRÁFICOS SIMPLES E DIRETO - NSL-KDD
==============================================

Script simplificado para gerar gráficos das métricas de cybersegurança
sem problemas de renderização.

Autor: Projeto de Iniciação Científica  
Data: Novembro 2025
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo para evitar problemas
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Configuração explícita do matplotlib
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.figsize': [10, 8],
    'savefig.dpi': 300,
    'savefig.format': 'png'
})

print("="*60)
print("GERADOR DE GRÁFICOS SIMPLES - NSL-KDD")  
print("="*60)

# Caminhos
DATA_DIR = '../../../data/nsl-kdd'
OUTPUT_DIR = '.'  # Salvar no diretório atual para garantir

try:
    print("\\n1️⃣ Carregando dados...")
    
    # Carregar apenas dados de treino para simplificar
    df = pd.read_csv(f'{DATA_DIR}/KDDTrain+_20Percent.txt', header=None)
    
    # Colunas essenciais
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
    
    df.columns = columns
    print(f"   ✅ {len(df):,} registros carregados")
    
    print("\\n2️⃣ Processando ataques...")
    
    # Classificar ataques U2R
    u2r_attacks = ['buffer_overflow', 'rootkit', 'loadmodule', 'perl', 
                   'httptunnel', 'ps', 'sqlattack', 'xterm']
    
    df['is_u2r_attack'] = df['attack_type'].isin(u2r_attacks).astype(int)
    
    u2r_count = df['is_u2r_attack'].sum()
    total_count = len(df)
    
    print(f"   ✅ Ataques U2R: {u2r_count} ({u2r_count/total_count*100:.2f}%)")
    print(f"   ✅ Normais/Outros: {total_count - u2r_count} ({(total_count - u2r_count)/total_count*100:.2f}%)")
    
    print("\\n3️⃣ Preparando features...")
    
    # Features numéricas simples
    numeric_cols = ['duration', 'src_bytes', 'dst_bytes', 'hot', 'num_failed_logins', 
                   'logged_in', 'count', 'srv_count']
    
    X = df[numeric_cols].copy()
    
    # Adicionar features categóricas encodadas
    le_protocol = LabelEncoder()
    le_service = LabelEncoder() 
    le_flag = LabelEncoder()
    
    X['protocol_encoded'] = le_protocol.fit_transform(df['protocol_type'].astype(str))
    X['service_encoded'] = le_service.fit_transform(df['service'].astype(str))
    X['flag_encoded'] = le_flag.fit_transform(df['flag'].astype(str))
    
    y = df['is_u2r_attack']
    
    print(f"   ✅ Features: {len(X.columns)}")
    
    # Balanceamento simples
    if u2r_count < 500:
        print("   ⚖️ Aplicando balanceamento...")
        u2r_indices = df[df['is_u2r_attack'] == 1].index.tolist()
        normal_indices = df[df['is_u2r_attack'] == 0].sample(n=len(u2r_indices)*4, random_state=42).index.tolist()
        
        selected_indices = u2r_indices + normal_indices
        X = X.loc[selected_indices]
        y = y.loc[selected_indices]
        
        print(f"   ✅ Amostra balanceada: {len(X)} registros")
    
    print("\\n4️⃣ Treinando modelo...")
    
    # Split simples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    print("   ✅ Modelo treinado")
    
    print("\\n5️⃣ Calculando métricas...")
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)  
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"   ✅ Accuracy: {accuracy:.3f}")
    print(f"   ✅ Precision: {precision:.3f}")
    print(f"   ✅ Recall: {recall:.3f}")
    print(f"   ✅ F1-Score: {f1:.3f}")
    
    print("\\n6️⃣ Gerando gráficos...")
    
    # GRÁFICO 1: MATRIZ DE CONFUSÃO
    print("   📊 Matriz de Confusão...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Heatmap simples
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    
    # Labels
    classes = ['Normal/Other', 'U2R Attack']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    # Título
    ax.set_title('Matriz de Confusão\\nDetecção de Ataques U2R', fontweight='bold', pad=20)
    ax.set_xlabel('Predição', fontweight='bold')
    ax.set_ylabel('Valor Real', fontweight='bold')
    
    # Valores nas células
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=16, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Número de Casos', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig('matriz_confusao_u2r.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✅ matriz_confusao_u2r.png salva")
    
    # GRÁFICO 2: MÉTRICAS
    print("   📊 Métricas de Performance...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy, precision, recall, f1]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Valores nas barras
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{value:.3f}\\n({value*100:.1f}%)',
               ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_title('Métricas de Performance\\nDetecção de Ataques U2R - Random Forest', 
                fontweight='bold', pad=20)
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    # Linha de referência
    ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='90% Target')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('metricas_performance_u2r.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✅ metricas_performance_u2r.png salva")
    
    # GRÁFICO 3: RESUMO
    print("   📊 Dashboard Resumo...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Matriz de confusão pequena
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
                    fontsize=12, fontweight='bold')
    
    # Métricas
    ax2.bar(metrics, values, color=colors, alpha=0.8)
    ax2.set_title('Métricas', fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis='y', alpha=0.3)
    
    for i, (metric, value) in enumerate(zip(metrics, values)):
        ax2.text(i, value + 0.02, f'{value:.2f}', ha='center', fontweight='bold')
    
    # Distribuição de ataques
    attack_counts = df['attack_type'].value_counts().head(8)
    ax3.barh(range(len(attack_counts)), attack_counts.values, color='skyblue')
    ax3.set_yticks(range(len(attack_counts)))
    ax3.set_yticklabels(attack_counts.index, fontsize=8)
    ax3.set_title('Top 8 Tipos de Ataque', fontweight='bold')
    ax3.set_xlabel('Frequência')
    
    # Estatísticas
    stats_text = f'''RESULTADOS

Dataset: NSL-KDD
Total: {total_count:,} registros
U2R Attacks: {u2r_count} ({u2r_count/total_count*100:.1f}%)

CONFUSION MATRIX:
TP: {tp} | TN: {tn}
FP: {fp} | FN: {fn}

METRICS:
Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)
Precision: {precision:.3f} ({precision*100:.1f}%)
Recall: {recall:.3f} ({recall*100:.1f}%)
F1-Score: {f1:.3f} ({f1*100:.1f}%)

INTERPRETATION:
• {precision*100:.1f}% of predicted attacks are real
• {recall*100:.1f}% of real attacks detected
• Only {fp} false alarms
• Only {fn} attacks missed'''
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Resumo', fontweight='bold')
    
    plt.suptitle('NSL-KDD CYBERSECURITY ANALYSIS\\nU2R Attack Detection', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('dashboard_u2r_completo.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✅ dashboard_u2r_completo.png salva")
    
    print("\\n" + "="*60)
    print("✅ TODOS OS GRÁFICOS GERADOS COM SUCESSO!")
    print("="*60)
    
    print(f"\\n📊 RESULTADOS FINAIS:")
    print(f"• Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"• Precision: {precision:.3f} ({precision*100:.1f}%)")  
    print(f"• Recall:    {recall:.3f} ({recall*100:.1f}%)")
    print(f"• F1-Score:  {f1:.3f} ({f1*100:.1f}%)")
    
    print(f"\\n📈 Confusion Matrix:")
    print(f"  True Positives:  {tp} (ataques detectados)")
    print(f"  True Negatives:  {tn} (normais identificados)")
    print(f"  False Positives: {fp} (falsos alarmes)")
    print(f"  False Negatives: {fn} (ataques perdidos)")
    
    print("\\n📁 Arquivos gerados:")
    print("  • matriz_confusao_u2r.png")
    print("  • metricas_performance_u2r.png") 
    print("  • dashboard_u2r_completo.png")
    
except Exception as e:
    print(f"\\n❌ ERRO: {e}")
    import traceback
    traceback.print_exc()
    
print("\\n🎯 Script finalizado!")
