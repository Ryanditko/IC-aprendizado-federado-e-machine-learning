"""
Gráficos Específicos - NSL-KDD
==============================
Gera matriz de confusão e gráfico de métricas (acurácia, precisão, recall)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configurações
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12

OUTPUT_DIR = 'output-nsl-kdd'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dados dos resultados (do script anterior)
resultados = {
    'Método': ['One-Class SVM', 'Local Outlier Factor', 'Isolation Forest', 'Elliptic Envelope'],
    'Accuracy': [0.9958, 0.9996, 0.9985, 0.9985],
    'Precision': [0.00196, 0.08824, 0.0, 0.0],
    'Recall': [1.0, 0.08333, 0.0, 0.0],
    'F1-Score': [0.00392, 0.08571, 0.0, 0.0],
    'TP': [36, 3, 0, 0],
    'FP': [18325, 31, 0, 0],
    'TN': [4791, 23085, 23116, 23116],
    'FN': [0, 33, 36, 36]
}

df_results = pd.DataFrame(resultados)

print("="*60)
print("GERANDO GRÁFICOS ESPECÍFICOS - NSL-KDD")
print("="*60)

# 1. GRÁFICO DE MÉTRICAS (Acurácia, Precisão, Recall)
print("\n1. Gráfico de Métricas...")

fig, ax = plt.subplots(figsize=(12, 8))

x = np.arange(len(df_results))
width = 0.25

# Barras para cada métrica
bars1 = ax.bar(x - width, df_results['Accuracy'], width, label='Acurácia', color='#2E86C1', alpha=0.8)
bars2 = ax.bar(x, df_results['Precision'], width, label='Precisão', color='#28B463', alpha=0.8)
bars3 = ax.bar(x + width, df_results['Recall'], width, label='Recall ⭐', color='#E74C3C', alpha=0.8)

# Configurações do gráfico
ax.set_xlabel('Métodos de Detecção', fontweight='bold', fontsize=14)
ax.set_ylabel('Score', fontweight='bold', fontsize=14)
ax.set_title('Comparação de Métricas: Acurácia, Precisão e Recall\nDataset NSL-KDD (Normal vs U2R)', 
             fontweight='bold', fontsize=16, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(df_results['Método'], rotation=45, ha='right')
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

# Adicionar valores nas barras
def add_value_labels(bars, values, format_str='{:.1%}'):
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                format_str.format(value), ha='center', va='bottom', fontweight='bold')

add_value_labels(bars1, df_results['Accuracy'])
add_value_labels(bars2, df_results['Precision'])
add_value_labels(bars3, df_results['Recall'])

# Destacar o melhor recall
max_recall_idx = df_results['Recall'].idxmax()
ax.annotate('MELHOR RECALL\n100%', 
            xy=(max_recall_idx + width, df_results['Recall'][max_recall_idx]), 
            xytext=(max_recall_idx + width + 0.5, 0.8),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=12, fontweight='bold', color='red',
            ha='center')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'metricas_acuracia_precisao_recall.png'), 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 2. MATRIZ DE CONFUSÃO DO MELHOR MÉTODO (One-Class SVM)
print("2. Matriz de Confusão...")

fig, ax = plt.subplots(figsize=(10, 8))

# Dados do One-Class SVM (melhor recall)
best_method = df_results.iloc[0]
cm_data = np.array([[best_method['TN'], best_method['FP']], 
                    [best_method['FN'], best_method['TP']]])

# Criar heatmap
sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'U2R Attack'], 
            yticklabels=['Normal', 'U2R Attack'],
            cbar_kws={'label': 'Quantidade de Casos'},
            annot_kws={'size': 16, 'weight': 'bold'})

plt.title(f'Matriz de Confusão - {best_method["Método"]}\n' +
          f'Recall: {best_method["Recall"]*100:.1f}% | ' +
          f'Precision: {best_method["Precision"]*100:.2f}% | ' +
          f'F1-Score: {best_method["F1-Score"]*100:.2f}%', 
          fontweight='bold', fontsize=16, pad=20)

plt.ylabel('Classe Verdadeira', fontweight='bold', fontsize=14)
plt.xlabel('Classe Predita', fontweight='bold', fontsize=14)

# Adicionar anotações explicativas
ax.text(0.5, -0.15, 'TN = True Negative (Normal correto)\nFP = False Positive (Normal como U2R)\n' +
                    'FN = False Negative (U2R como Normal)\nTP = True Positive (U2R correto)', 
        transform=ax.transAxes, ha='center', fontsize=10, 
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'matriz_confusao_detalhada.png'), 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 3. GRÁFICO COMPARATIVO RECALL vs OUTROS
print("3. Gráfico Recall vs Outras Métricas...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Subplot 1: Recall isolado (métrica principal)
colors = ['#E74C3C' if recall == 1.0 else '#95A5A6' for recall in df_results['Recall']]
bars = ax1.bar(df_results['Método'], df_results['Recall'], color=colors, alpha=0.8)

ax1.set_title('Recall por Método\n(Métrica Principal)', fontweight='bold', fontsize=14)
ax1.set_ylabel('Recall', fontweight='bold')
ax1.set_ylim(0, 1.1)
ax1.grid(True, alpha=0.3, axis='y')

# Adicionar valores
for bar, value in zip(bars, df_results['Recall']):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{value*100:.1f}%', ha='center', va='bottom', fontweight='bold')

# Destacar o melhor
ax1.text(0, 1.05, '⭐ MELHOR', ha='center', fontweight='bold', color='red', fontsize=12)

# Subplot 2: Comparação F1-Score vs Recall
x = np.arange(len(df_results))
width = 0.35

bars1 = ax2.bar(x - width/2, df_results['Recall'], width, label='Recall ⭐', color='#E74C3C', alpha=0.8)
bars2 = ax2.bar(x + width/2, df_results['F1-Score'], width, label='F1-Score', color='#3498DB', alpha=0.8)

ax2.set_title('Recall vs F1-Score\n(Trade-off de Métricas)', fontweight='bold', fontsize=14)
ax2.set_ylabel('Score')
ax2.set_xticks(x)
ax2.set_xticklabels(df_results['Método'], rotation=45, ha='right')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'recall_comparativo.png'), 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 4. RESUMO EXECUTIVO
print("4. Resumo Visual...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Quadrante 1: Recall
ax1.bar(['One-Class SVM', 'LOF', 'Isolation F.', 'Elliptic E.'], 
        df_results['Recall'], color=['#E74C3C', '#F39C12', '#95A5A6', '#95A5A6'])
ax1.set_title('Recall (Sensibilidade)', fontweight='bold')
ax1.set_ylabel('Score')
ax1.grid(True, alpha=0.3)

# Quadrante 2: Precision
ax2.bar(['One-Class SVM', 'LOF', 'Isolation F.', 'Elliptic E.'], 
        df_results['Precision'], color=['#E74C3C', '#F39C12', '#95A5A6', '#95A5A6'])
ax2.set_title('Precision (Precisão)', fontweight='bold')
ax2.set_ylabel('Score')
ax2.grid(True, alpha=0.3)

# Quadrante 3: Accuracy
ax3.bar(['One-Class SVM', 'LOF', 'Isolation F.', 'Elliptic E.'], 
        df_results['Accuracy'], color=['#E74C3C', '#F39C12', '#95A5A6', '#95A5A6'])
ax3.set_title('Accuracy (Acurácia)', fontweight='bold')
ax3.set_ylabel('Score')
ax3.grid(True, alpha=0.3)

# Quadrante 4: F1-Score
ax4.bar(['One-Class SVM', 'LOF', 'Isolation F.', 'Elliptic E.'], 
        df_results['F1-Score'], color=['#E74C3C', '#F39C12', '#95A5A6', '#95A5A6'])
ax4.set_title('F1-Score (Harmônica)', fontweight='bold')
ax4.set_ylabel('Score')
ax4.grid(True, alpha=0.3)

plt.suptitle('NSL-KDD: Análise Completa de Métricas\nNormal vs U2R Attacks', 
             fontweight='bold', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'resumo_completo_metricas.png'), 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("\n" + "="*60)
print("✅ GRÁFICOS GERADOS COM SUCESSO!")
print("="*60)
print("\n📊 ARQUIVOS CRIADOS:")
print("   • metricas_acuracia_precisao_recall.png")
print("   • matriz_confusao_detalhada.png") 
print("   • recall_comparativo.png")
print("   • resumo_completo_metricas.png")

print("\n🎯 DESTAQUES:")
print("   • One-Class SVM: 100% Recall (detectou todos os U2R)")
print("   • Trade-off: Alta sensibilidade, baixa precisão")
print("   • Adequado para cybersecurity (não perder ataques)")

print(f"\n📁 Localização: {os.path.abspath(OUTPUT_DIR)}")
