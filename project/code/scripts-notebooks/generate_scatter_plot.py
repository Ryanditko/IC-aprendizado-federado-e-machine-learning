"""
GERADOR DE GRÁFICO DE DISPERSÃO - OUTLIERS MALICIOSOS
=======================================================

Este script gera uma visualização 2D dos dados do dataset de cybersecurity,
destacando visualmente os dados maliciosos vs normais e os outliers detectados.

Autor: Projeto de Iniciação Científica
Data: Outubro 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuração
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("GERAÇÃO DE GRÁFICO DE DISPERSÃO - OUTLIERS MALICIOSOS")
print("="*80)

# 1. CARREGAR DADOS
print("\n[1/6] Carregando dataset...")
DATA_DIR = '../../data/cyber-outlier-detection'
OUTPUT_DIR = '../../notebooks/cyber-outlier-detection/output-images'

df = pd.read_csv(f'{DATA_DIR}/cyber-threat-intelligence_all.csv')
print(f"  ✓ {len(df):,} registros carregados")

# 2. FILTRAR DADOS COM LABELS
print("\n[2/6] Filtrando dados com labels...")
df_labeled = df[df['label'].notna() & df['text'].notna()].copy()
print(f"  ✓ {len(df_labeled):,} registros com labels")

# 3. CLASSIFICAR AMEAÇAS
print("\n[3/6] Classificando ameaças...")
threat_labels = ['malware', 'attack-pattern', 'threat-actor', 'vulnerability', 'tools']
df_labeled['is_threat'] = df_labeled['label'].apply(
    lambda x: 1 if x in threat_labels else 0
)

threat_count = (df_labeled['is_threat'] == 1).sum()
normal_count = (df_labeled['is_threat'] == 0).sum()

print(f"  ✓ Ameaças: {threat_count:,} ({threat_count/len(df_labeled)*100:.1f}%)")
print(f"  ✓ Normais: {normal_count:,} ({normal_count/len(df_labeled)*100:.1f}%)")

# 4. PROCESSAR TEXTO (TF-IDF + PCA)
print("\n[4/6] Processando texto (TF-IDF + PCA)...")

# Amostragem para visualização
sample_size = min(2000, len(df_labeled))
df_sample = df_labeled.sample(n=sample_size, random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=100, stop_words='english', min_df=2)
X_tfidf = vectorizer.fit_transform(df_sample['text'])
print(f"  ✓ TF-IDF: {X_tfidf.shape[1]} features")

# Normalização
scaler = StandardScaler(with_mean=False)
X_scaled = scaler.fit_transform(X_tfidf)

# PCA para 2D (visualização)
pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_scaled.toarray())
print(f"  ✓ PCA: 2 componentes ({pca_2d.explained_variance_ratio_.sum()*100:.2f}% variância)")

# 5. DETECTAR OUTLIERS
print("\n[5/6] Detectando outliers (Elliptic Envelope)...")
contamination = threat_count / len(df_labeled)
ee = EllipticEnvelope(contamination=contamination, random_state=42)
predictions = ee.fit_predict(X_pca_2d)

outliers_detected = (predictions == -1).sum()
print(f"  ✓ {outliers_detected} outliers detectados")

# 6. CRIAR VISUALIZAÇÃO
print("\n[6/6] Gerando gráfico de dispersão...")

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# ----- SUBPLOT 1: GROUND TRUTH -----
ax1 = axes[0]

# Separar dados
y_true = df_sample['is_threat'].values
normal_mask = (y_true == 0)
threat_mask = (y_true == 1)

# Plotar pontos normais
ax1.scatter(X_pca_2d[normal_mask, 0], 
           X_pca_2d[normal_mask, 1],
           c='#2ecc71', 
           marker='o', 
           s=60, 
           alpha=0.6, 
           edgecolors='black',
           linewidth=0.5,
           label=f'Normal ({normal_mask.sum()})')

# Plotar pontos maliciosos
ax1.scatter(X_pca_2d[threat_mask, 0], 
           X_pca_2d[threat_mask, 1],
           c='#e74c3c', 
           marker='^', 
           s=80, 
           alpha=0.8, 
           edgecolors='black',
           linewidth=0.8,
           label=f'Malicioso ({threat_mask.sum()})')

ax1.set_xlabel('Componente Principal 1', fontsize=13, fontweight='bold')
ax1.set_ylabel('Componente Principal 2', fontsize=13, fontweight='bold')
ax1.set_title('GROUND TRUTH: Dados Reais\n(Ameaças Conhecidas)', 
             fontsize=15, fontweight='bold', pad=20)
ax1.legend(loc='upper right', fontsize=11, frameon=True, shadow=True)
ax1.grid(alpha=0.3, linestyle='--')

# Adicionar estatísticas
stats_text1 = f'Total: {len(df_sample)}\nMaliciosos: {threat_mask.sum()}\nNormais: {normal_mask.sum()}'
ax1.text(0.02, 0.98, stats_text1, transform=ax1.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# ----- SUBPLOT 2: OUTLIERS DETECTADOS -----
ax2 = axes[1]

# Separar outliers detectados
outlier_mask = (predictions == -1)
normal_pred_mask = (predictions == 1)

# Plotar pontos normais (segundo modelo)
ax2.scatter(X_pca_2d[normal_pred_mask, 0], 
           X_pca_2d[normal_pred_mask, 1],
           c='#3498db', 
           marker='o', 
           s=60, 
           alpha=0.6, 
           edgecolors='black',
           linewidth=0.5,
           label=f'Normal - Predito ({normal_pred_mask.sum()})')

# Plotar outliers detectados
ax2.scatter(X_pca_2d[outlier_mask, 0], 
           X_pca_2d[outlier_mask, 1],
           c='#f39c12', 
           marker='X', 
           s=100, 
           alpha=0.9, 
           edgecolors='black',
           linewidth=1.0,
           label=f'Outlier - Detectado ({outlier_mask.sum()})')

# Destacar True Positives (outliers que são realmente ameaças)
true_positives_mask = outlier_mask & threat_mask
if true_positives_mask.sum() > 0:
    ax2.scatter(X_pca_2d[true_positives_mask, 0], 
               X_pca_2d[true_positives_mask, 1],
               c='none', 
               marker='o', 
               s=200, 
               edgecolors='#27ae60',
               linewidth=3,
               label=f'True Positives ({true_positives_mask.sum()})')

ax2.set_xlabel('Componente Principal 1', fontsize=13, fontweight='bold')
ax2.set_ylabel('Componente Principal 2', fontsize=13, fontweight='bold')
ax2.set_title('DETECÇÃO DE OUTLIERS: Elliptic Envelope\n(Sem Ver os Rótulos)', 
             fontsize=15, fontweight='bold', pad=20)
ax2.legend(loc='upper right', fontsize=11, frameon=True, shadow=True)
ax2.grid(alpha=0.3, linestyle='--')

# Calcular métricas de validação
tp = (outlier_mask & threat_mask).sum()
tn = (normal_pred_mask & normal_mask).sum()
fp = (outlier_mask & normal_mask).sum()
fn = (normal_pred_mask & threat_mask).sum()

accuracy = (tp + tn) / len(df_sample) * 100
precision = tp / outlier_mask.sum() * 100 if outlier_mask.sum() > 0 else 0

stats_text2 = f'''Outliers: {outlier_mask.sum()}
True Positives: {tp}
Accuracy: {accuracy:.1f}%
Precision: {precision:.1f}%'''

ax2.text(0.02, 0.98, stats_text2, transform=ax2.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

# Ajustes finais
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_scatter_plot_outliers.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ Gráfico salvo: {OUTPUT_DIR}/05_scatter_plot_outliers.png")

# 7. RESUMO
print("\n" + "="*80)
print("RESUMO DA VISUALIZAÇÃO")
print("="*80)
print(f"\n📊 Dataset:")
print(f"  • Total de pontos: {len(df_sample)}")
print(f"  • Ameaças reais: {threat_mask.sum()}")
print(f"  • Normais: {normal_mask.sum()}")

print(f"\n🎯 Detecção:")
print(f"  • Outliers detectados: {outlier_mask.sum()}")
print(f"  • True Positives: {tp}")
print(f"  • False Positives: {fp}")
print(f"  • True Negatives: {tn}")
print(f"  • False Negatives: {fn}")

print(f"\n📈 Métricas:")
print(f"  • Accuracy: {accuracy:.2f}%")
print(f"  • Precision: {precision:.2f}%")

print("\n" + "="*80)
print("✅ GRÁFICO DE DISPERSÃO GERADO COM SUCESSO!")
print("="*80)

print("\n💡 INTERPRETAÇÃO:")
print("  • ESQUERDA: Mostra os dados reais (verde=normal, vermelho=malicioso)")
print("  • DIREITA: Mostra o que o modelo detectou (azul=normal, laranja=outlier)")
print("  • CÍRCULOS VERDES: São os True Positives (acertos!)")
print("\n")
