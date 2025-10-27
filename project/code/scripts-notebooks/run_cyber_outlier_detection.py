"""
Script completo para detecção de outliers em Cyber Threat Intelligence
Atualizado para usar o dataset real baixado do Kaggle

Autor: Projeto Iniciação Científica - Faculdade Impacta
Data: Outubro 2025
"""

import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime

# Visualização
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn: Pré-processamento
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# Scikit-learn: Detecção de Outliers
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import DBSCAN

# Scikit-learn: Métricas
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Configurações
warnings.filterwarnings('ignore')
np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Diretórios
OUTPUT_DIR = 'output-images'
RESULTS_DIR = 'results'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("="*80)
print("DETECÇÃO DE OUTLIERS EM CYBER THREAT INTELLIGENCE")
print("Dataset: Text-Based Cyber Threat Detection (Kaggle)")
print("="*80)

# 1. CARREGAMENTO DOS DADOS
print("\n[1/7] Carregando dataset...")
data_path = '../../data/cyber-outlier-detection/cyber-threat-intelligence_all.csv'
df = pd.read_csv(data_path)
print(f"✓ Dataset carregado: {df.shape}")

# 2. PRÉ-PROCESSAMENTO
print("\n[2/7] Pré-processamento...")

# Filtrar registros com labels
df_labeled = df[df['label'].notna()].copy()
print(f"  Registros com labels: {len(df_labeled):,}")

# Criar labels binários (threat vs normal)
threat_labels = ['malware', 'attack-pattern', 'threat-actor', 'vulnerability', 'tools']
df_labeled['is_threat'] = df_labeled['label'].apply(
    lambda x: 1 if str(x).lower() in threat_labels else 0
)

threat_count = (df_labeled['is_threat'] == 1).sum()
normal_count = (df_labeled['is_threat'] == 0).sum()
print(f"  Threats: {threat_count:,} ({threat_count/len(df_labeled)*100:.1f}%)")
print(f"  Normal:  {normal_count:,} ({normal_count/len(df_labeled)*100:.1f}%)")

# Amostra estratificada para análise
sample_size = min(15000, len(df_labeled))

# Amostragem estratificada manual
df_threat = df_labeled[df_labeled['is_threat'] == 1]
df_normal = df_labeled[df_labeled['is_threat'] == 0]

n_threat = int(sample_size * (len(df_threat) / len(df_labeled)))
n_normal = sample_size - n_threat

df_threat_sample = df_threat.sample(n=min(n_threat, len(df_threat)), random_state=42)
df_normal_sample = df_normal.sample(n=min(n_normal, len(df_normal)), random_state=42)

df_sample = pd.concat([df_threat_sample, df_normal_sample]).sample(frac=1, random_state=42)
print(f"  Usando amostra de {len(df_sample):,} registros")

# Calcular contamination
contamination = threat_count / len(df_labeled)

# 2.5. VISUALIZAÇÃO EXPLORATÓRIA
print("\n[2.5/7] Gerando análise exploratória...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Distribuição de comprimento de texto
ax1 = axes[0, 0]
text_lengths = df_sample['text'].str.len()
ax1.hist(text_lengths, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
ax1.set_xlabel('Comprimento do Texto (caracteres)', fontsize=12)
ax1.set_ylabel('Frequência', fontsize=12)
ax1.set_title('Distribuição do Comprimento dos Textos', fontsize=14, fontweight='bold')
ax1.axvline(text_lengths.mean(), color='red', linestyle='--', linewidth=2, label=f'Média: {text_lengths.mean():.0f}')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Top 15 labels
ax2 = axes[0, 1]
top_labels = df_sample['label'].value_counts().head(15)
bars = ax2.barh(range(len(top_labels)), top_labels.values, color='coral', edgecolor='black')
ax2.set_yticks(range(len(top_labels)))
ax2.set_yticklabels(top_labels.index, fontsize=10)
ax2.set_xlabel('Contagem', fontsize=12)
ax2.set_title('Top 15 Categorias de Labels', fontsize=14, fontweight='bold')
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)

for i, (bar, value) in enumerate(zip(bars, top_labels.values)):
    ax2.text(value + 5, i, f'{value}', va='center', fontsize=9)

# 3. Distribuição de ameaças
ax3 = axes[1, 0]
threat_counts = df_sample['is_threat'].value_counts()
colors = ['#66c2a5', '#fc8d62']
explode = (0.05, 0)
wedges, texts, autotexts = ax3.pie(threat_counts.values, 
                                     labels=['Normal', 'Threat'],
                                     colors=colors,
                                     autopct='%1.1f%%',
                                     startangle=90,
                                     explode=explode,
                                     textprops={'fontsize': 12, 'fontweight': 'bold'})
ax3.set_title('Proporção: Ameaças vs Normal', fontsize=14, fontweight='bold')

# 4. Estatísticas do dataset
ax4 = axes[1, 1]
ax4.axis('off')
stats_text = f"""
ESTATÍSTICAS DO DATASET

Total de Registros: {len(df_sample):,}
Ameaças (Threats): {threat_count:,} ({threat_count/len(df_labeled)*100:.1f}%)
Normais: {normal_count:,} ({normal_count/len(df_labeled)*100:.1f}%)

Comprimento Médio: {text_lengths.mean():.0f} caracteres
Comprimento Mín/Máx: {text_lengths.min()}/{text_lengths.max()}

Labels Únicos: {df_sample['label'].nunique()}
Contaminação: {contamination:.3f}

Dataset Completo: 19,940 registros
Amostrados: {len(df_sample):,} registros
"""
ax4.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_exploratory_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ {OUTPUT_DIR}/01_exploratory_analysis.png")

# 3. FEATURE ENGINEERING (TF-IDF)
print("\n[3/7] Feature engineering (TF-IDF)...")

vectorizer = TfidfVectorizer(
    max_features=100,
    max_df=0.85,
    min_df=2,
    stop_words='english'
)

X_tfidf = vectorizer.fit_transform(df_sample['text'].fillna('')).toarray()
y_true = df_sample['is_threat'].values

print(f"  Features TF-IDF: {X_tfidf.shape}")
print(f"  Variância média: {X_tfidf.var(axis=0).mean():.4f}")

# Aplicar PCA para reduzir dimensionalidade
pca = PCA(n_components=min(50, X_tfidf.shape[1]), random_state=42)
X_pca = pca.fit_transform(X_tfidf)
print(f"  Features após PCA: {X_pca.shape}")
print(f"  Variância explicada: {pca.explained_variance_ratio_.sum():.2%}")

# Normalizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)
print(f"✓ Features normalizadas")

# 4. DETECÇÃO DE OUTLIERS
print("\n[4/7] Aplicando técnicas de detecção de outliers...")

# Configurar contamination baseado na proporção de threats
contamination = threat_count / len(df_labeled)
print(f"  Contamination estimado: {contamination:.4f}")

results = []

# 4.1 Isolation Forest
print("\n  [1/5] Isolation Forest...")
iso_forest = IsolationForest(
    contamination=contamination,
    random_state=42,
    n_estimators=100
)
predictions_if = iso_forest.fit_predict(X_scaled)
predictions_if_binary = (predictions_if == -1).astype(int)  # -1 = outlier, 1 = normal

accuracy_if = accuracy_score(y_true, predictions_if_binary)
precision_if = precision_score(y_true, predictions_if_binary, zero_division=0)
recall_if = recall_score(y_true, predictions_if_binary, zero_division=0)
f1_if = f1_score(y_true, predictions_if_binary, zero_division=0)

results.append({
    'Método': 'Isolation Forest',
    'Accuracy': accuracy_if,
    'Precision': precision_if,
    'Recall': recall_if,
    'F1-Score': f1_if,
    'Outliers Detectados': predictions_if_binary.sum(),
    'Outliers Reais': y_true.sum()
})
print(f"    Accuracy: {accuracy_if:.4f}")

# 4.2 Local Outlier Factor
print("  [2/5] Local Outlier Factor...")
lof = LocalOutlierFactor(
    contamination=contamination,
    novelty=False,
    n_neighbors=20
)
predictions_lof = lof.fit_predict(X_scaled)
predictions_lof_binary = (predictions_lof == -1).astype(int)

accuracy_lof = accuracy_score(y_true, predictions_lof_binary)
precision_lof = precision_score(y_true, predictions_lof_binary, zero_division=0)
recall_lof = recall_score(y_true, predictions_lof_binary, zero_division=0)
f1_lof = f1_score(y_true, predictions_lof_binary, zero_division=0)

results.append({
    'Método': 'LOF',
    'Accuracy': accuracy_lof,
    'Precision': precision_lof,
    'Recall': recall_lof,
    'F1-Score': f1_lof,
    'Outliers Detectados': predictions_lof_binary.sum(),
    'Outliers Reais': y_true.sum()
})
print(f"    Accuracy: {accuracy_lof:.4f}")

# 4.3 One-Class SVM
print("  [3/5] One-Class SVM...")
ocsvm = OneClassSVM(
    nu=contamination,
    kernel='rbf',
    gamma='auto'
)
predictions_ocsvm = ocsvm.fit_predict(X_scaled)
predictions_ocsvm_binary = (predictions_ocsvm == -1).astype(int)

accuracy_ocsvm = accuracy_score(y_true, predictions_ocsvm_binary)
precision_ocsvm = precision_score(y_true, predictions_ocsvm_binary, zero_division=0)
recall_ocsvm = recall_score(y_true, predictions_ocsvm_binary, zero_division=0)
f1_ocsvm = f1_score(y_true, predictions_ocsvm_binary, zero_division=0)

results.append({
    'Método': 'One-Class SVM',
    'Accuracy': accuracy_ocsvm,
    'Precision': precision_ocsvm,
    'Recall': recall_ocsvm,
    'F1-Score': f1_ocsvm,
    'Outliers Detectados': predictions_ocsvm_binary.sum(),
    'Outliers Reais': y_true.sum()
})
print(f"    Accuracy: {accuracy_ocsvm:.4f}")

# 4.4 Elliptic Envelope
print("  [4/5] Elliptic Envelope...")
elliptic = EllipticEnvelope(
    contamination=contamination,
    random_state=42
)
predictions_ee = elliptic.fit_predict(X_scaled)
predictions_ee_binary = (predictions_ee == -1).astype(int)

accuracy_ee = accuracy_score(y_true, predictions_ee_binary)
precision_ee = precision_score(y_true, predictions_ee_binary, zero_division=0)
recall_ee = recall_score(y_true, predictions_ee_binary, zero_division=0)
f1_ee = f1_score(y_true, predictions_ee_binary, zero_division=0)

results.append({
    'Método': 'Elliptic Envelope',
    'Accuracy': accuracy_ee,
    'Precision': precision_ee,
    'Recall': recall_ee,
    'F1-Score': f1_ee,
    'Outliers Detectados': predictions_ee_binary.sum(),
    'Outliers Reais': y_true.sum()
})
print(f"    Accuracy: {accuracy_ee:.4f}")

# 4.5 DBSCAN
print("  [5/5] DBSCAN...")
dbscan = DBSCAN(eps=0.5, min_samples=5)
predictions_dbscan = dbscan.fit_predict(X_scaled)
predictions_dbscan_binary = (predictions_dbscan == -1).astype(int)

accuracy_dbscan = accuracy_score(y_true, predictions_dbscan_binary)
precision_dbscan = precision_score(y_true, predictions_dbscan_binary, zero_division=0)
recall_dbscan = recall_score(y_true, predictions_dbscan_binary, zero_division=0)
f1_dbscan = f1_score(y_true, predictions_dbscan_binary, zero_division=0)

results.append({
    'Método': 'DBSCAN',
    'Accuracy': accuracy_dbscan,
    'Precision': precision_dbscan,
    'Recall': recall_dbscan,
    'F1-Score': f1_dbscan,
    'Outliers Detectados': predictions_dbscan_binary.sum(),
    'Outliers Reais': y_true.sum()
})
print(f"    Accuracy: {accuracy_dbscan:.4f}")

# 5. SALVAR RESULTADOS
print("\n[5/7] Salvando resultados...")
df_results = pd.DataFrame(results)
df_results.to_csv(f'{RESULTS_DIR}/evaluation_metrics.csv', index=False)
print(f"✓ Métricas salvas em: {RESULTS_DIR}/evaluation_metrics.csv")

# 6. VISUALIZAÇÕES
print("\n[6/7] Gerando visualizações...")

# 6.1 Métricas comparativas
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

for idx, (metric, color) in enumerate(zip(metrics, colors)):
    ax = axes[idx // 2, idx % 2]
    bars = ax.bar(df_results['Método'], df_results[metric], color=color, alpha=0.7, edgecolor='black')
    
    ax.axhline(y=0.70, color='red', linestyle='--', linewidth=2, label='Meta 70%', alpha=0.7)
    ax.set_ylabel(metric, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric} por Método', fontsize=14, fontweight='bold')
    ax.set_xticklabels(df_results['Método'], rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Adicionar valores
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_comparative_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ {OUTPUT_DIR}/02_comparative_metrics.png")

# 6.2 Comparação de outliers detectados vs ameaças reais
fig, ax = plt.subplots(figsize=(12, 7))

methods = list(df_results['Método'])
threats_real = y_true.sum()
outliers_detected = [
    predictions_if_binary.sum(),
    predictions_lof_binary.sum(),
    predictions_ocsvm_binary.sum(),
    predictions_ee_binary.sum(),
    predictions_dbscan_binary.sum()
]
true_positives = [
    ((y_true == 1) & (predictions_if_binary == 1)).sum(),
    ((y_true == 1) & (predictions_lof_binary == 1)).sum(),
    ((y_true == 1) & (predictions_ocsvm_binary == 1)).sum(),
    ((y_true == 1) & (predictions_ee_binary == 1)).sum(),
    ((y_true == 1) & (predictions_dbscan_binary == 1)).sum()
]

x = np.arange(len(methods))
width = 0.25

bars1 = ax.bar(x - width, [threats_real]*len(methods), width, label='Ameaças Reais', color='#e74c3c', edgecolor='black', alpha=0.8)
bars2 = ax.bar(x, outliers_detected, width, label='Outliers Detectados', color='#3498db', edgecolor='black', alpha=0.8)
bars3 = ax.bar(x + width, true_positives, width, label='True Positives', color='#2ecc71', edgecolor='black', alpha=0.8)

ax.set_xlabel('Método de Detecção', fontsize=13, fontweight='bold')
ax.set_ylabel('Número de Amostras', fontsize=13, fontweight='bold')
ax.set_title('Comparação: Outliers Detectados vs Ameaças Reais', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=20, ha='right', fontsize=11)
ax.legend(fontsize=11, loc='upper left')
ax.grid(axis='y', alpha=0.3)

# Adicionar valores nas barras
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_outliers_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ {OUTPUT_DIR}/03_outliers_comparison.png")

# 6.3 Matrizes de confusão
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

predictions_all = [
    ('Isolation Forest', predictions_if_binary),
    ('LOF', predictions_lof_binary),
    ('One-Class SVM', predictions_ocsvm_binary),
    ('Elliptic Envelope', predictions_ee_binary),
    ('DBSCAN', predictions_dbscan_binary)
]

for idx, (method, preds) in enumerate(predictions_all):
    ax = axes[idx // 3, idx % 3]
    cm = confusion_matrix(y_true, preds)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=['Normal', 'Threat'], 
                yticklabels=['Normal', 'Threat'])
    ax.set_title(f'{method}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Real', fontsize=10)
    ax.set_xlabel('Predito', fontsize=10)

# Remover subplot extra
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ {OUTPUT_DIR}/04_confusion_matrices.png")

# 7. RESUMO
print("\n[7/7] Resumo dos Resultados")
print("="*80)
print(df_results.to_string(index=False))
print("\n" + "="*80)

best_method = df_results.loc[df_results['Accuracy'].idxmax()]
print(f"\n✅ MELHOR MÉTODO: {best_method['Método']}")
print(f"   Accuracy:  {best_method['Accuracy']:.4f}")
print(f"   Precision: {best_method['Precision']:.4f}")
print(f"   Recall:    {best_method['Recall']:.4f}")
print(f"   F1-Score:  {best_method['F1-Score']:.4f}")

print("\n" + "="*80)
print("✓ ANÁLISE CONCLUÍDA COM SUCESSO!")
print(f"   Resultados salvos em: {RESULTS_DIR}/")
print(f"   Visualizações em: {OUTPUT_DIR}/")
print(f"     • 01_exploratory_analysis.png")
print(f"     • 02_comparative_metrics.png")
print(f"     • 03_outliers_comparison.png")
print(f"     • 04_confusion_matrices.png")
print("="*80)
