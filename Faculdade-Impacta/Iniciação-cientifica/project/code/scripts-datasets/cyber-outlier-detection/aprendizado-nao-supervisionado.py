"""
Aprendizado Não-Supervisionado - Cyber Threat Detection
========================================================

Dataset: Text-Based Cyber Threat Detection (Kaggle)
Objetivo: Detectar outliers/anomalias usando técnicas não-supervisionadas
Métodos: Isolation Forest, LOF, One-Class SVM, Elliptic Envelope, DBSCAN

Autor: Projeto de Iniciação Científica - Faculdade Impacta
Data: Outubro 2025
"""

import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
import time

# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# Métodos de detecção de outliers
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import DBSCAN

# Métricas
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================

# Caminhos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', '..', 'data', 'cyber-outlier-detection')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Criar diretórios
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parâmetros
SAMPLE_SIZE = 15000  # Amostra para performance
RANDOM_STATE = 42
N_PCA_COMPONENTS = 50
MAX_TFIDF_FEATURES = 500

print("="*80)
print("APRENDIZADO NÃO-SUPERVISIONADO - CYBER THREAT DETECTION")
print("="*80)
print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Dataset: Text-Based Cyber Threat Detection")
print(f"Objetivo: Detecção de Outliers/Anomalias")
print(f"Diretório de dados: {DATA_DIR}")
print(f"Diretório de saída: {OUTPUT_DIR}")
print("="*80)

# ============================================================================
# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ============================================================================

print("\n[1/5] CARREGAMENTO DOS DADOS")
print("-" * 80)

# Carregar dataset
data_path = os.path.join(DATA_DIR, 'cyber-threat-intelligence_all.csv')

if not os.path.exists(data_path):
    print(f"ERRO: Dataset não encontrado em {data_path}")
    print("Execute o script de download do dataset primeiro:")
    print("  python download_cyber_dataset.py")
    exit(1)

df = pd.read_csv(data_path)
print(f"✓ Dataset carregado: {len(df):,} registros, {len(df.columns)} colunas")

# Filtrar apenas registros com labels (para validação)
df_labeled = df[df['label'].notna()].copy()
print(f"✓ Registros com labels: {len(df_labeled):,} ({len(df_labeled)/len(df)*100:.1f}%)")

# Criar labels binários: threat vs normal
threat_labels = ['malware', 'attack-pattern', 'threat-actor', 'vulnerability', 'tools']
df_labeled['is_threat'] = df_labeled['label'].apply(
    lambda x: 1 if str(x).lower() in threat_labels else 0
)

contamination = (df_labeled['is_threat'] == 1).sum() / len(df_labeled)

print(f"\nDistribuição de classes:")
print(f"  Threats (1): {(df_labeled['is_threat'] == 1).sum():,} ({contamination*100:.1f}%)")
print(f"  Normal  (0): {(df_labeled['is_threat'] == 0).sum():,} ({(1-contamination)*100:.1f}%)")
print(f"\n  Contaminação estimada: {contamination:.3f}")

# Amostra estratificada
sample_size = min(SAMPLE_SIZE, len(df_labeled))
df_sample = df_labeled.sample(n=sample_size, random_state=RANDOM_STATE, stratify=df_labeled['is_threat'])
print(f"\n✓ Amostra estratificada: {len(df_sample):,} registros")

# ============================================================================
# 2. PRÉ-PROCESSAMENTO E FEATURE ENGINEERING
# ============================================================================

print("\n[2/5] PRÉ-PROCESSAMENTO E FEATURE ENGINEERING")
print("-" * 80)

# TF-IDF Vectorization
print("Aplicando TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=MAX_TFIDF_FEATURES,
    min_df=5,
    max_df=0.7,
    ngram_range=(1, 2),
    stop_words='english',
    strip_accents='unicode'
)

X_tfidf = vectorizer.fit_transform(df_sample['text'])
print(f"✓ TF-IDF: {X_tfidf.shape[0]:,} amostras, {X_tfidf.shape[1]} features")
print(f"  Sparsity: {(1.0 - X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1])) * 100:.2f}%")

# Converter para array denso
X_tfidf_dense = X_tfidf.toarray()

# PCA para redução de dimensionalidade
print(f"\nAplicando PCA ({N_PCA_COMPONENTS} componentes)...")
pca = PCA(n_components=N_PCA_COMPONENTS, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_tfidf_dense)
variance_explained = pca.explained_variance_ratio_.sum()

print(f"✓ PCA: {X_pca.shape[0]:,} amostras, {X_pca.shape[1]} componentes")
print(f"  Variância explicada: {variance_explained*100:.2f}%")

# Normalização
print("\nNormalizando features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)
print(f"✓ Normalização completa (mean={X_scaled.mean():.2e}, std={X_scaled.std():.2f})")

# Labels verdadeiros (para validação)
y_true = df_sample['is_threat'].values

# ============================================================================
# 3. DETECÇÃO DE OUTLIERS - MÚLTIPLAS TÉCNICAS
# ============================================================================

print("\n[3/5] DETECÇÃO DE OUTLIERS - MÚLTIPLAS TÉCNICAS")
print("-" * 80)
print(f"Contaminação esperada: {contamination:.3f} ({contamination*100:.1f}%)\n")

outlier_predictions = {}
execution_times = {}

# 1. ISOLATION FOREST
print("1. Isolation Forest...")
start = time.time()
iso_forest = IsolationForest(
    contamination=contamination,
    random_state=RANDOM_STATE,
    n_estimators=100,
    max_samples='auto',
    n_jobs=-1
)
pred_iso = iso_forest.fit_predict(X_scaled)
outlier_predictions['Isolation Forest'] = (pred_iso == -1).astype(int)
execution_times['Isolation Forest'] = time.time() - start
print(f"   ✓ Outliers: {(pred_iso == -1).sum():,} ({(pred_iso == -1).sum()/len(pred_iso)*100:.1f}%)")
print(f"   ✓ Tempo: {execution_times['Isolation Forest']:.2f}s")

# 2. LOCAL OUTLIER FACTOR
print("\n2. Local Outlier Factor (LOF)...")
start = time.time()
lof = LocalOutlierFactor(
    contamination=contamination,
    n_neighbors=20,
    n_jobs=-1
)
pred_lof = lof.fit_predict(X_scaled)
outlier_predictions['LOF'] = (pred_lof == -1).astype(int)
execution_times['LOF'] = time.time() - start
print(f"   ✓ Outliers: {(pred_lof == -1).sum():,} ({(pred_lof == -1).sum()/len(pred_lof)*100:.1f}%)")
print(f"   ✓ Tempo: {execution_times['LOF']:.2f}s")

# 3. ONE-CLASS SVM
print("\n3. One-Class SVM...")
start = time.time()
oc_svm = OneClassSVM(
    nu=contamination,
    kernel='rbf',
    gamma='auto'
)
pred_ocsvm = oc_svm.fit_predict(X_scaled)
outlier_predictions['One-Class SVM'] = (pred_ocsvm == -1).astype(int)
execution_times['One-Class SVM'] = time.time() - start
print(f"   ✓ Outliers: {(pred_ocsvm == -1).sum():,} ({(pred_ocsvm == -1).sum()/len(pred_ocsvm)*100:.1f}%)")
print(f"   ✓ Tempo: {execution_times['One-Class SVM']:.2f}s")

# 4. ELLIPTIC ENVELOPE
print("\n4. Elliptic Envelope...")
start = time.time()
ee = EllipticEnvelope(
    contamination=contamination,
    random_state=RANDOM_STATE,
    support_fraction=0.9
)
pred_ee = ee.fit_predict(X_scaled)
outlier_predictions['Elliptic Envelope'] = (pred_ee == -1).astype(int)
execution_times['Elliptic Envelope'] = time.time() - start
print(f"   ✓ Outliers: {(pred_ee == -1).sum():,} ({(pred_ee == -1).sum()/len(pred_ee)*100:.1f}%)")
print(f"   ✓ Tempo: {execution_times['Elliptic Envelope']:.2f}s")

# 5. DBSCAN
print("\n5. DBSCAN (Clustering + Outliers)...")
start = time.time()
dbscan = DBSCAN(
    eps=3.0,
    min_samples=10,
    n_jobs=-1
)
pred_dbscan = dbscan.fit_predict(X_scaled)
outlier_predictions['DBSCAN'] = (pred_dbscan == -1).astype(int)
execution_times['DBSCAN'] = time.time() - start
n_clusters = len(set(pred_dbscan)) - (1 if -1 in pred_dbscan else 0)
print(f"   ✓ Clusters: {n_clusters}")
print(f"   ✓ Outliers: {(pred_dbscan == -1).sum():,} ({(pred_dbscan == -1).sum()/len(pred_dbscan)*100:.1f}%)")
print(f"   ✓ Tempo: {execution_times['DBSCAN']:.2f}s")

# ============================================================================
# 4. VALIDAÇÃO CONTRA LABELS REAIS
# ============================================================================

print("\n[4/5] VALIDAÇÃO CONTRA LABELS REAIS")
print("-" * 80)

results = []

for method_name, y_pred in outlier_predictions.items():
    # Métricas
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    results.append({
        'Método': method_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn,
        'Tempo (s)': execution_times[method_name]
    })
    
    print(f"\n{method_name}:")
    print(f"  Accuracy:  {accuracy*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  F1-Score:  {f1*100:.2f}%")

# ============================================================================
# 5. ANÁLISE E EXPORTAÇÃO DOS RESULTADOS
# ============================================================================

print("\n[5/5] ANÁLISE E EXPORTAÇÃO DOS RESULTADOS")
print("-" * 80)

# Criar DataFrame com resultados
df_results = pd.DataFrame(results)
df_results = df_results.sort_values('F1-Score', ascending=False).reset_index(drop=True)
df_results['Rank'] = range(1, len(df_results) + 1)

print("\nRANKING DOS MÉTODOS (por F1-Score):")
print(df_results[['Rank', 'Método', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Tempo (s)']].to_string(index=False))

# Melhor método
best_method = df_results.iloc[0]
print(f"\n🏆 MELHOR MÉTODO: {best_method['Método']}")
print(f"   Accuracy:  {best_method['Accuracy']*100:.2f}%")
print(f"   Precision: {best_method['Precision']*100:.2f}%")
print(f"   Recall:    {best_method['Recall']*100:.2f}%")
print(f"   F1-Score:  {best_method['F1-Score']*100:.2f}%")
print(f"   Tempo:     {best_method['Tempo (s)']:.2f}s")

# Salvar resultados
output_path = os.path.join(OUTPUT_DIR, 'unsupervised_results.csv')
df_results.to_csv(output_path, index=False)
print(f"\n✓ Resultados salvos em: {output_path}")

# Estatísticas gerais
print("\n" + "="*80)
print("ESTATÍSTICAS GERAIS")
print("="*80)
print(f"Média Accuracy:  {df_results['Accuracy'].mean()*100:.2f}%")
print(f"Média Precision: {df_results['Precision'].mean()*100:.2f}%")
print(f"Média Recall:    {df_results['Recall'].mean()*100:.2f}%")
print(f"Média F1-Score:  {df_results['F1-Score'].mean()*100:.2f}%")
print(f"Tempo total:     {df_results['Tempo (s)'].sum():.2f}s")

# Insights
print("\n" + "="*80)
print("INSIGHTS")
print("="*80)
print(f"✓ {len(outlier_predictions)} técnicas de detecção de outliers avaliadas")
print(f"✓ Validação contra {len(y_true):,} amostras com labels reais")
print(f"✓ Melhor método: {best_method['Método']} (F1: {best_method['F1-Score']*100:.1f}%)")

avg_precision = df_results['Precision'].mean()
avg_recall = df_results['Recall'].mean()

if avg_precision > 0.5:
    print(f"✓ Boa precision média ({avg_precision*100:.1f}%) - outliers correspondem a ameaças")
else:
    print(f"⚠ Precision baixa ({avg_precision*100:.1f}%) - muitos falsos positivos")

if avg_recall > 0.5:
    print(f"✓ Boa recall média ({avg_recall*100:.1f}%) - maioria das ameaças detectadas")
else:
    print(f"⚠ Recall baixo ({avg_recall*100:.1f}%) - muitas ameaças não detectadas")

print("\n" + "="*80)
print("✅ ANÁLISE NÃO-SUPERVISIONADA CONCLUÍDA COM SUCESSO!")
print("="*80)
print(f"\nAplicabilidade em Aprendizado Federado:")
print(f"  • Detecção de agentes maliciosos em ambientes distribuídos")
print(f"  • Identificação de atualizações suspeitas de modelos")
print(f"  • Mitigação de ataques por envenenamento de dados")
