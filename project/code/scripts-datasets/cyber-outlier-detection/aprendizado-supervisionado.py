"""
Aprendizado Supervisionado - Cyber Threat Detection
====================================================

Dataset: Text-Based Cyber Threat Detection (Kaggle)
Objetivo: Classificar textos como ameaças cibernéticas ou normais
Modelos: Decision Tree, Random Forest, SVM, Naive Bayes, K-NN, Logistic Regression

Autor: Projeto de Iniciação Científica - Faculdade Impacta
Data: Outubro 2025
"""

import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# Modelos
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Métricas
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
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
TEST_SIZE = 0.3
RANDOM_STATE = 42
N_PCA_COMPONENTS = 50
MAX_TFIDF_FEATURES = 500

print("="*80)
print("APRENDIZADO SUPERVISIONADO - CYBER THREAT DETECTION")
print("="*80)
print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Dataset: Text-Based Cyber Threat Detection")
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

# Filtrar apenas registros com labels
df_labeled = df[df['label'].notna()].copy()
print(f"✓ Registros com labels: {len(df_labeled):,} ({len(df_labeled)/len(df)*100:.1f}%)")

# Criar labels binários: threat vs normal
threat_labels = ['malware', 'attack-pattern', 'threat-actor', 'vulnerability', 'tools']
df_labeled['is_threat'] = df_labeled['label'].apply(
    lambda x: 1 if str(x).lower() in threat_labels else 0
)

print(f"\nDistribuição de classes:")
print(f"  Threats (1): {(df_labeled['is_threat'] == 1).sum():,} ({(df_labeled['is_threat'] == 1).sum()/len(df_labeled)*100:.1f}%)")
print(f"  Normal  (0): {(df_labeled['is_threat'] == 0).sum():,} ({(df_labeled['is_threat'] == 0).sum()/len(df_labeled)*100:.1f}%)")

# Amostra estratificada
sample_size = min(SAMPLE_SIZE, len(df_labeled))
df_sample = df_labeled.sample(n=sample_size, random_state=RANDOM_STATE, stratify=df_labeled['is_threat'])
print(f"\n✓ Amostra estratificada: {len(df_sample):,} registros")
print(f"  Threats: {(df_sample['is_threat'] == 1).sum():,}")
print(f"  Normal:  {(df_sample['is_threat'] == 0).sum():,}")

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

# Labels
y = df_sample['is_threat'].values

# ============================================================================
# 3. DIVISÃO TREINO/TESTE
# ============================================================================

print("\n[3/5] DIVISÃO TREINO/TESTE")
print("-" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"Conjunto de treino: {len(X_train):,} amostras")
print(f"  Threats: {(y_train == 1).sum():,} ({(y_train == 1).sum()/len(y_train)*100:.1f}%)")
print(f"  Normal:  {(y_train == 0).sum():,} ({(y_train == 0).sum()/len(y_train)*100:.1f}%)")

print(f"\nConjunto de teste: {len(X_test):,} amostras")
print(f"  Threats: {(y_test == 1).sum():,} ({(y_test == 1).sum()/len(y_test)*100:.1f}%)")
print(f"  Normal:  {(y_test == 0).sum():,} ({(y_test == 0).sum()/len(y_test)*100:.1f}%)")

# ============================================================================
# 4. TREINAMENTO DOS MODELOS
# ============================================================================

print("\n[4/5] TREINAMENTO DOS MODELOS")
print("-" * 80)

# Dicionário de modelos
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=10),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    'SVM': SVC(kernel='rbf', random_state=RANDOM_STATE),
    'Naive Bayes': GaussianNB(),
    'K-NN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, n_jobs=-1)
}

results = []

for name, model in models.items():
    print(f"\nTreinando {name}...")
    
    # Treinar
    model.fit(X_train, y_train)
    
    # Predições
    y_pred = model.predict(X_test)
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    results.append({
        'Modelo': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    
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
df_results = df_results.sort_values('Accuracy', ascending=False).reset_index(drop=True)
df_results['Rank'] = range(1, len(df_results) + 1)

print("\nRANKING DOS MODELOS (por Accuracy):")
print(df_results.to_string(index=False))

# Melhor modelo
best_model = df_results.iloc[0]
print(f"\n🏆 MELHOR MODELO: {best_model['Modelo']}")
print(f"   Accuracy:  {best_model['Accuracy']*100:.2f}%")
print(f"   Precision: {best_model['Precision']*100:.2f}%")
print(f"   Recall:    {best_model['Recall']*100:.2f}%")
print(f"   F1-Score:  {best_model['F1-Score']*100:.2f}%")

# Salvar resultados
output_path = os.path.join(OUTPUT_DIR, 'supervised_results.csv')
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

print("\n" + "="*80)
print("✅ ANÁLISE SUPERVISIONADA CONCLUÍDA COM SUCESSO!")
print("="*80)
