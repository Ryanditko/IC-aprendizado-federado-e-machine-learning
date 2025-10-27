# Script de execução: Penguins Dataset Evaluation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    silhouette_score, davies_bouldin_score
)
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*60)
print("AVALIAÇÃO - PENGUINS DATASET")
print("="*60)

# 1. CARREGAMENTO DOS DADOS
print("\n[1/5] Carregando dataset...")
# Tentar carregar do arquivo local primeiro
try:
    df = pd.read_csv('../../data/penguins.csv')
    print("✓ Dataset carregado do arquivo local")
except:
    # Se não funcionar, usar seaborn
    import seaborn as sns
    df = sns.load_dataset('penguins')
    print("✓ Dataset carregado do seaborn")

df_clean = df.dropna()
print(f"✓ {len(df_clean)} amostras carregadas (após limpeza)")

# 2. VISUALIZAÇÃO EXPLORATÓRIA
print("\n[2/5] Gerando visualização exploratória...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for species in df_clean['species'].unique():
    data = df_clean[df_clean['species'] == species]
    axes[0, 0].scatter(data['bill_length_mm'], data['bill_depth_mm'], 
                       label=species, alpha=0.6, s=50)
axes[0, 0].set_xlabel('Bill Length (mm)', fontweight='bold')
axes[0, 0].set_ylabel('Bill Depth (mm)', fontweight='bold')
axes[0, 0].set_title('Bill Dimensions', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

for species in df_clean['species'].unique():
    data = df_clean[df_clean['species'] == species]
    axes[0, 1].scatter(data['flipper_length_mm'], data['body_mass_g'], 
                       label=species, alpha=0.6, s=50)
axes[0, 1].set_xlabel('Flipper Length (mm)', fontweight='bold')
axes[0, 1].set_ylabel('Body Mass (g)', fontweight='bold')
axes[0, 1].set_title('Body Dimensions', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

pd.crosstab(df_clean['island'], df_clean['species']).plot(kind='bar', ax=axes[1, 0])
axes[1, 0].set_title('Distribuição por Ilha', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Ilha', fontweight='bold')
axes[1, 0].set_ylabel('Contagem', fontweight='bold')
axes[1, 0].legend(title='Espécie')

pd.crosstab(df_clean['sex'], df_clean['species']).plot(kind='bar', ax=axes[1, 1])
axes[1, 1].set_title('Distribuição por Sexo', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Sexo', fontweight='bold')
axes[1, 1].set_ylabel('Contagem', fontweight='bold')
axes[1, 1].legend(title='Espécie')

plt.tight_layout()
plt.savefig('penguins_exploratory.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ penguins_exploratory.png")

# 3. PREPARAÇÃO DOS DADOS
print("\n[3/5] Preparando dados e treinando modelos...")
numeric_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
X_numeric = df_clean[numeric_features].values

le_island = LabelEncoder()
le_sex = LabelEncoder()
island_encoded = le_island.fit_transform(df_clean['island']).reshape(-1, 1)
sex_encoded = le_sex.fit_transform(df_clean['sex']).reshape(-1, 1)

X = np.hstack([X_numeric, island_encoded, sex_encoded])

le_species = LabelEncoder()
y = le_species.fit_transform(df_clean['species'])

# APRENDIZADO SUPERVISIONADO
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "SVM": SVC(random_state=42),
    "Naive Bayes": GaussianNB(),
    "K-NN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

results_supervised = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    results_supervised.append({
        'Modelo': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1-Score': f1_score(y_test, y_pred, average='weighted')
    })
    print(f"  ✓ {name}")

df_supervised = pd.DataFrame(results_supervised)
df_supervised.to_csv('penguins_supervised_results.csv', index=False)

# Visualização supervisionada
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    bars = ax.bar(df_supervised['Modelo'], df_supervised[metric], color='lightcoral', edgecolor='black')
    
    ax.axhline(y=0.70, color='red', linestyle='--', linewidth=2, label='Meta 70%')
    
    for bar, value in zip(bars, df_supervised[metric]):
        if value >= 0.70:
            bar.set_color('green')
            bar.set_alpha(0.7)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel(metric, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric} por Modelo - Penguins Dataset', fontsize=14, fontweight='bold')
    ax.set_xticklabels(df_supervised['Modelo'], rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('penguins_supervised_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ penguins_supervised_metrics.png")

# 4. CLUSTERING
print("\n[4/5] Executando clustering...")
X_scaled = StandardScaler().fit_transform(X_numeric)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters_kmeans = kmeans.fit_predict(X_scaled)
silhouette_kmeans = silhouette_score(X_scaled, clusters_kmeans)
davies_bouldin_kmeans = davies_bouldin_score(X_scaled, clusters_kmeans)

hierarchical = AgglomerativeClustering(n_clusters=3, linkage='average')
clusters_hier = hierarchical.fit_predict(X_scaled)
silhouette_hier = silhouette_score(X_scaled, clusters_hier)
davies_bouldin_hier = davies_bouldin_score(X_scaled, clusters_hier)

results_unsupervised = pd.DataFrame({
    'Método': ['K-Means', 'Hierarchical'],
    'Silhouette Score': [silhouette_kmeans, silhouette_hier],
    'Davies-Bouldin Index': [davies_bouldin_kmeans, davies_bouldin_hier]
})
results_unsupervised.to_csv('penguins_unsupervised_results.csv', index=False)

# Visualização clustering
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=50, alpha=0.7, edgecolors='k')
axes[0].set_title('Classes Reais - Penguins', fontsize=14, fontweight='bold')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.colorbar(scatter1, ax=axes[0])

scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_kmeans, cmap='viridis', s=50, alpha=0.7, edgecolors='k')
axes[1].set_title(f'K-Means\n(Silhouette: {silhouette_kmeans:.3f})', fontsize=14, fontweight='bold')
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.colorbar(scatter2, ax=axes[1])

scatter3 = axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_hier, cmap='viridis', s=50, alpha=0.7, edgecolors='k')
axes[2].set_title(f'Hierarchical\n(Silhouette: {silhouette_hier:.3f})', fontsize=14, fontweight='bold')
axes[2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.colorbar(scatter3, ax=axes[2])

plt.tight_layout()
plt.savefig('penguins_clustering_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ penguins_clustering_comparison.png")

# 5. PCA ANALYSIS
print("\n[5/5] Análise PCA...")
pca_full = PCA()
pca_full.fit(X_scaled)

var_exp = pca_full.explained_variance_ratio_
var_cum = np.cumsum(var_exp)

n_comp_95 = (var_cum >= 0.95).argmax() + 1

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].bar(range(1, len(var_exp)+1), var_exp, alpha=0.7, color='steelblue', edgecolor='black')
axes[0].set_xlabel('Componente', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Variância Explicada', fontsize=12, fontweight='bold')
axes[0].set_title('Variância por Componente - Penguins', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

axes[1].plot(range(1, len(var_cum)+1), var_cum, marker='o', linewidth=2, markersize=8, color='darkorange')
axes[1].axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95%')
axes[1].axvline(x=n_comp_95, color='green', linestyle='--', linewidth=2, label=f'{n_comp_95} Comp.')
axes[1].set_xlabel('Número de Componentes', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Variância Acumulada', fontsize=12, fontweight='bold')
axes[1].set_title('Variância Acumulada - Penguins', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('penguins_pca_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ penguins_pca_analysis.png")

# RESUMO
print("\n" + "="*60)
print("RESULTADOS - PENGUINS DATASET")
print("="*60)

print("\n1. SUPERVISIONADO:")
print(df_supervised.round(4).to_string(index=False))
best = df_supervised.loc[df_supervised['Accuracy'].idxmax()]
print(f"\n   Melhor: {best['Modelo']} ({best['Accuracy']:.4f})")

print("\n2. NÃO SUPERVISIONADO:")
print(f"   K-Means Silhouette: {silhouette_kmeans:.4f}")
print(f"   Hierarchical Silhouette: {silhouette_hier:.4f}")

print("\n3. ARQUIVOS GERADOS:")
print("   ✓ penguins_exploratory.png")
print("   ✓ penguins_supervised_metrics.png")
print("   ✓ penguins_clustering_comparison.png")
print("   ✓ penguins_pca_analysis.png")
print("   ✓ penguins_supervised_results.csv")
print("   ✓ penguins_unsupervised_results.csv")

print("\n" + "="*60)
print("✓ AVALIAÇÃO CONCLUÍDA COM SUCESSO!")
print("="*60)
