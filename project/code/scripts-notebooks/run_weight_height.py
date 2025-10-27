# Script de execução: Weight-Height Dataset Evaluation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    silhouette_score, davies_bouldin_score
)
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*60)
print("AVALIAÇÃO - WEIGHT-HEIGHT DATASET")
print("="*60)

# 1. GERAÇÃO DOS DADOS
print("\n[1/5] Gerando dados...")
np.random.seed(42)
n_samples = 10000

n_male = n_samples // 2
male_height = np.random.normal(178, 7, n_male)
male_weight = np.random.normal(85, 12, n_male)

n_female = n_samples - n_male
female_height = np.random.normal(163, 6, n_female)
female_weight = np.random.normal(65, 10, n_female)

df = pd.DataFrame({
    'Gender': ['Male'] * n_male + ['Female'] * n_female,
    'Height': np.concatenate([male_height, female_height]),
    'Weight': np.concatenate([male_weight, female_weight])
})

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"✓ {len(df)} amostras geradas")

# 2. VISUALIZAÇÃO EXPLORATÓRIA
print("\n[2/5] Gerando visualização exploratória...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for gender, color in [('Male', 'blue'), ('Female', 'red')]:
    data = df[df['Gender'] == gender]
    axes[0, 0].scatter(data['Height'], data['Weight'], 
                       c=color, alpha=0.4, s=10, label=gender)
axes[0, 0].set_xlabel('Altura (cm)', fontweight='bold')
axes[0, 0].set_ylabel('Peso (kg)', fontweight='bold')
axes[0, 0].set_title('Altura vs Peso', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

axes[0, 1].hist([df[df['Gender']=='Male']['Height'],
                 df[df['Gender']=='Female']['Height']],
                label=['Male', 'Female'], alpha=0.7, bins=50)
axes[0, 1].set_xlabel('Altura (cm)', fontweight='bold')
axes[0, 1].set_ylabel('Frequência', fontweight='bold')
axes[0, 1].set_title('Distribuição de Altura', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(axis='y', alpha=0.3)

axes[1, 0].hist([df[df['Gender']=='Male']['Weight'],
                 df[df['Gender']=='Female']['Weight']],
                label=['Male', 'Female'], alpha=0.7, bins=50)
axes[1, 0].set_xlabel('Peso (kg)', fontweight='bold')
axes[1, 0].set_ylabel('Frequência', fontweight='bold')
axes[1, 0].set_title('Distribuição de Peso', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

df.boxplot(column=['Height', 'Weight'], by='Gender', ax=axes[1, 1])
axes[1, 1].set_title('Boxplots por Gênero', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Medida', fontweight='bold')
plt.suptitle('')

plt.tight_layout()
plt.savefig('weight_height_exploratory.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ weight_height_exploratory.png")

# 3. APRENDIZADO SUPERVISIONADO
print("\n[3/5] Treinando modelos supervisionados...")
X = df[['Height', 'Weight']].values
y = np.where(df['Gender'] == 'Male', 1, 0)

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

results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    results.append({
        'Modelo': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    })
    print(f"  ✓ {name}")

df_results = pd.DataFrame(results)
df_results.to_csv('weight_height_supervised_results.csv', index=False)

# Visualização
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    bars = ax.bar(df_results['Modelo'], df_results[metric], color='lightyellow', edgecolor='black')
    
    ax.axhline(y=0.70, color='red', linestyle='--', linewidth=2, label='Meta 70%')
    
    for bar, value in zip(bars, df_results[metric]):
        if value >= 0.70:
            bar.set_color('green')
            bar.set_alpha(0.7)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel(metric, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric} - Weight-Height Dataset', fontsize=14, fontweight='bold')
    ax.set_xticklabels(df_results['Modelo'], rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('weight_height_supervised_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ weight_height_supervised_metrics.png")

# 4. CLUSTERING
print("\n[4/5] Executando clustering...")
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=42)
clusters_km = kmeans.fit_predict(X_scaled)
sil_km = silhouette_score(X_scaled, clusters_km)
db_km = davies_bouldin_score(X_scaled, clusters_km)

hier = AgglomerativeClustering(n_clusters=2, linkage='average')
clusters_hier = hier.fit_predict(X_scaled)
sil_hier = silhouette_score(X_scaled, clusters_hier)
db_hier = davies_bouldin_score(X_scaled, clusters_hier)

df_unsup = pd.DataFrame({
    'Método': ['K-Means', 'Hierarchical'],
    'Silhouette': [sil_km, sil_hier],
    'Davies-Bouldin': [db_km, db_hier]
})
df_unsup.to_csv('weight_height_unsupervised_results.csv', index=False)

# Visualização clustering
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for gender, color in zip([1, 0], ['blue', 'red']):
    mask = y == gender
    axes[0].scatter(X_scaled[mask, 0], X_scaled[mask, 1], 
                   c=color, alpha=0.4, s=10, label=['Male', 'Female'][1-gender])
axes[0].set_title('Classes Reais', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Height (normalizada)')
axes[0].set_ylabel('Weight (normalizada)')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters_km, cmap='viridis', alpha=0.4, s=10)
axes[1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
               c='red', marker='X', s=300, edgecolors='black', linewidths=2, label='Centróides')
axes[1].set_title(f'K-Means\n(Silhouette: {sil_km:.3f})', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Height (normalizada)')
axes[1].set_ylabel('Weight (normalizada)')
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[2].scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters_hier, cmap='viridis', alpha=0.4, s=10)
axes[2].set_title(f'Hierarchical\n(Silhouette: {sil_hier:.3f})', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Height (normalizada)')
axes[2].set_ylabel('Weight (normalizada)')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('weight_height_clustering_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ weight_height_clustering_comparison.png")

# 5. RESUMO
print("\n[5/5] Gerando resumo...")
print("\n" + "="*60)
print("RESULTADOS - WEIGHT-HEIGHT DATASET")
print("="*60)

print("\n1. SUPERVISIONADO:")
print(df_results.round(4).to_string(index=False))
best = df_results.loc[df_results['Accuracy'].idxmax()]
print(f"\n   Melhor: {best['Modelo']} ({best['Accuracy']:.4f})")

print("\n2. NÃO SUPERVISIONADO:")
print(f"   K-Means Silhouette: {sil_km:.4f}")
print(f"   Hierarchical Silhouette: {sil_hier:.4f}")

print("\n3. ARQUIVOS GERADOS:")
print("   ✓ weight_height_exploratory.png")
print("   ✓ weight_height_supervised_metrics.png")
print("   ✓ weight_height_clustering_comparison.png")
print("   ✓ weight_height_supervised_results.csv")
print("   ✓ weight_height_unsupervised_results.csv")

print("\n" + "="*60)
print("✓ AVALIAÇÃO CONCLUÍDA COM SUCESSO!")
print("="*60)
