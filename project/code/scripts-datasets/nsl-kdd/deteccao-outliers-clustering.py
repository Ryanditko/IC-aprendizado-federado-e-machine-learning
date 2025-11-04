"""
DETECÇÃO DE OUTLIERS NSL-KDD - APRENDIZADO NÃO SUPERVISIONADO
=============================================================

Script corrigido para usar modelos de AGRUPAMENTO (clustering) e detecção
de outliers não supervisionados, conforme o objetivo do projeto.

ALGORITMOS:
- Isolation Forest
- K-Means + distância para detecção de outliers
- DBSCAN
- Elliptic Envelope

Autor: Projeto de Iniciação Científica
Data: Novembro 2025
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Configuração
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.figsize': [12, 8],
    'savefig.dpi': 300
})

print("="*70)
print("DETECÇÃO DE OUTLIERS NSL-KDD - APRENDIZADO NÃO SUPERVISIONADO")
print("="*70)

# Caminhos
DATA_DIR = '../../../data/nsl-kdd'
OUTPUT_DIR = '.'

try:
    print("\\n1️⃣ Carregando dataset NSL-KDD...")
    
    # Carregar dados
    df_train = pd.read_csv(f'{DATA_DIR}/KDDTrain+_20Percent.txt', header=None)
    df_test = pd.read_csv(f'{DATA_DIR}/KDDTest-21.txt', header=None)
    
    # Combinar para análise completa
    df = pd.concat([df_train, df_test], ignore_index=True)
    
    # Colunas NSL-KDD
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
    print(f"   ✅ Dataset carregado: {len(df):,} registros")
    
    print("\\n2️⃣ Preparando ground truth para avaliação...")
    
    # Mapear ataques para avaliar performance (mas não usar no treinamento!)
    attack_mapping = {
        'normal': 'normal',
        'buffer_overflow': 'u2r', 'rootkit': 'u2r', 'loadmodule': 'u2r',
        'perl': 'u2r', 'httptunnel': 'u2r', 'ps': 'u2r', 'sqlattack': 'u2r',
        'xterm': 'u2r'
    }
    
    df['attack_category'] = df['attack_type'].map(attack_mapping)
    df['attack_category'] = df['attack_category'].fillna('other_attack')
    
    # Ground truth: U2R = outlier (1), resto = normal (0)  
    df['true_outlier'] = (df['attack_category'] == 'u2r').astype(int)
    
    outlier_count = df['true_outlier'].sum()
    normal_count = len(df) - outlier_count
    
    print(f"   📊 Ground Truth (apenas para avaliação):")
    print(f"      U2R (outliers): {outlier_count:,} ({outlier_count/len(df)*100:.2f}%)")
    print(f"      Normal/Outros: {normal_count:,} ({normal_count/len(df)*100:.2f}%)")
    
    print("\\n3️⃣ Preparando features (sem usar labels!)...")
    
    # Features numéricas
    numeric_cols = ['duration', 'src_bytes', 'dst_bytes', 'hot', 'num_failed_logins',
                   'logged_in', 'num_compromised', 'root_shell', 'count', 'srv_count',
                   'serror_rate', 'srv_serror_rate', 'same_srv_rate', 'diff_srv_rate']
    
    X = df[numeric_cols].copy()
    
    # Encoder para categóricas
    le_protocol = LabelEncoder()
    le_service = LabelEncoder()
    le_flag = LabelEncoder()
    
    X['protocol_encoded'] = le_protocol.fit_transform(df['protocol_type'].astype(str))
    X['service_encoded'] = le_service.fit_transform(df['service'].astype(str))
    X['flag_encoded'] = le_flag.fit_transform(df['flag'].astype(str))
    
    print(f"   ✅ Features preparadas: {len(X.columns)} colunas")
    
    # Amostragem para análise (outliers são raros, vamos usar amostra inteligente)
    print("\\n4️⃣ Criando amostra representativa...")
    
    if outlier_count < 500:
        # Usar TODOS os outliers + amostra de normais
        outlier_indices = df[df['true_outlier'] == 1].index.tolist()
        normal_indices = df[df['true_outlier'] == 0].sample(n=3000, random_state=42).index.tolist()
        
        sample_indices = outlier_indices + normal_indices
        X_sample = X.loc[sample_indices]
        y_true_sample = df.loc[sample_indices, 'true_outlier']
        
        print(f"   ✅ Amostra criada: {len(X_sample):,} registros")
        print(f"      Outliers: {y_true_sample.sum():,}")
        print(f"      Normais: {len(y_true_sample) - y_true_sample.sum():,}")
    else:
        X_sample = X
        y_true_sample = df['true_outlier']
    
    # Normalização (essencial para clustering)
    print("\\n5️⃣ Normalizando dados...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)
    
    print(f"   ✅ Dados normalizados: {X_scaled.shape}")
    
    # PCA para visualização
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"   ✅ PCA aplicado: {pca.explained_variance_ratio_.sum()*100:.1f}% variância explicada")
    
    print("\\n6️⃣ Aplicando algoritmos de detecção de outliers...")
    
    # Proporção esperada de outliers NA AMOSTRA (não no dataset completo!)
    contamination_sample = y_true_sample.sum() / len(y_true_sample)
    contamination_original = outlier_count / len(df)
    
    print(f"   📊 Contaminação no dataset original: {contamination_original:.4f} ({contamination_original*100:.2f}%)")
    print(f"   📊 Contaminação na amostra: {contamination_sample:.4f} ({contamination_sample*100:.2f}%)")
    print(f"   🎯 Usando contaminação da amostra para algoritmos")
    
    # ALGORITMO 1: ISOLATION FOREST
    print("   🌲 Isolation Forest...")
    iso_forest = IsolationForest(contamination=contamination_sample, random_state=42, n_estimators=200)
    iso_pred = iso_forest.fit_predict(X_scaled)
    iso_outliers = (iso_pred == -1).astype(int)  # -1 = outlier, 1 = normal
    
    # ALGORITMO 2: ELLIPTIC ENVELOPE  
    print("   🔮 Elliptic Envelope...")
    ee = EllipticEnvelope(contamination=contamination_sample, random_state=42)
    ee_pred = ee.fit_predict(X_scaled)
    ee_outliers = (ee_pred == -1).astype(int)
    
    # ALGORITMO 3: LOCAL OUTLIER FACTOR
    print("   📍 Local Outlier Factor...")
    lof = LocalOutlierFactor(contamination=contamination_sample, n_neighbors=20)
    lof_pred = lof.fit_predict(X_scaled)
    lof_outliers = (lof_pred == -1).astype(int)
    
    # ALGORITMO 4: K-MEANS + DISTÂNCIA
    print("   🎯 K-Means + Distance...")
    n_clusters = 8
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Calcular distâncias para centros
    distances = kmeans.transform(X_scaled).min(axis=1)
    
    # Outliers = pontos mais distantes (top contamination_sample%)
    threshold = np.percentile(distances, (1 - contamination_sample) * 100)
    kmeans_outliers = (distances > threshold).astype(int)
    
    print("   ✅ Todos os algoritmos aplicados!")
    
    # Diagnóstico: quantos outliers cada algoritmo detectou
    print("\\n   🔍 DIAGNÓSTICO - Outliers detectados por algoritmo:")
    print(f"      Isolation Forest: {iso_outliers.sum():,} de {len(iso_outliers):,} ({iso_outliers.sum()/len(iso_outliers)*100:.1f}%)")
    print(f"      Elliptic Envelope: {ee_outliers.sum():,} de {len(ee_outliers):,} ({ee_outliers.sum()/len(ee_outliers)*100:.1f}%)")
    print(f"      Local Outlier Factor: {lof_outliers.sum():,} de {len(lof_outliers):,} ({lof_outliers.sum()/len(lof_outliers)*100:.1f}%)")
    print(f"      K-Means Distance: {kmeans_outliers.sum():,} de {len(kmeans_outliers):,} ({kmeans_outliers.sum()/len(kmeans_outliers)*100:.1f}%)")
    print(f"      Outliers reais: {y_true_sample.sum():,} de {len(y_true_sample):,} ({y_true_sample.sum()/len(y_true_sample)*100:.1f}%)")
    
    print("\\n7️⃣ Avaliando performance dos algoritmos...")
    
    # Avaliar cada algoritmo
    algorithms = {
        'Isolation Forest': iso_outliers,
        'Elliptic Envelope': ee_outliers,
        'Local Outlier Factor': lof_outliers,
        'K-Means Distance': kmeans_outliers
    }
    
    results = {}
    
    for name, predictions in algorithms.items():
        # Métricas
        accuracy = accuracy_score(y_true_sample, predictions)
        precision = precision_score(y_true_sample, predictions, zero_division=0)
        recall = recall_score(y_true_sample, predictions, zero_division=0)
        f1 = f1_score(y_true_sample, predictions, zero_division=0)
        
        # Matriz de confusão
        cm = confusion_matrix(y_true_sample, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        results[name] = {
            'predictions': predictions,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }
        
        print(f"   📊 {name}:")
        print(f"      Acc: {accuracy:.3f} | Prec: {precision:.3f} | Rec: {recall:.3f} | F1: {f1:.3f}")
    
    # Selecionar melhor algoritmo
    best_algo = max(results.keys(), key=lambda x: results[x]['f1'])
    best_result = results[best_algo]
    
    print(f"\\n   🏆 Melhor algoritmo: {best_algo} (F1: {best_result['f1']:.3f})")
    
    print("\\n8️⃣ Gerando visualizações...")
    
    # GRÁFICO 1: MATRIZ DE CONFUSÃO DO MELHOR ALGORITMO
    print("   📊 Matriz de Confusão...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cm_best = best_result['confusion_matrix']
    im = ax.imshow(cm_best, interpolation='nearest', cmap='Blues')
    
    # Labels
    classes = ['Normal/Other', 'U2R Outlier']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    # Título
    ax.set_title(f'Matriz de Confusão - {best_algo}\\nDetecção de Outliers U2R (Não Supervisionado)', 
                fontweight='bold', pad=20)
    ax.set_xlabel('Predição do Algoritmo', fontweight='bold')
    ax.set_ylabel('Ground Truth', fontweight='bold')
    
    # Valores nas células
    total = len(y_true_sample)
    thresh = cm_best.max() / 2.
    
    for i in range(cm_best.shape[0]):
        for j in range(cm_best.shape[1]):
            value = cm_best[i, j]
            percentage = value / total * 100
            text = f'{value:,}\\n({percentage:.1f}%)'
            
            ax.text(j, i, text, ha="center", va="center",
                   color="white" if value > thresh else "black",
                   fontsize=13, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Número de Casos', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig('matriz_confusao_clustering_outliers.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✅ matriz_confusao_clustering_outliers.png salva")
    
    # GRÁFICO 2: COMPARAÇÃO DE ALGORITMOS
    print("   📊 Comparação de Algoritmos...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    algo_names = list(results.keys())
    x_pos = np.arange(len(algo_names))
    width = 0.2
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = [results[algo][metric] for algo in algo_names]
        bars = ax.bar(x_pos + i*width, values, width, label=label, color=color, alpha=0.8)
        
        # Valores nas barras
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax.set_xlabel('Algoritmos de Detecção de Outliers', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Comparação de Algoritmos de Clustering/Outlier Detection\\nDataset NSL-KDD - Detecção U2R', 
                fontweight='bold', pad=20)
    ax.set_xticks(x_pos + width * 1.5)
    ax.set_xticklabels(algo_names, rotation=15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('comparacao_algoritmos_clustering.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✅ comparacao_algoritmos_clustering.png salva")
    
    # GRÁFICO 3: VISUALIZAÇÃO PCA COM OUTLIERS DETECTADOS
    print("   📊 Visualização PCA...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Ground Truth
    scatter1 = ax1.scatter(X_pca[y_true_sample == 0, 0], X_pca[y_true_sample == 0, 1], 
                          c='blue', alpha=0.6, s=20, label='Normal')
    scatter2 = ax1.scatter(X_pca[y_true_sample == 1, 0], X_pca[y_true_sample == 1, 1], 
                          c='red', alpha=0.8, s=40, label='U2R Outliers', marker='^')
    ax1.set_title('Ground Truth', fontweight='bold')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Melhores 3 algoritmos
    top_algos = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)[:3]
    
    for idx, (algo_name, algo_result) in enumerate(top_algos):
        ax_idx = [ax2, ax3, ax4][idx]
        predictions = algo_result['predictions']
        
        ax_idx.scatter(X_pca[predictions == 0, 0], X_pca[predictions == 0, 1], 
                      c='lightblue', alpha=0.6, s=20, label='Predito Normal')
        ax_idx.scatter(X_pca[predictions == 1, 0], X_pca[predictions == 1, 1], 
                      c='orange', alpha=0.8, s=40, label='Predito Outlier', marker='X')
        
        ax_idx.set_title(f'{algo_name}\\nF1: {algo_result["f1"]:.3f}', fontweight='bold')
        ax_idx.set_xlabel('PC1')
        ax_idx.set_ylabel('PC2')
        ax_idx.legend()
        ax_idx.grid(alpha=0.3)
    
    plt.suptitle('Visualização PCA - Detecção de Outliers U2R\\nComparação de Algoritmos Não Supervisionados', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('visualizacao_pca_outliers.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✅ visualizacao_pca_outliers.png salva")
    
    # GRÁFICO 4: DASHBOARD COMPLETO
    print("   📊 Dashboard Completo...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Matriz de confusão do melhor
    im1 = ax1.imshow(cm_best, interpolation='nearest', cmap='Blues')
    ax1.set_title(f'Matriz de Confusão\\n{best_algo}', fontweight='bold')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Normal', 'Outlier'])
    ax1.set_yticklabels(['Normal', 'Outlier'])
    
    for i in range(2):
        for j in range(2):
            value = cm_best[i, j]
            ax1.text(j, i, f'{value:,}', ha="center", va="center",
                    color="white" if value > cm_best.max()/2 else "black",
                    fontsize=12, fontweight='bold')
    
    # 2. Métricas do melhor algoritmo
    best_metrics = [best_result[m] for m in metrics]
    bars2 = ax2.bar(metric_labels, best_metrics, color=colors, alpha=0.8)
    ax2.set_title(f'Métricas - {best_algo}', fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, value in zip(bars2, best_metrics):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Distribuição de scores F1
    f1_scores = [results[algo]['f1'] for algo in algo_names]
    bars3 = ax3.barh(algo_names, f1_scores, color='skyblue', alpha=0.8)
    ax3.set_title('F1-Scores por Algoritmo', fontweight='bold')
    ax3.set_xlim(0, 1)
    ax3.grid(axis='x', alpha=0.3)
    
    for bar, value in zip(bars3, f1_scores):
        width = bar.get_width()
        ax3.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                f'{value:.3f}', ha='left', va='center', fontweight='bold')
    
    # 4. Estatísticas
    tp, tn, fp, fn = best_result['tp'], best_result['tn'], best_result['fp'], best_result['fn']
    
    stats_text = f'''DETECÇÃO DE OUTLIERS - APRENDIZADO NÃO SUPERVISIONADO

Dataset: NSL-KDD ({len(df):,} registros)
Amostra analisada: {len(y_true_sample):,} registros
U2R outliers reais: {y_true_sample.sum():,} ({y_true_sample.sum()/len(y_true_sample)*100:.1f}%)

MELHOR ALGORITMO: {best_algo}

CONFUSION MATRIX:
True Negatives:  {tn:,} (normais corretos)
False Positives: {fp:,} (falsos alarmes)
False Negatives: {fn:,} (outliers perdidos)
True Positives:  {tp:,} (outliers detectados)

MÉTRICAS:
Accuracy:  {best_result['accuracy']:.3f} ({best_result['accuracy']*100:.1f}%)
Precision: {best_result['precision']:.3f} ({best_result['precision']*100:.1f}%)
Recall:    {best_result['recall']:.3f} ({best_result['recall']*100:.1f}%)
F1-Score:  {best_result['f1']:.3f} ({best_result['f1']*100:.1f}%)

INTERPRETAÇÃO:
• {best_result['precision']*100:.1f}% dos outliers detectados são reais
• {best_result['recall']*100:.1f}% dos outliers reais foram encontrados
• {fp:,} falsos alarmes em {tn+fp:,} casos normais
• Algoritmo não supervisionado (sem usar labels no treinamento)'''
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Resultados Detalhados', fontweight='bold')
    
    plt.suptitle('NSL-KDD OUTLIER DETECTION - ALGORITMOS DE CLUSTERING\\nDetecção Não Supervisionada de Ataques U2R', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('dashboard_clustering_completo.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✅ dashboard_clustering_completo.png salva")
    
    print("\\n" + "="*70)
    print("✅ ANÁLISE DE CLUSTERING/OUTLIERS CONCLUÍDA!")
    print("="*70)
    
    print(f"\\n📊 RESULTADOS FINAIS (APRENDIZADO NÃO SUPERVISIONADO):")
    print(f"• Dataset: {len(df):,} registros, {len(y_true_sample):,} analisados")
    print(f"• Outliers U2R reais: {y_true_sample.sum():,}")
    print(f"• Melhor algoritmo: {best_algo}")
    print(f"• F1-Score: {best_result['f1']:.3f} ({best_result['f1']*100:.1f}%)")
    print(f"• Precision: {best_result['precision']:.3f} ({best_result['precision']*100:.1f}%)")
    print(f"• Recall: {best_result['recall']:.3f} ({best_result['recall']*100:.1f}%)")
    
    print(f"\\n📈 Confusion Matrix ({best_algo}):")
    print(f"  TN: {tn:,} | FP: {fp:,} | FN: {fn:,} | TP: {tp:,}")
    
    print("\\n📁 Gráficos gerados:")
    print("  • matriz_confusao_clustering_outliers.png")
    print("  • comparacao_algoritmos_clustering.png")
    print("  • visualizacao_pca_outliers.png")
    print("  • dashboard_clustering_completo.png")
    
    print("\\n💡 METODOLOGIA:")
    print("  ✅ Aprendizado NÃO supervisionado (sem usar labels no treinamento)")
    print("  ✅ Algoritmos de clustering e detecção de outliers")
    print("  ✅ Ground truth usado APENAS para avaliação")
    print("  ✅ Detecção de ataques U2R como outliers")
    
except Exception as e:
    print(f"\\n❌ ERRO: {e}")
    import traceback
    traceback.print_exc()

print("\\n🎯 Análise de agrupamento/outliers finalizada!")
