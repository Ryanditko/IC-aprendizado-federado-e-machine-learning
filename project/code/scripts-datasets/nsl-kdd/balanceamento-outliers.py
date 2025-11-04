"""
DETECÇÃO DE OUTLIERS NSL-KDD COM BALANCEAMENTO DE DADOS
=======================================================

Script para testar diferentes técnicas de balanceamento em detecção de outliers
não supervisionada para melhorar a performance dos algoritmos.

TÉCNICAS DE BALANCEAMENTO:
1. OVERSAMPLING - SMOTE (Synthetic Minority Oversampling Technique)
2. UNDERSAMPLING - Random Undersampling  
3. COMBINADO - SMOTEENN (SMOTE + Edited Nearest Neighbours)
4. BASELINE - Sem balanceamento (controle)

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
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
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

print("="*80)
print("DETECÇÃO DE OUTLIERS NSL-KDD - BALANCEAMENTO DE DADOS")
print("="*80)

# Caminhos
DATA_DIR = '../../../data/nsl-kdd'
OUTPUT_DIR = '.'

def explain_technique(technique_name, description, how_it_works, pros, cons):
    """Explica uma técnica de balanceamento"""
    print(f"\\n{'='*60}")
    print(f"📚 EXPLICAÇÃO: {technique_name}")
    print('='*60)
    print(f"\\n🎯 DESCRIÇÃO:")
    print(f"   {description}")
    print(f"\\n⚙️ COMO FUNCIONA:")
    for step in how_it_works:
        print(f"   • {step}")
    print(f"\\n✅ VANTAGENS:")
    for pro in pros:
        print(f"   • {pro}")
    print(f"\\n❌ DESVANTAGENS:")
    for con in cons:
        print(f"   • {con}")
    print('='*60)

def evaluate_algorithms(X_scaled, y_true, contamination, technique_name):
    """Avalia algoritmos de detecção de outliers"""
    
    print(f"\\n🔍 Avaliando algoritmos com {technique_name}...")
    
    # Ajustar contaminação para não exceder 0.5 (limite do Isolation Forest)
    contamination_adjusted = min(contamination, 0.49)
    if contamination != contamination_adjusted:
        print(f"   ⚠️ Contaminação ajustada: {contamination:.3f} → {contamination_adjusted:.3f}")
    
    # ALGORITMO 1: ISOLATION FOREST
    iso_forest = IsolationForest(contamination=contamination_adjusted, random_state=42, n_estimators=200)
    iso_pred = iso_forest.fit_predict(X_scaled)
    iso_outliers = (iso_pred == -1).astype(int)
    
    # ALGORITMO 2: ELLIPTIC ENVELOPE  
    ee = EllipticEnvelope(contamination=contamination_adjusted, random_state=42)
    ee_pred = ee.fit_predict(X_scaled)
    ee_outliers = (ee_pred == -1).astype(int)
    
    # ALGORITMO 3: LOCAL OUTLIER FACTOR
    lof = LocalOutlierFactor(contamination=contamination_adjusted, n_neighbors=20)
    lof_pred = lof.fit_predict(X_scaled)
    lof_outliers = (lof_pred == -1).astype(int)
    
    # Avaliar cada algoritmo
    algorithms = {
        'Isolation Forest': iso_outliers,
        'Elliptic Envelope': ee_outliers,
        'Local Outlier Factor': lof_outliers
    }
    
    results = {}
    
    for name, predictions in algorithms.items():
        # Métricas
        accuracy = accuracy_score(y_true, predictions)
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)
        
        # Matriz de confusão
        cm = confusion_matrix(y_true, predictions)
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
        
        print(f"   📊 {name}: Acc:{accuracy:.3f} | Prec:{precision:.3f} | Rec:{recall:.3f} | F1:{f1:.3f}")
    
    # Melhor algoritmo
    best_algo = max(results.keys(), key=lambda x: results[x]['f1'])
    best_result = results[best_algo]
    
    print(f"   🏆 Melhor: {best_algo} (F1: {best_result['f1']:.3f})")
    
    return results, best_algo, best_result

try:
    print("\\n1️⃣ Carregando dataset NSL-KDD...")
    
    # Carregar dados
    df_train = pd.read_csv(f'{DATA_DIR}/KDDTrain+_20Percent.txt', header=None)
    df_test = pd.read_csv(f'{DATA_DIR}/KDDTest-21.txt', header=None)
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
    
    # Ground truth
    attack_mapping = {
        'normal': 'normal',
        'buffer_overflow': 'u2r', 'rootkit': 'u2r', 'loadmodule': 'u2r',
        'perl': 'u2r', 'httptunnel': 'u2r', 'ps': 'u2r', 'sqlattack': 'u2r',
        'xterm': 'u2r'
    }
    
    df['attack_category'] = df['attack_type'].map(attack_mapping)
    df['attack_category'] = df['attack_category'].fillna('other_attack')
    df['true_outlier'] = (df['attack_category'] == 'u2r').astype(int)
    
    outlier_count = df['true_outlier'].sum()
    normal_count = len(df) - outlier_count
    
    print(f"   📊 U2R outliers: {outlier_count:,} ({outlier_count/len(df)*100:.2f}%)")
    print(f"   📊 Normal/Outros: {normal_count:,} ({normal_count/len(df)*100:.2f}%)")
    
    print("\\n2️⃣ Preparando features...")
    
    # Features numéricas + categóricas codificadas
    numeric_cols = ['duration', 'src_bytes', 'dst_bytes', 'hot', 'num_failed_logins',
                   'logged_in', 'num_compromised', 'root_shell', 'count', 'srv_count',
                   'serror_rate', 'srv_serror_rate', 'same_srv_rate', 'diff_srv_rate']
    
    X = df[numeric_cols].copy()
    
    # Encoder categóricas
    le_protocol = LabelEncoder()
    le_service = LabelEncoder() 
    le_flag = LabelEncoder()
    
    X['protocol_encoded'] = le_protocol.fit_transform(df['protocol_type'].astype(str))
    X['service_encoded'] = le_service.fit_transform(df['service'].astype(str))
    X['flag_encoded'] = le_flag.fit_transform(df['flag'].astype(str))
    
    y = df['true_outlier'].values
    
    print(f"   ✅ Features: {len(X.columns)} colunas, {len(X):,} registros")
    
    # Amostra inicial para experimentos
    print("\\n3️⃣ Criando amostra base...")
    outlier_indices = df[df['true_outlier'] == 1].index.tolist()
    normal_indices = df[df['true_outlier'] == 0].sample(n=2000, random_state=42).index.tolist()
    
    base_indices = outlier_indices + normal_indices
    X_base = X.loc[base_indices].reset_index(drop=True)
    y_base = y[base_indices]
    
    print(f"   ✅ Amostra base: {len(X_base):,} registros")
    print(f"      Outliers: {y_base.sum():,} ({y_base.sum()/len(y_base)*100:.1f}%)")
    print(f"      Normais: {len(y_base) - y_base.sum():,} ({(len(y_base) - y_base.sum())/len(y_base)*100:.1f}%)")
    
    # Normalizar features base
    scaler = StandardScaler()
    X_base_scaled = scaler.fit_transform(X_base)
    
    # Armazenar resultados de todas as técnicas
    all_results = {}
    
    print("\\n" + "="*80)
    print("TESTANDO TÉCNICAS DE BALANCEAMENTO")
    print("="*80)
    
    # =============================================================================
    # TÉCNICA 1: BASELINE (SEM BALANCEAMENTO)
    # =============================================================================
    
    explain_technique(
        "BASELINE - SEM BALANCEAMENTO",
        "Usa os dados originais sem qualquer técnica de balanceamento",
        [
            "Mantém a distribuição original dos dados",
            "Aplica algoritmos diretamente nos dados desbalanceados",
            "Contamination baseada na proporção real de outliers"
        ],
        [
            "Preserva a distribuição natural dos dados",
            "Não introduz viés artificial", 
            "Computacionalmente eficiente"
        ],
        [
            "Algoritmos têm dificuldade com classes minoritárias",
            "Baixa detecção de outliers raros",
            "Tendência a classificar tudo como classe majoritária"
        ]
    )
    
    contamination_baseline = y_base.sum() / len(y_base)
    print(f"\\n🎯 Executando BASELINE (contaminação: {contamination_baseline:.4f})")
    
    results_baseline, best_algo_baseline, best_result_baseline = evaluate_algorithms(
        X_base_scaled, y_base, contamination_baseline, "BASELINE"
    )
    all_results['Baseline'] = {
        'results': results_baseline,
        'best_algo': best_algo_baseline,
        'best_result': best_result_baseline,
        'sample_size': len(X_base_scaled),
        'outlier_ratio': contamination_baseline
    }
    
    # =============================================================================
    # TÉCNICA 2: OVERSAMPLING - SMOTE
    # =============================================================================
    
    explain_technique(
        "OVERSAMPLING - SMOTE (Synthetic Minority Oversampling Technique)",
        "Gera exemplos sintéticos da classe minoritária para equilibrar o dataset",
        [
            "Seleciona um exemplo da classe minoritária aleatoriamente",
            "Encontra seus k vizinhos mais próximos da mesma classe",
            "Cria novos exemplos interpolando entre o exemplo e seus vizinhos",
            "Novos pontos ficam na linha entre exemplos existentes",
            "Processo repetido até atingir o balanceamento desejado"
        ],
        [
            "Aumenta informação da classe minoritária",
            "Preserva características da distribuição original",
            "Melhora detecção de padrões raros",
            "Não remove dados existentes"
        ],
        [
            "Pode gerar exemplos irrealistas em regiões de ruído",
            "Aumenta tamanho do dataset (mais processamento)",
            "Pode causar overfitting se mal aplicado",
            "Assume que interpolação entre exemplos é válida"
        ]
    )
    
    print(f"\\n🎯 Executando SMOTE...")
    
    # Aplicar SMOTE
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_smote, y_smote = smote.fit_resample(X_base, y_base)
    
    print(f"   📊 Dados após SMOTE:")
    print(f"      Total: {len(X_smote):,} registros (antes: {len(X_base):,})")
    print(f"      Outliers: {y_smote.sum():,} ({y_smote.sum()/len(y_smote)*100:.1f}%)")
    print(f"      Normais: {len(y_smote) - y_smote.sum():,} ({(len(y_smote) - y_smote.sum())/len(y_smote)*100:.1f}%)")
    
    # Normalizar dados SMOTE
    X_smote_scaled = scaler.fit_transform(X_smote)
    contamination_smote = y_smote.sum() / len(y_smote)
    
    results_smote, best_algo_smote, best_result_smote = evaluate_algorithms(
        X_smote_scaled, y_smote, contamination_smote, "SMOTE"
    )
    all_results['SMOTE'] = {
        'results': results_smote,
        'best_algo': best_algo_smote,
        'best_result': best_result_smote,
        'sample_size': len(X_smote_scaled),
        'outlier_ratio': contamination_smote
    }
    
    # =============================================================================
    # TÉCNICA 3: UNDERSAMPLING - RANDOM UNDERSAMPLING
    # =============================================================================
    
    explain_technique(
        "UNDERSAMPLING - RANDOM UNDERSAMPLING",
        "Remove exemplos da classe majoritária aleatoriamente para equilibrar",
        [
            "Identifica a classe majoritária (normal) e minoritária (outliers)",
            "Calcula quantos exemplos remover da classe majoritária",
            "Seleciona aleatoriamente exemplos da classe majoritária",
            "Remove os exemplos selecionados do dataset",
            "Mantém todos os exemplos da classe minoritária"
        ],
        [
            "Reduz tamanho do dataset (processamento mais rápido)",
            "Simples de implementar e entender",
            "Não adiciona dados sintéticos",
            "Elimina possível ruído da classe majoritária"
        ],
        [
            "Perde informação ao remover dados",
            "Pode remover exemplos importantes da classe majoritária", 
            "Reduz representatividade da classe majoritária",
            "Pode criar gaps na distribuição dos dados"
        ]
    )
    
    print(f"\\n🎯 Executando RANDOM UNDERSAMPLING...")
    
    # Aplicar Random Undersampling
    under_sampler = RandomUnderSampler(random_state=42)
    X_under, y_under = under_sampler.fit_resample(X_base, y_base)
    
    print(f"   📊 Dados após Undersampling:")
    print(f"      Total: {len(X_under):,} registros (antes: {len(X_base):,})")
    print(f"      Outliers: {y_under.sum():,} ({y_under.sum()/len(y_under)*100:.1f}%)")
    print(f"      Normais: {len(y_under) - y_under.sum():,} ({(len(y_under) - y_under.sum())/len(y_under)*100:.1f}%)")
    
    # Normalizar dados undersampling
    X_under_scaled = scaler.fit_transform(X_under)
    contamination_under = y_under.sum() / len(y_under)
    
    results_under, best_algo_under, best_result_under = evaluate_algorithms(
        X_under_scaled, y_under, contamination_under, "UNDERSAMPLING"
    )
    all_results['Undersampling'] = {
        'results': results_under,
        'best_algo': best_algo_under,
        'best_result': best_result_under,
        'sample_size': len(X_under_scaled),
        'outlier_ratio': contamination_under
    }
    
    # =============================================================================
    # TÉCNICA 4: COMBINADO - SMOTEENN
    # =============================================================================
    
    explain_technique(
        "COMBINADO - SMOTEENN (SMOTE + Edited Nearest Neighbours)",
        "Combina SMOTE para oversampling com ENN para limpeza de dados",
        [
            "PASSO 1: Aplica SMOTE para gerar exemplos sintéticos da classe minoritária",
            "PASSO 2: Aplica Edited Nearest Neighbours para limpar dados",
            "ENN remove exemplos mal classificados pelos vizinhos",
            "ENN identifica exemplos que diferem da maioria dos vizinhos",
            "Remove outliers e ruído de ambas as classes"
        ],
        [
            "Combina benefícios de oversampling e limpeza",
            "Remove exemplos problemáticos gerados pelo SMOTE",
            "Melhora qualidade da fronteira entre classes",
            "Reduz ruído mantendo balanceamento"
        ],
        [
            "Mais complexo computacionalmente",
            "Pode remover exemplos legítimos por engano",
            "Resultado final pode não estar perfeitamente balanceado",
            "Sensível à escolha de parâmetros de vizinhança"
        ]
    )
    
    print(f"\\n🎯 Executando SMOTEENN...")
    
    # Aplicar SMOTEENN
    smoteenn = SMOTEENN(random_state=42)
    X_smoteenn, y_smoteenn = smoteenn.fit_resample(X_base, y_base)
    
    print(f"   📊 Dados após SMOTEENN:")
    print(f"      Total: {len(X_smoteenn):,} registros (antes: {len(X_base):,})")
    print(f"      Outliers: {y_smoteenn.sum():,} ({y_smoteenn.sum()/len(y_smoteenn)*100:.1f}%)")
    print(f"      Normais: {len(y_smoteenn) - y_smoteenn.sum():,} ({(len(y_smoteenn) - y_smoteenn.sum())/len(y_smoteenn)*100:.1f}%)")
    
    # Normalizar dados SMOTEENN
    X_smoteenn_scaled = scaler.fit_transform(X_smoteenn)
    contamination_smoteenn = y_smoteenn.sum() / len(y_smoteenn)
    
    results_smoteenn, best_algo_smoteenn, best_result_smoteenn = evaluate_algorithms(
        X_smoteenn_scaled, y_smoteenn, contamination_smoteenn, "SMOTEENN"
    )
    all_results['SMOTEENN'] = {
        'results': results_smoteenn,
        'best_algo': best_algo_smoteenn,
        'best_result': best_result_smoteenn,
        'sample_size': len(X_smoteenn_scaled),
        'outlier_ratio': contamination_smoteenn
    }
    
    print("\\n" + "="*80)
    print("GERANDO COMPARAÇÕES E VISUALIZAÇÕES")
    print("="*80)
    
    # GRÁFICO 1: COMPARAÇÃO GERAL
    print("\\n📊 Comparação geral das técnicas...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Preparar dados para comparação
    techniques = list(all_results.keys())
    f1_scores = [all_results[tech]['best_result']['f1'] for tech in techniques]
    precision_scores = [all_results[tech]['best_result']['precision'] for tech in techniques]
    recall_scores = [all_results[tech]['best_result']['recall'] for tech in techniques]
    accuracy_scores = [all_results[tech]['best_result']['accuracy'] for tech in techniques]
    
    # F1-Scores
    bars1 = ax1.bar(techniques, f1_scores, color=['gray', 'lightblue', 'orange', 'lightgreen'], alpha=0.8)
    ax1.set_title('F1-Score por Técnica de Balanceamento', fontweight='bold')
    ax1.set_ylabel('F1-Score')
    ax1.set_ylim(0, max(f1_scores) * 1.1)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, score in zip(bars1, f1_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Precision
    bars2 = ax2.bar(techniques, precision_scores, color=['gray', 'lightblue', 'orange', 'lightgreen'], alpha=0.8)
    ax2.set_title('Precision por Técnica', fontweight='bold')
    ax2.set_ylabel('Precision')
    ax2.set_ylim(0, max(precision_scores) * 1.1)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, score in zip(bars2, precision_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Recall
    bars3 = ax3.bar(techniques, recall_scores, color=['gray', 'lightblue', 'orange', 'lightgreen'], alpha=0.8)
    ax3.set_title('Recall por Técnica', fontweight='bold')
    ax3.set_ylabel('Recall')
    ax3.set_ylim(0, max(recall_scores) * 1.1)
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, score in zip(bars3, recall_scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Tamanho da amostra
    sample_sizes = [all_results[tech]['sample_size'] for tech in techniques]
    bars4 = ax4.bar(techniques, sample_sizes, color=['gray', 'lightblue', 'orange', 'lightgreen'], alpha=0.8)
    ax4.set_title('Tamanho da Amostra por Técnica', fontweight='bold')
    ax4.set_ylabel('Número de Registros')
    ax4.grid(axis='y', alpha=0.3)
    
    for bar, size in zip(bars4, sample_sizes):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 100,
                f'{size:,}', ha='center', va='bottom', fontweight='bold', rotation=45)
    
    plt.suptitle('Comparação de Técnicas de Balanceamento - NSL-KDD Outlier Detection', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('comparacao_tecnicas_balanceamento.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✅ comparacao_tecnicas_balanceamento.png salva")
    
    # GRÁFICO 2: MATRIZ DE CONFUSÃO DA MELHOR TÉCNICA
    print("\\n📊 Matriz de confusão da melhor técnica...")
    
    # Encontrar melhor técnica
    best_technique = max(all_results.keys(), key=lambda x: all_results[x]['best_result']['f1'])
    best_overall = all_results[best_technique]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cm_best = best_overall['best_result']['confusion_matrix']
    im = ax.imshow(cm_best, interpolation='nearest', cmap='Blues')
    
    classes = ['Normal', 'U2R Outlier']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    ax.set_title(f'Melhor Resultado: {best_technique} + {best_overall["best_algo"]}\\nF1-Score: {best_overall["best_result"]["f1"]:.3f}', 
                fontweight='bold', pad=20)
    ax.set_xlabel('Predição', fontweight='bold')
    ax.set_ylabel('Ground Truth', fontweight='bold')
    
    # Valores nas células
    thresh = cm_best.max() / 2.
    for i in range(cm_best.shape[0]):
        for j in range(cm_best.shape[1]):
            value = cm_best[i, j]
            ax.text(j, i, f'{value:,}', ha="center", va="center",
                   color="white" if value > thresh else "black",
                   fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig('melhor_resultado_balanceamento.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✅ melhor_resultado_balanceamento.png salva")
    
    print("\\n" + "="*80)
    print("RESUMO DOS RESULTADOS")
    print("="*80)
    
    print(f"\\n🏆 MELHOR TÉCNICA: {best_technique}")
    print(f"🏆 MELHOR ALGORITMO: {best_overall['best_algo']}")
    print(f"🏆 F1-SCORE: {best_overall['best_result']['f1']:.3f} ({best_overall['best_result']['f1']*100:.1f}%)")
    
    print(f"\\n📊 COMPARAÇÃO DETALHADA:")
    for technique in techniques:
        result = all_results[technique]
        print(f"\\n   {technique}:")
        print(f"      Algoritmo: {result['best_algo']}")
        print(f"      F1-Score: {result['best_result']['f1']:.3f}")
        print(f"      Precision: {result['best_result']['precision']:.3f}")
        print(f"      Recall: {result['best_result']['recall']:.3f}")
        print(f"      Tamanho: {result['sample_size']:,} registros")
        print(f"      Razão outliers: {result['outlier_ratio']:.3f}")
    
    print("\\n" + "="*80)
    print("CONCLUSÕES")
    print("="*80)
    
    print(f"\\n💡 INSIGHTS:")
    print(f"   • Técnica mais eficaz: {best_technique}")
    print(f"   • Melhoria no F1-Score: {best_overall['best_result']['f1']:.3f} vs {all_results['Baseline']['best_result']['f1']:.3f}")
    print(f"   • Balanceamento mostrou {'MELHORIA' if best_overall['best_result']['f1'] > all_results['Baseline']['best_result']['f1'] else 'POUCA DIFERENÇA'}")
    
    print(f"\\n📁 Arquivos gerados:")
    print(f"   • comparacao_tecnicas_balanceamento.png")
    print(f"   • melhor_resultado_balanceamento.png")

except Exception as e:
    print(f"\\n❌ ERRO: {e}")
    import traceback
    traceback.print_exc()

print("\\n🎯 Análise de balanceamento finalizada!")
