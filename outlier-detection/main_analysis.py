"""
Script principal para análise de detecção de outliers.

Este script executa uma análise completa de detecção de outliers usando
múltiplas técnicas de aprendizado não supervisionado.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Adicionar o diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from outlier_detector import OutlierDetector
from model_evaluator import ModelEvaluator

# Configurações
plt.style.use('default')
sns.set_palette("husl")
np.random.seed(42)

def load_data():
    """Carrega os dados para análise."""
    
    data_path = os.path.join('data', 'weight_height_data.csv')
    
    if not os.path.exists(data_path):
        print("Dados não encontrados. Executando script de download...")
        
        # Executar script de download
        download_script = os.path.join('data', 'download_data.py')
        if os.path.exists(download_script):
            os.system(f'python "{download_script}"')
            
            # Mover o arquivo para o diretório correto
            if os.path.exists('weight_height_data.csv'):
                import shutil
                shutil.move('weight_height_data.csv', data_path)
        else:
            print("Script de download não encontrado!")
            return None
    
    try:
        df = pd.read_csv(data_path)
        print(f"Dados carregados com sucesso: {len(df)} registros")
        return df
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return None

def exploratory_analysis(df):
    """Realiza análise exploratória dos dados."""
    
    print("\n" + "="*50)
    print("ANÁLISE EXPLORATÓRIA DOS DADOS")
    print("="*50)
    
    # Informações básicas
    print("\nInformações do Dataset:")
    print(df.info())
    
    print("\nEstatísticas Descritivas:")
    print(df.describe())
    
    # Verificar valores nulos
    print(f"\nValores nulos:")
    print(df.isnull().sum())
    
    # Distribuição das variáveis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Análise Exploratória dos Dados', fontsize=16)
        
        # Histograma da primeira variável
        col1 = numeric_cols[0]
        axes[0, 0].hist(df[col1], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title(f'Distribuição - {col1}')
        axes[0, 0].set_xlabel(col1)
        axes[0, 0].set_ylabel('Frequência')
        
        # Histograma da segunda variável
        col2 = numeric_cols[1]
        axes[0, 1].hist(df[col2], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title(f'Distribuição - {col2}')
        axes[0, 1].set_xlabel(col2)
        axes[0, 1].set_ylabel('Frequência')
        
        # Scatter plot
        axes[1, 0].scatter(df[col1], df[col2], alpha=0.6)
        axes[1, 0].set_title(f'{col1} vs {col2}')
        axes[1, 0].set_xlabel(col1)
        axes[1, 0].set_ylabel(col2)
        
        # Boxplot conjunto
        df_numeric = df[numeric_cols].copy()
        df_normalized = (df_numeric - df_numeric.mean()) / df_numeric.std()
        axes[1, 1].boxplot([df_normalized[col] for col in numeric_cols], 
                          labels=numeric_cols)
        axes[1, 1].set_title('Boxplots Normalizados')
        axes[1, 1].set_ylabel('Valores Normalizados')
        
        plt.tight_layout()
        
        # Salvar gráfico
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/exploratory_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return df

def run_outlier_detection(df):
    """Executa a detecção de outliers com múltiplas técnicas."""
    
    print("\n" + "="*50)
    print("DETECÇÃO DE OUTLIERS")
    print("="*50)
    
    # Preparar dados (remover coluna de labels se existir)
    data_for_detection = df.copy()
    true_labels = None
    
    if 'Is_Outlier' in df.columns:
        true_labels = df['Is_Outlier'].values.astype(bool)
        data_for_detection = df.drop('Is_Outlier', axis=1)
        print(f"Labels verdadeiros encontrados: {np.sum(true_labels)} outliers de {len(true_labels)} pontos")
    
    # Inicializar detector
    detector = OutlierDetector(data_for_detection)
    
    # Executar todos os métodos
    results = detector.run_all_methods()
    
    # Comparar métodos
    print("\nComparação dos Métodos:")
    comparison = detector.compare_methods()
    print(comparison.to_string(index=False))
    
    return detector, results, true_labels

def evaluate_models(detector, true_labels):
    """Avalia os modelos de detecção de outliers."""
    
    print("\n" + "="*50)
    print("AVALIAÇÃO DOS MODELOS")
    print("="*50)
    
    # Inicializar avaliador
    evaluator = ModelEvaluator(true_labels)
    
    # Adicionar predições
    for method_name, result in detector.results.items():
        evaluator.add_predictions(method_name, result['outliers'])
    
    # Calcular métricas
    metrics = evaluator.calculate_metrics()
    print("\nMétricas de Avaliação:")
    print(metrics.to_string(index=False))
    
    # Análise de concordância
    print("\nAnálise de Concordância entre Métodos:")
    agreement = evaluator.analyze_method_agreement()
    print(agreement.round(3))
    
    # Análise ensemble
    if len(detector.results) > 1:
        print("\nAnálise Ensemble:")
        ensemble_pred, vote_ratio = evaluator.ensemble_prediction()
        print(f"Ensemble detectou {np.sum(ensemble_pred)} outliers")
        
        if true_labels is not None:
            from sklearn.metrics import classification_report
            print("\nRelatório do Ensemble:")
            print(classification_report(true_labels, ensemble_pred, 
                                       target_names=['Normal', 'Outlier']))
    
    return evaluator, metrics

def generate_visualizations(detector, evaluator, true_labels):
    """Gera visualizações dos resultados."""
    
    print("\n" + "="*50)
    print("GERANDO VISUALIZAÇÕES")
    print("="*50)
    
    os.makedirs('results', exist_ok=True)
    
    # 1. Visualizar outliers para cada método
    for method_name in detector.results.keys():
        print(f"Gerando visualização para {method_name}...")
        fig = detector.plot_outliers(method_name)
        if fig:
            plt.savefig(f'results/outliers_{method_name}.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    # 2. Matrizes de confusão (se labels estiverem disponíveis)
    if true_labels is not None:
        print("Gerando matrizes de confusão...")
        evaluator.plot_confusion_matrices()
        plt.savefig('results/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 3. Concordância entre métodos
    print("Gerando heatmap de concordância...")
    evaluator.plot_method_agreement()
    plt.savefig('results/method_agreement.png', dpi=300, bbox_inches='tight')
    
    # 4. Análise ensemble
    if len(detector.results) > 1:
        print("Gerando análise ensemble...")
        evaluator.plot_ensemble_analysis()
        plt.savefig('results/ensemble_analysis.png', dpi=300, bbox_inches='tight')
    
    print("Visualizações salvas na pasta 'results/'")

def save_results(detector, evaluator, metrics):
    """Salva os resultados em arquivos."""
    
    print("\n" + "="*50)
    print("SALVANDO RESULTADOS")
    print("="*50)
    
    os.makedirs('results', exist_ok=True)
    
    # Salvar comparação de métodos
    comparison = detector.compare_methods()
    comparison.to_csv('results/method_comparison.csv', index=False)
    print("Comparação de métodos salva em 'results/method_comparison.csv'")
    
    # Salvar métricas (se disponíveis)
    if metrics is not None and not metrics.empty:
        metrics.to_csv('results/evaluation_metrics.csv', index=False)
        print("Métricas de avaliação salvas em 'results/evaluation_metrics.csv'")
    
    # Salvar concordância entre métodos
    agreement = evaluator.analyze_method_agreement()
    agreement.to_csv('results/method_agreement.csv')
    print("Concordância entre métodos salva em 'results/method_agreement.csv'")
    
    # Salvar predições detalhadas
    predictions_df = detector.data.copy()
    for method_name, result in detector.results.items():
        predictions_df[f'{method_name}_outlier'] = result['outliers'].astype(int)
    
    predictions_df.to_csv('results/detailed_predictions.csv', index=False)
    print("Predições detalhadas salvas em 'results/detailed_predictions.csv'")

def main():
    """Função principal."""
    
    print("="*70)
    print("ANÁLISE DE DETECÇÃO DE OUTLIERS EM APRENDIZADO NÃO SUPERVISIONADO")
    print("="*70)
    
    # 1. Carregar dados
    df = load_data()
    if df is None:
        print("Erro: Não foi possível carregar os dados.")
        return
    
    # 2. Análise exploratória
    df = exploratory_analysis(df)
    
    # 3. Detecção de outliers
    detector, results, true_labels = run_outlier_detection(df)
    
    # 4. Avaliação dos modelos
    evaluator, metrics = evaluate_models(detector, true_labels)
    
    # 5. Gerar visualizações
    generate_visualizations(detector, evaluator, true_labels)
    
    # 6. Salvar resultados
    save_results(detector, evaluator, metrics)
    
    print("\n" + "="*70)
    print("ANÁLISE CONCLUÍDA COM SUCESSO!")
    print("="*70)
    print("\nResultados salvos em:")
    print("- results/method_comparison.csv")
    print("- results/evaluation_metrics.csv")
    print("- results/method_agreement.csv")
    print("- results/detailed_predictions.csv")
    print("- results/*.png (gráficos)")

if __name__ == "__main__":
    main()
