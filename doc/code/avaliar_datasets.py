"""
Script para Avaliação Automática dos Datasets Iris e Penguins
Gera resultados formatados para preenchimento da planilha de avaliação.
"""

import pandas as pd
import numpy as np
from avaliador_nao_supervisionado import AvaliadorNaoSupervisionado
import os

def carregar_iris():
    """Carrega e prepara o dataset Iris."""
    caminho_iris = r"c:\Users\Administrador\Iniciação-cientifica\doc\code\iris-dataset\iris.csv"
    
    if os.path.exists(caminho_iris):
        dados = pd.read_csv(caminho_iris)
        # Remove a coluna de espécie se existir
        colunas_numericas = dados.select_dtypes(include=[np.number]).columns
        dados_numericos = dados[colunas_numericas]
        print(f"Dataset Iris carregado: {dados_numericos.shape}")
        return dados_numericos
    else:
        print("Arquivo iris.csv não encontrado. Usando dataset do sklearn.")
        from sklearn.datasets import load_iris
        iris = load_iris()
        return pd.DataFrame(iris.data, columns=iris.feature_names)

def carregar_penguins():
    """Carrega e prepara o dataset Penguins."""
    caminho_penguins = r"c:\Users\Administrador\Iniciação-cientifica\doc\code\penguin-dataset\penguins.csv"
    
    if os.path.exists(caminho_penguins):
        dados = pd.read_csv(caminho_penguins)
        # Remove colunas não numéricas e trata valores faltantes
        colunas_numericas = dados.select_dtypes(include=[np.number]).columns
        dados_numericos = dados[colunas_numericas].dropna()
        print(f"Dataset Penguins carregado: {dados_numericos.shape}")
        return dados_numericos
    else:
        print("Arquivo penguins.csv não encontrado.")
        return None

def avaliar_dataset(nome_dataset, dados):
    """
    Avalia um dataset completo e retorna resultados formatados.
    """
    print(f"\n{'='*60}")
    print(f"AVALIAÇÃO COMPLETA - DATASET {nome_dataset.upper()}")
    print(f"{'='*60}")
    
    # Criar avaliador
    avaliador = AvaliadorNaoSupervisionado(dados)
    
    # Executar todas as avaliações
    print("Executando avaliações...")
    avaliador.avaliar_agrupamento_particional(k_min=2, k_max=8)
    avaliador.avaliar_agrupamento_hierarquico()
    avaliador.avaliar_reducao_dimensionalidade()
    avaliador.avaliar_deteccao_anomalias()
    
    # Gerar relatório completo
    relatorio = avaliador.gerar_relatorio_completo()
    
    # Exportar para planilha específica
    nome_arquivo = f"{nome_dataset}_avaliacao_completa.xlsx"
    avaliador.exportar_para_planilha(nome_arquivo)
    
    return avaliador, relatorio

def gerar_tabela_resumo_planilha(relatorios):
    """
    Gera uma tabela resumo formatada para preenchimento da planilha.
    """
    print(f"\n{'='*80}")
    print("TABELA RESUMO PARA PLANILHA")
    print(f"{'='*80}")
    
    # Criar DataFrame para o resumo
    resumo_data = []
    
    for dataset, relatorio in relatorios.items():
        melhores = relatorio['melhores_resultados']
        
        # K-means
        resumo_data.append({
            'Dataset': dataset,
            'Técnica': 'K-means',
            'Métrica': 'Coeficiente de Silhueta',
            'Valor': round(melhores['agrupamento_particional']['silhueta_score'], 3),
            'Parâmetros': f"k={melhores['agrupamento_particional']['melhor_k']}",
            'Interpretação': 'Valores próximos de 1 indicam clusters bem definidos'
        })
        
        # Clustering Hierárquico
        resumo_data.append({
            'Dataset': dataset,
            'Técnica': 'Clustering Hierárquico',
            'Métrica': 'Coeficiente Cofenético',
            'Valor': round(melhores['agrupamento_hierarquico']['coeficiente_cofenetico'], 3),
            'Parâmetros': melhores['agrupamento_hierarquico']['melhor_metodo'],
            'Interpretação': 'Valores próximos de 1 indicam boa preservação das distâncias'
        })
        
        # PCA
        resumo_data.append({
            'Dataset': dataset,
            'Técnica': 'PCA',
            'Métrica': 'Componentes para 95% variância',
            'Valor': melhores['reducao_dimensionalidade']['componentes_95_pct'],
            'Parâmetros': f"Var. acum.: {round(melhores['reducao_dimensionalidade']['variancia_total_95'], 3)}",
            'Interpretação': 'Menor número de componentes indica dados mais compressíveis'
        })
    
    df_resumo = pd.DataFrame(resumo_data)
    
    # Salvar tabela resumo
    df_resumo.to_excel('resumo_avaliacoes_planilha.xlsx', index=False)
    
    # Exibir tabela formatada
    print("\nTABELA RESUMO:")
    print("-" * 120)
    print(f"{'Dataset':<10} {'Técnica':<20} {'Métrica':<25} {'Valor':<8} {'Parâmetros':<15} {'Interpretação':<40}")
    print("-" * 120)
    
    for _, row in df_resumo.iterrows():
        print(f"{row['Dataset']:<10} {row['Técnica']:<20} {row['Métrica']:<25} {row['Valor']:<8} {row['Parâmetros']:<15} {row['Interpretação']:<40}")
    
    return df_resumo

def gerar_metricas_detalhadas(relatorios):
    """
    Gera métricas detalhadas para análise aprofundada.
    """
    print(f"\n{'='*80}")
    print("MÉTRICAS DETALHADAS POR TÉCNICA")
    print(f"{'='*80}")
    
    for dataset, relatorio in relatorios.items():
        print(f"\n--- {dataset.upper()} ---")
        
        # Agrupamento Particional detalhado
        if 'agrupamento_particional' in relatorio['detalhes_completos']:
            part_data = relatorio['detalhes_completos']['agrupamento_particional']
            print("\nK-MEANS - Análise por K:")
            for i, k in enumerate(part_data['k']):
                print(f"  K={k}: Silhueta={part_data['silhueta'][i]:.3f}, "
                      f"Davies-Bouldin={part_data['davies_bouldin'][i]:.3f}, "
                      f"Inércia={part_data['inercia'][i]:.0f}")
        
        # Agrupamento Hierárquico detalhado
        if 'agrupamento_hierarquico' in relatorio['detalhes_completos']:
            hier_data = relatorio['detalhes_completos']['agrupamento_hierarquico']
            print(f"\nCLUSTERING HIERÁRQUICO:")
            for metodo, resultados in hier_data.items():
                print(f"  {metodo}: Coef. Cofenético = {resultados['coeficiente_cofenetico']:.3f}")
        
        # Redução de Dimensionalidade detalhada
        if 'reducao_dimensionalidade' in relatorio['detalhes_completos']:
            pca_data = relatorio['detalhes_completos']['reducao_dimensionalidade']
            print(f"\nPCA - VARIÂNCIA EXPLICADA:")
            print(f"  80% da variância: {pca_data['componentes_80_pct']} componentes")
            print(f"  90% da variância: {pca_data['componentes_90_pct']} componentes")
            print(f"  95% da variância: {pca_data['componentes_95_pct']} componentes")
            
            # Primeiros 5 componentes
            for i in range(min(5, len(pca_data['variancia_explicada']))):
                var_exp = pca_data['variancia_explicada'][i]
                var_acum = pca_data['variancia_acumulada'][i]
                print(f"  Componente {i+1}: {var_exp:.3f} individual, {var_acum:.3f} acumulada")

def main():
    """Função principal que executa toda a análise."""
    print("SISTEMA DE AVALIAÇÃO AUTOMÁTICA")
    print("Datasets: Iris e Penguins")
    print("="*50)
    
    relatorios = {}
    
    # Avaliar Iris
    print("Carregando dataset Iris...")
    dados_iris = carregar_iris()
    if dados_iris is not None:
        avaliador_iris, relatorio_iris = avaliar_dataset("iris", dados_iris)
        relatorios['Iris'] = relatorio_iris
    
    # Avaliar Penguins
    print("\nCarregando dataset Penguins...")
    dados_penguins = carregar_penguins()
    if dados_penguins is not None:
        avaliador_penguins, relatorio_penguins = avaliar_dataset("penguins", dados_penguins)
        relatorios['Penguins'] = relatorio_penguins
    
    # Gerar resumo para planilha
    if relatorios:
        df_resumo = gerar_tabela_resumo_planilha(relatorios)
        gerar_metricas_detalhadas(relatorios)
        
        print(f"\n{'='*80}")
        print("ARQUIVOS GERADOS:")
        print(f"{'='*80}")
        print("- resumo_avaliacoes_planilha.xlsx (Tabela resumo)")
        if 'Iris' in relatorios:
            print("- iris_avaliacao_completa.xlsx (Detalhes Iris)")
        if 'Penguins' in relatorios:
            print("- penguins_avaliacao_completa.xlsx (Detalhes Penguins)")
        
        print(f"\n{'='*80}")
        print("INSTRUÇÕES PARA PREENCHIMENTO DA PLANILHA:")
        print(f"{'='*80}")
        print("1. Abra o arquivo 'resumo_avaliacoes_planilha.xlsx'")
        print("2. Copie os valores da coluna 'Valor' para sua planilha de resultados")
        print("3. Use as interpretações fornecidas para análise qualitativa")
        print("4. Consulte os arquivos detalhados para análises mais profundas")
        
        return relatorios
    else:
        print("Nenhum dataset foi carregado com sucesso.")
        return None

if __name__ == "__main__":
    relatorios = main()
