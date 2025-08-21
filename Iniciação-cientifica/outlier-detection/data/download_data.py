"""
Script para download e preparação dos dados de peso e altura para detecção de outliers.

Conjunto de dados: Weight and Height Data for Outlier Detection
URL: https://www.kaggle.com/datasets/krishnaraj30/weight-and-height-data-outlier-detection

Como usar:
1. Instale o kaggle: pip install kaggle
2. Configure suas credenciais do Kaggle em ~/.kaggle/kaggle.json
3. Execute este script: python download_data.py
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def create_sample_data():
    """
    Cria dados sintéticos similares aos dados de peso e altura
    para demonstração das técnicas de detecção de outliers.
    """
    np.random.seed(42)
    
    # Gerando dados normais para altura (em cm) e peso (em kg)
    n_samples = 1000
    
    # Altura: distribuição normal com média 170cm e desvio padrão 10cm
    height_normal = np.random.normal(170, 10, n_samples)
    
    # Peso: correlacionado com altura + ruído
    weight_normal = 2.3 * height_normal - 220 + np.random.normal(0, 8, n_samples)
    
    # Adicionando outliers artificiais
    n_outliers = 50
    
    # Outliers de altura (muito altos ou muito baixos)
    height_outliers = np.concatenate([
        np.random.uniform(120, 140, n_outliers//2),  # Muito baixos
        np.random.uniform(210, 230, n_outliers//2)   # Muito altos
    ])
    
    # Outliers de peso (não correlacionados com altura)
    weight_outliers = np.concatenate([
        np.random.uniform(40, 50, n_outliers//2),    # Muito leves
        np.random.uniform(150, 180, n_outliers//2)   # Muito pesados
    ])
    
    # Combinando dados normais e outliers
    height = np.concatenate([height_normal, height_outliers])
    weight = np.concatenate([weight_normal, weight_outliers])
    
    # Criando labels (0 = normal, 1 = outlier)
    labels = np.concatenate([
        np.zeros(n_samples),  # Dados normais
        np.ones(n_outliers)   # Outliers
    ])
    
    # Criando DataFrame
    df = pd.DataFrame({
        'Height_cm': height,
        'Weight_kg': weight,
        'Is_Outlier': labels.astype(int)
    })
    
    # Embaralhando os dados
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

def download_kaggle_data():
    """
    Tenta baixar os dados do Kaggle se as credenciais estiverem configuradas.
    """
    try:
        import kaggle
        
        print("Baixando dados do Kaggle...")
        kaggle.api.dataset_download_files(
            'krishnaraj30/weight-and-height-data-outlier-detection',
            path='.',
            unzip=True
        )
        print("Dados baixados com sucesso!")
        return True
    except Exception as e:
        print(f"Erro ao baixar dados do Kaggle: {e}")
        print("Usando dados sintéticos...")
        return False

def main():
    """Função principal para preparação dos dados."""
    
    # Tentativa de download do Kaggle
    kaggle_success = download_kaggle_data()
    
    if not kaggle_success:
        # Criando dados sintéticos
        print("Criando dados sintéticos para demonstração...")
        df = create_sample_data()
        
        # Salvando os dados
        df.to_csv('weight_height_data.csv', index=False)
        print("Dados sintéticos criados e salvos em 'weight_height_data.csv'")
        
        # Mostrando estatísticas básicas
        print("\nEstatísticas dos dados criados:")
        print(df.describe())
        print(f"\nTotal de outliers: {df['Is_Outlier'].sum()}")
        print(f"Porcentagem de outliers: {df['Is_Outlier'].mean()*100:.2f}%")
    
    else:
        # Se o download foi bem-sucedido, carregando e analisando os dados
        files = [f for f in os.listdir('.') if f.endswith('.csv')]
        if files:
            df = pd.read_csv(files[0])
            print(f"Dados carregados de {files[0]}")
            print(df.head())
            print(df.info())

if __name__ == "__main__":
    main()
