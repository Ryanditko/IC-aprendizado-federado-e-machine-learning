"""
Script para baixar automaticamente o dataset de Cyber Threat Detection do Kaggle
Autor: Projeto Iniciação Científica - Faculdade Impacta
Data: Outubro 2025
"""

import os
import sys
import shutil

print("="*70)
print("DOWNLOAD AUTOMÁTICO - CYBER THREAT DETECTION DATASET")
print("="*70)
print()

# Verificar se kagglehub está instalado
try:
    import kagglehub
    print("✓ kagglehub instalado")
except ImportError:
    print("❌ kagglehub não encontrado!")
    print("\nInstalando kagglehub...")
    os.system("pip install kagglehub")
    import kagglehub
    print("✓ kagglehub instalado com sucesso")

print("\n[1/3] Baixando dataset do Kaggle...")
print("Dataset: ramoliyafenil/text-based-cyber-threat-detection")
print("Aguarde, isso pode levar alguns minutos...\n")

try:
    # Download do dataset
    path = kagglehub.dataset_download("ramoliyafenil/text-based-cyber-threat-detection")
    print(f"\n✓ Download concluído!")
    print(f"Localização: {path}")
    
    # Listar arquivos baixados
    print("\n[2/3] Arquivos encontrados:")
    files = os.listdir(path)
    for file in files:
        file_path = os.path.join(path, file)
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"  • {file} ({size_mb:.2f} MB)")
    
    # Copiar para a pasta data/
    print("\n[3/3] Copiando para pasta data/...")
    
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    csv_found = False
    for file in files:
        if file.endswith('.csv'):
            source = os.path.join(path, file)
            destination = os.path.join(data_dir, file)
            
            shutil.copy2(source, destination)
            print(f"  ✓ {file} → data/{file}")
            csv_found = True
    
    if not csv_found:
        print("  ⚠ Nenhum arquivo CSV encontrado!")
    
    print("\n" + "="*70)
    print("✅ DATASET BAIXADO E INSTALADO COM SUCESSO!")
    print("="*70)
    print("\nPróximos passos:")
    print("  1. Verifique a pasta 'data/' para confirmar o arquivo CSV")
    print("  2. Execute o notebook: notebooks/cyber-outlier-detection/outlier_detection_analysis.ipynb")
    print("  3. Ou use diretamente no Python:")
    print()
    print("     import pandas as pd")
    print(f"     df = pd.read_csv('data/{files[0] if csv_found else 'arquivo.csv'}')")
    print("     print(df.head())")
    print()
    
except Exception as e:
    print("\n" + "="*70)
    print("❌ ERRO AO BAIXAR DATASET")
    print("="*70)
    print(f"\nErro: {str(e)}")
    print("\nPossíveis soluções:")
    print("  1. Verifique sua conexão com a internet")
    print("  2. Configure as credenciais do Kaggle (kaggle.json)")
    print("  3. Tente o download manual:")
    print("     https://www.kaggle.com/datasets/ramoliyafenil/text-based-cyber-threat-detection")
    print()
    sys.exit(1)
