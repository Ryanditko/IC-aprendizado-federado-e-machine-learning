"""
DOWNLOAD DO DATASET NSL-KDD
============================

Este script baixa o dataset NSL-KDD do Kaggle para análise de ataques de cybersegurança.

Dataset: NSL-KDD (Network Security Laboratory - Knowledge Discovery and Data Mining)
Fonte: https://www.kaggle.com/hassan06/nslkdd

Autor: Projeto de Iniciação Científica
Data: Novembro 2025
"""

import kagglehub
import os
import shutil
import pandas as pd

print("="*80)
print("DOWNLOAD DO DATASET NSL-KDD")
print("="*80)

# Configuração de diretórios
PROJECT_DIR = '..'
DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'nsl-kdd')

try:
    print("\n[1/3] Baixando dataset do Kaggle...")
    
    # Download do dataset
    path = kagglehub.dataset_download("hassan06/nslkdd")
    print(f"  ✓ Dataset baixado em: {path}")
    
    print("\n[2/3] Organizando arquivos...")
    
    # Criar diretório de destino se não existir
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Copiar arquivos para o diretório do projeto
    for file in os.listdir(path):
        src = os.path.join(path, file)
        dst = os.path.join(DATA_DIR, file)
        
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"  ✓ {file} copiado")
    
    print("\n[3/3] Verificando arquivos baixados...")
    
    # Listar arquivos baixados
    files = os.listdir(DATA_DIR)
    print(f"  ✓ Total de arquivos: {len(files)}")
    
    for file in files:
        file_path = os.path.join(DATA_DIR, file)
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"    • {file} ({size_mb:.2f} MB)")
        
        # Se for arquivo CSV, mostrar informações básicas
        if file.endswith('.csv'):
            try:
                df = pd.read_csv(file_path, nrows=5)
                print(f"      - Colunas: {len(df.columns)}")
                print(f"      - Primeiras colunas: {list(df.columns[:5])}")
            except Exception as e:
                print(f"      - Erro ao ler: {e}")
    
    print("\n" + "="*80)
    print("✅ DATASET NSL-KDD BAIXADO COM SUCESSO!")
    print("="*80)
    print(f"\n📁 Localização: {os.path.abspath(DATA_DIR)}")
    print("\n💡 Próximos passos:")
    print("  1. Execute o script de análise exploratória")
    print("  2. Execute os algoritmos de detecção de ataques")
    print("  3. Visualize os resultados nos notebooks")

except Exception as e:
    print(f"\n❌ ERRO: {e}")
    print("\n💡 Soluções possíveis:")
    print("  1. Verifique sua conexão com a internet")
    print("  2. Configure suas credenciais do Kaggle:")
    print("     - Crie uma conta no Kaggle")
    print("     - Vá em Account > API > Create New API Token")
    print("     - Salve o arquivo kaggle.json em ~/.kaggle/")
    print("  3. Instale o kagglehub: pip install kagglehub")
