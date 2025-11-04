"""
TESTE RÁPIDO - NSL-KDD DATASET
==============================

Script para verificar se o ambiente está configurado corretamente
para executar a análise do dataset NSL-KDD.

Autor: Projeto de Iniciação Científica
Data: Novembro 2025
"""

import sys
import os

def test_imports():
    """Testa se todas as bibliotecas necessárias estão instaladas"""
    print("🔍 Testando importações...")
    
    try:
        import pandas as pd
        print("  ✅ pandas")
    except ImportError:
        print("  ❌ pandas - Execute: pip install pandas")
        return False
    
    try:
        import numpy as np
        print("  ✅ numpy")
    except ImportError:
        print("  ❌ numpy - Execute: pip install numpy")
        return False
    
    try:
        import sklearn
        print("  ✅ scikit-learn")
    except ImportError:
        print("  ❌ scikit-learn - Execute: pip install scikit-learn")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("  ✅ matplotlib")
    except ImportError:
        print("  ❌ matplotlib - Execute: pip install matplotlib")
        return False
    
    try:
        import seaborn as sns
        print("  ✅ seaborn")
    except ImportError:
        print("  ❌ seaborn - Execute: pip install seaborn")
        return False
    
    try:
        import kagglehub
        print("  ✅ kagglehub")
    except ImportError:
        print("  ❌ kagglehub - Execute: pip install kagglehub")
        return False
    
    return True

def test_directories():
    """Verifica se os diretórios necessários existem"""
    print("\n📁 Verificando estrutura de diretórios...")
    
    dirs_to_check = [
        '../data/nsl-kdd',
        '../notebooks/nsl-kdd',
        '../notebooks/nsl-kdd/output-images',
        '../notebooks/nsl-kdd/results'
    ]
    
    all_exist = True
    for dir_path in dirs_to_check:
        if os.path.exists(dir_path):
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path} - Será criado automaticamente")
            all_exist = False
    
    return all_exist

def test_kaggle_config():
    """Verifica se o Kaggle está configurado"""
    print("\n🔑 Verificando configuração do Kaggle...")
    
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    
    if os.path.exists(kaggle_json):
        print("  ✅ Arquivo kaggle.json encontrado")
        return True
    else:
        print("  ⚠️  kaggle.json não encontrado")
        print("     Para baixar o dataset automaticamente:")
        print("     1. Crie uma conta no Kaggle")
        print("     2. Vá em Account > API > Create New API Token")
        print("     3. Salve o arquivo kaggle.json em ~/.kaggle/")
        return False

def main():
    """Função principal do teste"""
    print("="*60)
    print("TESTE DE CONFIGURAÇÃO - NSL-KDD DATASET")
    print("="*60)
    
    # Testes
    imports_ok = test_imports()
    dirs_ok = test_directories()
    kaggle_ok = test_kaggle_config()
    
    # Resumo
    print("\n" + "="*60)
    print("RESUMO DOS TESTES")
    print("="*60)
    
    if imports_ok:
        print("✅ Bibliotecas: Todas instaladas")
    else:
        print("❌ Bibliotecas: Algumas faltando")
    
    if dirs_ok:
        print("✅ Diretórios: Estrutura completa")
    else:
        print("⚠️  Diretórios: Serão criados automaticamente")
    
    if kaggle_ok:
        print("✅ Kaggle: Configurado")
    else:
        print("⚠️  Kaggle: Configuração necessária para download automático")
    
    # Status geral
    if imports_ok:
        print("\n🎉 AMBIENTE PRONTO PARA EXECUÇÃO!")
        print("\n📋 Próximos passos:")
        print("  1. Execute: python ../downloads/download_nsl_kdd_dataset.py")
        print("  2. Execute: python deteccao-ataques-nsl-kdd.py")
        print("  3. Ou abra o notebook: notebooks/nsl-kdd/nsl_kdd_evaluation.ipynb")
    else:
        print("\n❌ AMBIENTE PRECISA DE CONFIGURAÇÃO")
        print("   Execute: pip install -r ../dependencies/requirements.txt")

if __name__ == "__main__":
    main()
