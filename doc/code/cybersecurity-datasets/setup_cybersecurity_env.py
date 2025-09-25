"""
Script para instalar dependências necessárias para análise do dataset cybersecurity
"""

import subprocess
import sys
import os

def install_package(package_name):
    """Instala um pacote Python usando pip"""
    try:
        print(f"📦 Instalando {package_name}...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", package_name, "--upgrade"
        ], capture_output=True, text=True, check=True)
        print(f"✅ {package_name} instalado com sucesso!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao instalar {package_name}: {e.stderr}")
        return False

def setup_kaggle_credentials():
    """Verifica e orienta sobre configuração das credenciais Kaggle"""
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_file = os.path.join(kaggle_dir, "kaggle.json")
    
    if os.path.exists(kaggle_file):
        print("✅ Credenciais Kaggle encontradas!")
        return True
    else:
        print("⚠️ Credenciais Kaggle não encontradas!")
        print("🔧 Para configurar:")
        print("   1. Vá para https://www.kaggle.com/settings/account")
        print("   2. Role para baixo até 'API' e clique em 'Create New Token'")
        print("   3. Baixe o arquivo kaggle.json")
        print("   4. Mova o arquivo para:")
        print(f"      {kaggle_file}")
        print("   5. No Windows, garanta que a pasta .kaggle existe em seu diretório home")
        return False

def main():
    """Instalação completa das dependências"""
    print("🚀 CONFIGURAÇÃO DO AMBIENTE PARA ANÁLISE CYBERSECURITY")
    print("=" * 60)
    
    # Pacotes essenciais
    packages = [
        "kagglehub>=0.2.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0"
    ]
    
    print("📋 Instalando dependências Python...")
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n📊 Resultado da instalação: {success_count}/{len(packages)} pacotes")
    
    if success_count == len(packages):
        print("✅ Todas as dependências foram instaladas com sucesso!")
    else:
        print("⚠️ Algumas dependências falharam. Tente instalar manualmente:")
        print("   pip install -r requirements.txt")
    
    print("\n🔐 Verificando credenciais Kaggle...")
    setup_kaggle_credentials()
    
    print("\n" + "=" * 60)
    print("🎯 PRÓXIMOS PASSOS:")
    print("1. Configure as credenciais Kaggle (se necessário)")
    print("2. Execute: python cybersecurity_analysis.py")
    print("3. Os resultados serão salvos em cybersecurity-datasets/")

if __name__ == "__main__":
    main()
