# install_dependencies.py - Script para instalação automática de dependências

import subprocess
import sys
import os

def install_package(package):
    """Instala um pacote usando pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"[OK] {package} instalado com sucesso!")
        return True
    except subprocess.CalledProcessError:
        print(f"[ERRO] Erro ao instalar {package}")
        return False

def check_package(package):
    """Verifica se um pacote está instalado"""
    try:
        __import__(package)
        return True
    except ImportError:
        return False

def main():
    """Função principal"""
    print("INSTALADOR DE DEPENDÊNCIAS - PROJETO MACHINE LEARNING")
    print("="*60)
    
    # Lista de pacotes necessários
    packages = {
        'pandas': 'pandas>=1.3.0',
        'numpy': 'numpy>=1.21.0',
        'sklearn': 'scikit-learn>=1.0.0',
        'matplotlib': 'matplotlib>=3.5.0',
        'seaborn': 'seaborn>=0.11.0',
        'scipy': 'scipy>=1.7.0'
    }
    
    print("\nVerificando dependências...")
    
    to_install = []
    
    for package_import, package_pip in packages.items():
        if check_package(package_import):
            print(f"[OK] {package_import} já está instalado")
        else:
            print(f"[FALTA] {package_import} não encontrado")
            to_install.append(package_pip)
    
    if not to_install:
        print("\nTodas as dependências já estão instaladas!")
        return
    
    print(f"\nInstalando {len(to_install)} pacotes...")
    
    success_count = 0
    for package in to_install:
        if install_package(package):
            success_count += 1
    
    print(f"\nResumo:")
    print(f"• Pacotes instalados com sucesso: {success_count}")
    print(f"• Pacotes com erro: {len(to_install) - success_count}")
    
    if success_count == len(to_install):
        print("\nInstalação completa! Você pode executar os scripts agora.")
    else:
        print("\nAlguns pacotes falharam. Tente instalar manualmente:")
        print("pip install pandas numpy scikit-learn matplotlib seaborn scipy")

if __name__ == "__main__":
    main()
