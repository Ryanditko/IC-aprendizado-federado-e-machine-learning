"""
EXECUTOR NOTEBOOK NSL-KDD - DETECÇÃO DE ATAQUES
===============================================

Este script executa a análise de detecção de ataques no dataset NSL-KDD
seguindo o padrão dos outros notebooks do projeto.

Autor: Projeto de Iniciação Científica
Data: Novembro 2025
"""

import subprocess
import sys
import os

def run_nsl_kdd_analysis():
    """
    Executa o script de análise de detecção de ataques NSL-KDD
    """
    print("="*80)
    print("EXECUTOR - ANÁLISE NSL-KDD")
    print("="*80)
    
    script_path = '../scripts-datasets/nsl-kdd/deteccao-ataques-nsl-kdd.py'
    
    if not os.path.exists(script_path):
        print(f"❌ Erro: Script não encontrado em {script_path}")
        return False
    
    try:
        print("\n🚀 Iniciando análise de detecção de ataques...")
        print("  (Isso pode levar alguns minutos...)")
        
        # Executar o script
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, cwd=os.path.dirname(script_path))
        
        if result.returncode == 0:
            print("✅ Análise concluída com sucesso!")
            print("\n📊 Output:")
            print(result.stdout)
            
            return True
        else:
            print("❌ Erro durante a execução:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return False

if __name__ == "__main__":
    success = run_nsl_kdd_analysis()
    
    if success:
        print("\n" + "="*80)
        print("🎉 ANÁLISE NSL-KDD EXECUTADA COM SUCESSO!")
        print("="*80)
        print("\n📁 Verifique os arquivos gerados em:")
        print("  • notebooks/nsl-kdd/output-images/")
        print("  • notebooks/nsl-kdd/results/")
    else:
        print("\n" + "="*80)
        print("❌ FALHA NA EXECUÇÃO")
        print("="*80)
        print("\n💡 Verifique se:")
        print("  1. O dataset foi baixado (execute download_nsl_kdd_dataset.py)")
        print("  2. As dependências estão instaladas")
        print("  3. Os diretórios existem")
