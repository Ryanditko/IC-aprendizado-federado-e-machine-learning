"""
📋 GUIA COMPLETO: Análise Dataset Cybersecurity
Este script mostra todas as opções disponíveis para análise
"""

import os
import sys

def show_menu():
    """Mostra menu de opções"""
    print("🚀 ANÁLISE DATASET CYBERSECURITY - OPÇÕES DISPONÍVEIS")
    print("=" * 60)
    print()
    print("1️⃣  Demonstração (dados simulados) - RECOMENDADO PARA COMEÇAR")
    print("    📝 Usa dados artificiais para mostrar como funciona")
    print("    ⚡ Execução rápida, sem necessidade de credenciais")
    print("    🎯 Comando: python demo_cybersecurity.py")
    print()
    print("2️⃣  Análise Completa (dados reais da Kaggle)")
    print("    📊 Baixa dataset real de cybersecurity")
    print("    🔐 Requer credenciais Kaggle configuradas")
    print("    🎯 Comando: python cybersecurity_analysis.py")
    print()
    print("3️⃣  Testar Ambiente")
    print("    🧪 Verifica se todas dependências estão instaladas")
    print("    🎯 Comando: python test_cybersecurity_env.py")
    print()
    print("4️⃣  Configurar Ambiente")
    print("    📦 Instala todas as dependências necessárias")
    print("    🎯 Comando: python setup_cybersecurity_env.py")
    print()
    print("=" * 60)

def show_file_structure():
    """Mostra estrutura dos arquivos"""
    print("📁 ESTRUTURA DOS ARQUIVOS:")
    print("=" * 40)
    
    files_info = {
        "cybersecurity_analysis.py": "🔥 Script principal - Análise completa com dados reais",
        "demo_cybersecurity.py": "🎭 Demonstração com dados simulados",
        "test_cybersecurity_env.py": "🧪 Teste do ambiente",
        "setup_cybersecurity_env.py": "⚙️ Configuração do ambiente", 
        "avaliador_nao_supervisionado.py": "🔄 Classe para análise não supervisionada",
        "cybersecurity-datasets/": "📊 Pasta com resultados das análises"
    }
    
    for file, description in files_info.items():
        status = "✅" if os.path.exists(file) else "❌"
        print(f"{status} {file}")
        print(f"   {description}")
        print()

def show_requirements():
    """Mostra requisitos do sistema"""
    print("🔧 REQUISITOS:")
    print("=" * 20)
    print("✅ Python 3.8+")
    print("✅ Bibliotecas: pandas, numpy, scikit-learn, matplotlib, seaborn, scipy")
    print("✅ kagglehub (para dados reais)")
    print("✅ Conta Kaggle (apenas para dados reais)")
    print()

def show_kaggle_setup():
    """Mostra como configurar Kaggle"""
    print("🔐 CONFIGURAR CREDENCIAIS KAGGLE:")
    print("=" * 35)
    print("1. Acesse: https://www.kaggle.com/settings/account")
    print("2. Role até 'API' e clique 'Create New Token'")
    print("3. Baixe o arquivo kaggle.json")
    print("4. Windows: Mova para C:\\Users\\{seu_usuario}\\.kaggle\\kaggle.json")
    print("5. Linux/Mac: Mova para ~/.kaggle/kaggle.json")
    print()

def show_results_info():
    """Mostra informações sobre resultados"""
    print("📊 RESULTADOS GERADOS:")
    print("=" * 25)
    print("📋 Relatório em Markdown (.md)")
    print("   - Análise completa com estatísticas")
    print("   - Interpretação dos resultados")
    print("   - Recomendações técnicas")
    print()
    print("📈 Arquivo CSV para planilhas")
    print("   - Métricas de avaliação")
    print("   - Comparação entre técnicas")
    print("   - Dados para gráficos")
    print()
    print("🎯 Insights esperados:")
    print("   - Qualidade dos dados de cybersecurity")
    print("   - Padrões de ameaças identificados")
    print("   - Eficácia dos algoritmos testados")
    print("   - Anomalias nos dados")
    print()

def run_interactive():
    """Executa modo interativo"""
    while True:
        show_menu()
        print("\nEscolha uma opção:")
        print("1 - Executar demonstração")
        print("2 - Análise completa (requer Kaggle)")
        print("3 - Testar ambiente")
        print("4 - Configurar ambiente")
        print("5 - Mostrar estrutura de arquivos")
        print("6 - Mostrar requisitos")
        print("7 - Ajuda credenciais Kaggle")
        print("8 - Informações sobre resultados")
        print("0 - Sair")
        
        choice = input("\n👉 Digite sua opção (0-8): ").strip()
        
        if choice == "1":
            print("\n🎭 Executando demonstração...")
            os.system("python demo_cybersecurity.py")
            
        elif choice == "2":
            print("\n🔥 Executando análise completa...")
            print("⚠️ Certifique-se de ter configurado as credenciais Kaggle!")
            input("Pressione Enter para continuar ou Ctrl+C para cancelar...")
            os.system("python cybersecurity_analysis.py")
            
        elif choice == "3":
            print("\n🧪 Testando ambiente...")
            os.system("python test_cybersecurity_env.py")
            
        elif choice == "4":
            print("\n⚙️ Configurando ambiente...")
            os.system("python setup_cybersecurity_env.py")
            
        elif choice == "5":
            print("\n")
            show_file_structure()
            
        elif choice == "6":
            print("\n")
            show_requirements()
            
        elif choice == "7":
            print("\n")
            show_kaggle_setup()
            
        elif choice == "8":
            print("\n")
            show_results_info()
            
        elif choice == "0":
            print("👋 Até logo!")
            break
            
        else:
            print("❌ Opção inválida! Tente novamente.")
        
        input("\nPressione Enter para continuar...")
        print("\n" + "="*60 + "\n")

def main():
    """Função principal"""
    print("🎯 GUIA DE ANÁLISE CYBERSECURITY")
    print("Escolha como você quer executar:")
    print()
    
    if len(sys.argv) > 1:
        option = sys.argv[1].lower()
        
        if option in ['demo', 'd', '1']:
            os.system("python demo_cybersecurity.py")
        elif option in ['full', 'f', '2']:
            os.system("python cybersecurity_analysis.py") 
        elif option in ['test', 't', '3']:
            os.system("python test_cybersecurity_env.py")
        elif option in ['setup', 's', '4']:
            os.system("python setup_cybersecurity_env.py")
        else:
            print(f"❌ Opção '{option}' não reconhecida")
            show_menu()
    else:
        # Modo interativo
        run_interactive()

if __name__ == "__main__":
    main()
