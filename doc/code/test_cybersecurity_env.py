"""
Script de teste para verificar se o ambiente está configurado corretamente
"""

import sys

def test_imports():
    """Testa se todas as bibliotecas necessárias podem ser importadas"""
    print("🧪 Testando importações...")
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy', 
        'matplotlib': 'matplotlib.pyplot',
        'seaborn': 'seaborn',
        'sklearn': 'sklearn',
        'scipy': 'scipy'
    }
    
    failed_imports = []
    
    for name, import_path in required_packages.items():
        try:
            __import__(import_path)
            print(f"   ✅ {name}")
        except ImportError as e:
            print(f"   ❌ {name}: {str(e)}")
            failed_imports.append(name)
    
    # Teste especial para kagglehub
    try:
        import kagglehub
        print(f"   ✅ kagglehub")
    except ImportError as e:
        print(f"   ❌ kagglehub: {str(e)}")
        failed_imports.append('kagglehub')
    
    return failed_imports

def test_avaliador():
    """Testa se o avaliador não supervisionado funciona"""
    print("\n🔬 Testando AvaliadorNaoSupervisionado...")
    
    try:
        import numpy as np
        from avaliador_nao_supervisionado import AvaliadorNaoSupervisionado
        
        # Criar dados de teste
        test_data = np.random.rand(100, 4)
        avaliador = AvaliadorNaoSupervisionado(test_data)
        
        print("   ✅ AvaliadorNaoSupervisionado inicializado com sucesso")
        return True
        
    except Exception as e:
        print(f"   ❌ Erro no AvaliadorNaoSupervisionado: {str(e)}")
        return False

def main():
    """Função principal de teste"""
    print("🚀 TESTE DO AMBIENTE CYBERSECURITY")
    print("=" * 50)
    
    # Teste de importações
    failed_imports = test_imports()
    
    # Teste do avaliador
    avaliador_ok = test_avaliador()
    
    print("\n" + "=" * 50)
    print("📋 RESULTADO DOS TESTES:")
    
    if not failed_imports and avaliador_ok:
        print("✅ Todos os testes passaram! Ambiente configurado corretamente.")
        print("🎯 Você pode executar: python cybersecurity_analysis.py")
    else:
        print("❌ Alguns problemas foram encontrados:")
        
        if failed_imports:
            print(f"   Bibliotecas com problema: {', '.join(failed_imports)}")
            print("   Execute: python setup_cybersecurity_env.py")
        
        if not avaliador_ok:
            print("   Problema com AvaliadorNaoSupervisionado")
            print("   Verifique se o arquivo está no diretório correto")

if __name__ == "__main__":
    main()
