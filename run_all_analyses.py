# run_all_analyses.py - Script para executar todas as análises

import os
import sys
import subprocess
import time

def run_script(script_path, script_name):
    """Executa um script Python e retorna o resultado"""
    print(f"\n{'='*60}")
    print(f"🚀 Executando: {script_name}")
    print(f"📁 Caminho: {script_path}")
    print('='*60)
    
    try:
        # Mudar para o diretório do script
        original_dir = os.getcwd()
        script_dir = os.path.dirname(script_path)
        script_file = os.path.basename(script_path)
        
        os.chdir(script_dir)
        
        # Executar o script
        result = subprocess.run([sys.executable, script_file], 
                              capture_output=True, text=True, timeout=300)
        
        # Voltar ao diretório original
        os.chdir(original_dir)
        
        if result.returncode == 0:
            print(f"✅ {script_name} executado com sucesso!")
            if result.stdout:
                print(f"📄 Saída:\n{result.stdout[:500]}...")
        else:
            print(f"❌ Erro ao executar {script_name}")
            if result.stderr:
                print(f"🚨 Erro: {result.stderr}")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout: {script_name} demorou mais de 5 minutos")
        return False
    except Exception as e:
        print(f"💥 Exceção ao executar {script_name}: {e}")
        return False

def main():
    """Função principal"""
    print("🧠 EXECUTOR COMPLETO - PROJETO MACHINE LEARNING")
    print("="*60)
    print("Este script executará todas as análises em sequência:")
    print("• Aprendizado Supervisionado - Iris")
    print("• Aprendizado Não Supervisionado - Iris") 
    print("• Aprendizado Supervisionado - Penguin")
    print("• Aprendizado Não Supervisionado - Penguin")
    print("="*60)
    
    # Obter o diretório base do projeto
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Lista de scripts para executar
    scripts = [
        {
            'path': os.path.join(base_dir, 'doc', 'code', 'iris-dataset', 'aprendizado-supervisionado.py'),
            'name': 'Aprendizado Supervisionado - Iris'
        },
        {
            'path': os.path.join(base_dir, 'doc', 'code', 'iris-dataset', 'aprendizado-nao-supervisionado.py'),
            'name': 'Aprendizado Não Supervisionado - Iris'
        },
        {
            'path': os.path.join(base_dir, 'doc', 'code', 'penguin-dataset', 'aprendizado-supervisionado.py'),
            'name': 'Aprendizado Supervisionado - Penguin'
        },
        {
            'path': os.path.join(base_dir, 'doc', 'code', 'penguin-dataset', 'aprendizado-nao-supervisionado.py'),
            'name': 'Aprendizado Não Supervisionado - Penguin'
        }
    ]
    
    # Verificar se os arquivos existem
    print("\n🔍 Verificando arquivos...")
    missing_files = []
    for script in scripts:
        if os.path.exists(script['path']):
            print(f"✅ {script['name']}")
        else:
            print(f"❌ {script['name']} - Arquivo não encontrado: {script['path']}")
            missing_files.append(script['name'])
    
    if missing_files:
        print(f"\n🚨 Arquivos faltantes: {len(missing_files)}")
        print("Certifique-se de que todos os scripts estão na estrutura correta.")
        return
    
    # Perguntar se o usuário quer continuar
    response = input("\n❓ Deseja executar todas as análises? (s/n): ").lower().strip()
    if response not in ['s', 'sim', 'y', 'yes']:
        print("❌ Execução cancelada pelo usuário.")
        return
    
    print("\n🎬 Iniciando execução...")
    start_time = time.time()
    
    # Executar cada script
    results = []
    for i, script in enumerate(scripts, 1):
        print(f"\n📊 Progresso: {i}/{len(scripts)}")
        success = run_script(script['path'], script['name'])
        results.append({
            'name': script['name'],
            'success': success
        })
        
        # Pausa entre execuções
        if i < len(scripts):
            print("⏸️ Aguardando 3 segundos antes da próxima análise...")
            time.sleep(3)
    
    # Resumo final
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "="*60)
    print("🏁 EXECUÇÃO COMPLETA!")
    print("="*60)
    
    print(f"⏱️ Tempo total: {duration:.1f} segundos")
    
    successful = sum(1 for r in results if r['success'])
    print(f"📊 Resumo:")
    print(f"• Scripts executados com sucesso: {successful}/{len(results)}")
    print(f"• Scripts com erro: {len(results) - successful}/{len(results)}")
    
    print(f"\n📋 Detalhes:")
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"{status} {result['name']}")
    
    if successful == len(results):
        print("\n🎉 Todas as análises foram executadas com sucesso!")
        print("📈 Verifique os gráficos e resultados gerados.")
    else:
        print("\n⚠️ Algumas análises falharam.")
        print("💡 Dica: Execute os scripts individualmente para mais detalhes.")
    
    print("\n📚 Para mais informações, consulte o README.md")

if __name__ == "__main__":
    main()
