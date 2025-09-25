"""
Demo: Aprendizado Não Supervisionado em Cybersecurity
Demonstra as três técnicas principais com dados simulados:
1. K-Means (Agrupamento Particional)
2. AGNES (Agrupamento Hierárquico) 
3. PCA (Redução de Dimensionalidade)
"""

import pandas as pd
import numpy as np
from unsupervised_cybersecurity_analysis import UnsupervisedCybersecurityAnalyzer
import os
import warnings
warnings.filterwarnings('ignore')

def create_cybersecurity_dataset():
    """Cria dataset simulado de cybersecurity para demonstração"""
    print("📊 Criando dataset simulado de cybersecurity...")
    
    np.random.seed(42)
    n_samples = 1500
    
    # Simular diferentes tipos de tráfego de rede
    # Grupo 1: Tráfego Normal
    normal_size = int(0.6 * n_samples)  # 60%
    normal_data = {
        'packet_size': np.random.normal(1500, 200, normal_size),
        'connection_duration': np.random.exponential(5, normal_size),
        'bytes_sent': np.random.lognormal(7, 1, normal_size),
        'bytes_received': np.random.lognormal(6.5, 1, normal_size),
        'num_failed_logins': np.random.poisson(0.1, normal_size),
        'port_scan_count': np.random.poisson(0.2, normal_size),
        'protocol_violations': np.random.poisson(0.1, normal_size),
        'network_delay': np.random.normal(50, 10, normal_size)
    }
    
    # Grupo 2: Atividade Suspeita
    suspicious_size = int(0.3 * n_samples)  # 30%
    suspicious_data = {
        'packet_size': np.random.normal(2000, 400, suspicious_size),
        'connection_duration': np.random.exponential(15, suspicious_size),
        'bytes_sent': np.random.lognormal(8, 1.5, suspicious_size),
        'bytes_received': np.random.lognormal(7, 1.2, suspicious_size),
        'num_failed_logins': np.random.poisson(2, suspicious_size),
        'port_scan_count': np.random.poisson(3, suspicious_size),
        'protocol_violations': np.random.poisson(1, suspicious_size),
        'network_delay': np.random.normal(100, 25, suspicious_size)
    }
    
    # Grupo 3: Ataques Maliciosos
    malicious_size = n_samples - normal_size - suspicious_size  # 10%
    malicious_data = {
        'packet_size': np.random.normal(3000, 600, malicious_size),
        'connection_duration': np.random.exponential(30, malicious_size),
        'bytes_sent': np.random.lognormal(9, 2, malicious_size),
        'bytes_received': np.random.lognormal(8, 1.8, malicious_size),
        'num_failed_logins': np.random.poisson(8, malicious_size),
        'port_scan_count': np.random.poisson(15, malicious_size),
        'protocol_violations': np.random.poisson(5, malicious_size),
        'network_delay': np.random.normal(200, 50, malicious_size)
    }
    
    # Combinar todos os dados
    all_data = {}
    for feature in normal_data.keys():
        all_data[feature] = np.concatenate([
            normal_data[feature],
            suspicious_data[feature], 
            malicious_data[feature]
        ])
    
    # Criar labels verdadeiros (para validação, mas não usados na análise)
    true_labels = np.concatenate([
        np.zeros(normal_size),      # 0 = Normal
        np.ones(suspicious_size),   # 1 = Suspicious
        np.full(malicious_size, 2)  # 2 = Malicious
    ])
    
    # Embaralhar dados
    indices = np.random.permutation(n_samples)
    for feature in all_data.keys():
        all_data[feature] = all_data[feature][indices]
    true_labels = true_labels[indices]
    
    # Criar DataFrame
    df = pd.DataFrame(all_data)
    
    print(f"✅ Dataset criado com {n_samples} amostras")
    print(f"   Features: {list(df.columns)}")
    print(f"   Normal: {normal_size} ({normal_size/n_samples*100:.1f}%)")
    print(f"   Suspicious: {suspicious_size} ({suspicious_size/n_samples*100:.1f}%)")
    print(f"   Malicious: {malicious_size} ({malicious_size/n_samples*100:.1f}%)")
    
    return df, true_labels

class DemoUnsupervisedAnalyzer(UnsupervisedCybersecurityAnalyzer):
    """Versão demo do analisador não supervisionado"""
    
    def __init__(self, demo_data):
        super().__init__()
        self.data = demo_data
        self.dataset_path = "demo_data"
        
        # Informações básicas
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.results['dataset_info'] = {
            'total_samples': self.data.shape[0],
            'total_features': self.data.shape[1],
            'numeric_features': len(numeric_cols),
            'feature_names': list(numeric_cols),
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024**2
        }
    
    def download_dataset(self):
        """Override para usar dados simulados"""
        print("📥 Usando dados simulados para demonstração...")
        return True
    
    def load_and_explore_data(self):
        """Override para dados já carregados"""
        print("\n🔍 Explorando dados simulados de cybersecurity...")
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        print(f"📈 Informações do Dataset:")
        print(f"   Linhas: {self.data.shape[0]:,}")
        print(f"   Colunas: {self.data.shape[1]}")
        print(f"   Colunas numéricas: {len(numeric_cols)}")
        
        print(f"\n📋 Features para análise não supervisionada:")
        for i, col in enumerate(numeric_cols, 1):
            mean_val = self.data[col].mean()
            std_val = self.data[col].std()
            print(f"   {i:2d}. {col}: μ={mean_val:.2f}, σ={std_val:.2f}")
        
        return True
    
    def generate_results_table(self):
        """Override para incluir informações de demonstração"""
        print("\n📋 5. GERANDO TABELA DE RESULTADOS (DEMONSTRAÇÃO)")
        print("=" * 50)
        
        # Executar método pai
        results_table = super().generate_results_table()
        
        if results_table is not None:
            # Adicionar informações específicas da demo
            output_dir = r"c:\Users\Administrador\Faculdade-Impacta\Iniciação-cientifica\doc\code\cybersecurity-datasets"
            
            # Arquivo específico da demonstração
            demo_file = os.path.join(output_dir, "DEMO_resultados_nao_supervisionado.md")
            with open(demo_file, 'w', encoding='utf-8') as f:
                f.write("# 🎭 DEMONSTRAÇÃO - Aprendizado Não Supervisionado\n\n")
                f.write("**IMPORTANTE**: Este é um relatório de DEMONSTRAÇÃO usando dados simulados.\n\n")
                f.write("## Objetivo da Demonstração\n\n")
                f.write("Demonstrar a implementação das três técnicas principais:\n\n")
                f.write("1. **K-Means** (Agrupamento Particional)\n")
                f.write("2. **AGNES** (Agrupamento Hierárquico)\n")
                f.write("3. **PCA** (Redução de Dimensionalidade)\n\n")
                
                f.write("## Dataset Simulado\n\n")
                f.write(f"- **Amostras**: {self.results['dataset_info']['total_samples']:,}\n")
                f.write(f"- **Features**: {self.results['dataset_info']['numeric_features']}\n")
                f.write(f"- **Grupos simulados**: Normal (60%), Suspeito (30%), Malicioso (10%)\n\n")
                
                # Incluir resultados da tabela
                f.write("## Resultados da Análise\n\n")
                f.write(results_table.to_markdown(index=False))
                f.write("\n\n")
                
                # Incluir interpretação específica
                summary = self.results['comparative_results']['summary']
                f.write("## Interpretação dos Resultados\n\n")
                f.write("### ✅ Sucessos da Demonstração\n\n")
                f.write(f"- **K-Means**: Silhouette Score = {summary['kmeans']['silhouette_score']:.4f} ({summary['kmeans']['quality']} qualidade)\n")
                f.write(f"- **Hierárquico**: Coef. Cofenético = {summary['hierarchical']['cophenetic_correlation']:.4f} ({summary['hierarchical']['quality']} preservação)\n")
                f.write(f"- **PCA**: {summary['pca']['variance_explained']*100:.1f}% da variância preservada com {summary['pca']['components_used']} componentes\n\n")
                
                f.write("### 🎯 Para Dados Reais\n\n")
                f.write("Para executar com dados reais da Kaggle:\n")
                f.write("1. Configure as credenciais da Kaggle\n")
                f.write("2. Execute: `python unsupervised_cybersecurity_analysis.py`\n")
                f.write("3. Compare os resultados com esta demonstração\n")
            
            print(f"📋 Relatório de demonstração salvo: {demo_file}")
        
        return results_table

def main():
    """Executa demonstração completa de aprendizado não supervisionado"""
    print("🚀 DEMO: APRENDIZADO NÃO SUPERVISIONADO EM CYBERSECURITY")
    print("=" * 60)
    print("Esta demonstração implementa as 3 técnicas principais da pesquisa:")
    print("1. 🔵 K-Means (Agrupamento Particional)")  
    print("2. 🔴 AGNES (Agrupamento Hierárquico)")
    print("3. 🟡 PCA (Redução de Dimensionalidade)")
    print("=" * 60)
    
    try:
        # 1. Criar dados de exemplo
        demo_data, true_labels = create_cybersecurity_dataset()
        
        # 2. Criar analisador demo
        analyzer = DemoUnsupervisedAnalyzer(demo_data)
        
        # 3. Executar análise completa
        results = analyzer.run_complete_analysis()
        
        if results is not None:
            print("\n" + "🎯 DEMONSTRAÇÃO CONCLUÍDA COM SUCESSO!")
            print("=" * 40)
            print("📊 Tabela de resultados gerada")
            print("📋 Relatórios salvos em cybersecurity-datasets/")
            print("🔬 Resultados prontos para pesquisa acadêmica")
            print("\n💡 Para usar dados reais:")
            print("   1. Configure credenciais Kaggle")
            print("   2. Execute: python unsupervised_cybersecurity_analysis.py")
        
    except Exception as e:
        print(f"❌ Erro durante a demonstração: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
