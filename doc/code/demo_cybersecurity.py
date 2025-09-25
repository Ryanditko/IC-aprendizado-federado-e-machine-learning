"""
Demo: Análise Cybersecurity - Exemplo Básico
Demonstra como o script de análise funciona sem precisar configurar credenciais Kaggle
"""

import pandas as pd
import numpy as np
from cybersecurity_analysis import CybersecurityAnalyzer
import os
import warnings
warnings.filterwarnings('ignore')

def create_sample_cybersecurity_data():
    """Cria dados de exemplo para demonstração"""
    print("📊 Criando dataset de exemplo para demonstração...")
    
    # Simular dados de cybersecurity
    np.random.seed(42)
    n_samples = 1000
    
    # Features simuladas
    data = {
        'packet_size': np.random.normal(1500, 500, n_samples),
        'connection_duration': np.random.exponential(10, n_samples),
        'bytes_sent': np.random.lognormal(8, 2, n_samples),
        'bytes_received': np.random.lognormal(7, 1.5, n_samples),
        'num_connections': np.random.poisson(5, n_samples),
        'port_scan_attempts': np.random.poisson(2, n_samples),
        'failed_logins': np.random.poisson(1, n_samples),
        'protocol_violations': np.random.poisson(0.5, n_samples)
    }
    
    # Criar classes: 0 = Normal, 1 = Suspicious, 2 = Malicious
    threat_scores = (
        0.3 * (data['packet_size'] / 1500) +
        0.2 * (data['port_scan_attempts'] / 10) +
        0.3 * (data['failed_logins'] / 5) +
        0.2 * (data['protocol_violations'] / 3)
    )
    
    labels = []
    for score in threat_scores:
        if score < 0.3:
            labels.append(0)  # Normal
        elif score < 0.7:
            labels.append(1)  # Suspicious  
        else:
            labels.append(2)  # Malicious
    
    data['threat_type'] = labels
    
    # Criar DataFrame
    df = pd.DataFrame(data)
    
    print(f"✅ Dataset criado com {n_samples} amostras")
    print(f"   Normal: {sum(np.array(labels) == 0)}")
    print(f"   Suspicious: {sum(np.array(labels) == 1)}")
    print(f"   Malicious: {sum(np.array(labels) == 2)}")
    
    return df

class DemoAnalyzer(CybersecurityAnalyzer):
    """Versão demo do analisador que usa dados simulados"""
    
    def __init__(self, demo_data):
        super().__init__()
        self.data = demo_data
        self.dataset_path = "demo_data"
        
        # Informações básicas
        self.results['dataset_info'] = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024**2
        }
        
    def download_dataset(self):
        """Override para usar dados simulados"""
        print("📥 Usando dados simulados para demonstração...")
        return True
        
    def load_and_explore_data(self):
        """Override para usar dados já carregados"""
        print("\n🔍 Explorando dados simulados...")
        
        # Informações básicas
        print(f"📈 Informações do Dataset:")
        print(f"   Linhas: {self.data.shape[0]:,}")
        print(f"   Colunas: {self.data.shape[1]}")
        
        # Colunas disponíveis
        print(f"\n📋 Features disponíveis:")
        for i, col in enumerate(self.data.columns, 1):
            print(f"   {i:2d}. {col}")
        
        # Classes
        print(f"\n🎯 Distribuição das classes:")
        class_counts = self.data['threat_type'].value_counts().sort_index()
        class_names = {0: 'Normal', 1: 'Suspicious', 2: 'Malicious'}
        for class_id, count in class_counts.items():
            print(f"   {class_names[class_id]}: {count} ({count/len(self.data)*100:.1f}%)")
            
        return True
        
    def generate_report(self):
        """Gera relatório para dados demo"""
        print("\n📋 Gerando Relatório de Demonstração...")
        
        # Criar diretório de resultados
        output_dir = r"c:\Users\Administrador\Faculdade-Impacta\Iniciação-cientifica\doc\code\cybersecurity-datasets"
        os.makedirs(output_dir, exist_ok=True)
        
        # Relatório demo
        report_file = os.path.join(output_dir, "demo_cybersecurity_report.md")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Relatório Demo - Análise Cybersecurity\n\n")
            f.write("**ESTE É UM RELATÓRIO DE DEMONSTRAÇÃO COM DADOS SIMULADOS**\n\n")
            f.write(f"Data da análise: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Informações do dataset
            f.write("## 📊 Informações do Dataset (Simulado)\n\n")
            f.write(f"- **Dimensões**: {self.data.shape[0]:,} linhas × {self.data.shape[1]} colunas\n")
            f.write(f"- **Features**: {', '.join([col for col in self.data.columns if col != 'threat_type'])}\n")
            f.write(f"- **Classes**: Normal (0), Suspicious (1), Malicious (2)\n\n")
            
            # Análise supervisionada (se executada)
            if 'supervised_learning' in self.results and self.results['supervised_learning']:
                sup = self.results['supervised_learning']
                f.write("## 🎯 Análise Supervisionada\n\n")
                f.write(f"- **Modelo**: {sup['model_type']}\n")
                f.write(f"- **Acurácia**: {sup['accuracy']:.4f}\n\n")
                
                f.write("### Features Mais Importantes:\n")
                for i, feat in enumerate(sup['feature_importance'][:5], 1):
                    f.write(f"{i}. {feat['feature']}: {feat['importance']:.4f}\n")
                f.write("\n")
                
            # Análise não supervisionada (se executada)
            if 'unsupervised_learning' in self.results and self.results['unsupervised_learning']:
                unsup = self.results['unsupervised_learning']
                f.write("## 🔄 Análise Não Supervisionada\n\n")
                f.write(f"- **Melhor K (clusters)**: {unsup['best_k']}\n")
                f.write(f"- **Silhouette Score**: {unsup['best_silhouette_score']:.4f}\n\n")
                
            # Detecção de anomalias (se executada)
            if 'anomaly_detection' in self.results and self.results['anomaly_detection']:
                anom = self.results['anomaly_detection']
                f.write("## 🚨 Detecção de Anomalias\n\n")
                f.write(f"- **Isolation Forest**: {anom['isolation_forest_percentage']:.1f}% anomalias\n")
                f.write(f"- **LOF**: {anom['lof_percentage']:.1f}% anomalias\n\n")
                
            f.write("## 💡 Próximos Passos\n\n")
            f.write("1. Configure as credenciais Kaggle para usar dados reais\n")
            f.write("2. Execute `python cybersecurity_analysis.py` para análise completa\n")
            f.write("3. Compare os resultados com dados reais vs simulados\n")
            
        print(f"✅ Relatório demo salvo em: {report_file}")
        return True


def main():
    """Executa demonstração completa"""
    print("🚀 DEMO: ANÁLISE CYBERSECURITY")
    print("=" * 50)
    print("Esta demonstração usa dados simulados para mostrar como o script funciona")
    print("Para usar dados reais, configure as credenciais Kaggle e execute cybersecurity_analysis.py")
    print("=" * 50)
    
    try:
        # 1. Criar dados de exemplo
        demo_data = create_sample_cybersecurity_data()
        
        # 2. Criar analisador demo
        analyzer = DemoAnalyzer(demo_data)
        
        # 3. Executar análises
        print("\n" + "🔄 INICIANDO ANÁLISES...")
        
        analyzer.load_and_explore_data()
        analyzer.preprocess_data()
        analyzer.supervised_analysis()
        analyzer.unsupervised_analysis()
        analyzer.anomaly_detection()
        analyzer.generate_report()
        
        print("\n" + "=" * 50)
        print("🎉 DEMONSTRAÇÃO CONCLUÍDA!")
        print("📁 Confira o relatório em cybersecurity-datasets/")
        print("💡 Para análise com dados reais:")
        print("   1. Configure credenciais Kaggle")
        print("   2. Execute: python cybersecurity_analysis.py")
        
    except Exception as e:
        print(f"❌ Erro durante a demonstração: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
