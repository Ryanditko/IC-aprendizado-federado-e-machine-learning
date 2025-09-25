"""
Script Completo para Análise do Dataset de Cybersecurity da Kaggle
Realiza download, exploração e análise completa dos dados de ameaças cibernéticas.
"""

import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import LocalOutlierFactor
from avaliador_nao_supervisionado import AvaliadorNaoSupervisionado
import os
import warnings
warnings.filterwarnings('ignore')

class CybersecurityAnalyzer:
    """
    Classe para análise completa do dataset de cybersecurity
    """
    
    def __init__(self):
        self.dataset_path = None
        self.data = None
        self.processed_data = None
        self.results = {
            'dataset_info': {},
            'exploratory_analysis': {},
            'supervised_learning': {},
            'unsupervised_learning': {},
            'anomaly_detection': {}
        }
    
    def download_dataset(self):
        """Baixa o dataset da Kaggle"""
        print("📥 Baixando dataset de cybersecurity da Kaggle...")
        
        try:
            # Download do dataset
            path = kagglehub.dataset_download("ramoliyafenil/text-based-cyber-threat-detection")
            self.dataset_path = path
            print(f"✅ Dataset baixado com sucesso!")
            print(f"📁 Localização: {path}")
            
            # Listar arquivos baixados
            files = os.listdir(path)
            print(f"📋 Arquivos encontrados: {files}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao baixar dataset: {str(e)}")
            print("💡 Verifique se você tem as credenciais da Kaggle configuradas")
            return False
    
    def load_and_explore_data(self):
        """Carrega e explora os dados"""
        print("\n🔍 Explorando estrutura dos dados...")
        
        if not self.dataset_path:
            print("❌ Dataset não foi baixado ainda. Execute download_dataset() primeiro.")
            return False
        
        try:
            # Encontrar e carregar o arquivo principal
            files = os.listdir(self.dataset_path)
            csv_files = [f for f in files if f.endswith('.csv')]
            
            if not csv_files:
                print("❌ Nenhum arquivo CSV encontrado no dataset")
                return False
            
            # Carregar o primeiro arquivo CSV encontrado
            main_file = csv_files[0]
            file_path = os.path.join(self.dataset_path, main_file)
            print(f"📊 Carregando arquivo: {main_file}")
            
            self.data = pd.read_csv(file_path)
            
            # Informações básicas do dataset
            print(f"\n📈 Informações do Dataset:")
            print(f"   Linhas: {self.data.shape[0]:,}")
            print(f"   Colunas: {self.data.shape[1]}")
            print(f"   Tamanho em memória: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Colunas disponíveis
            print(f"\n📋 Colunas do dataset:")
            for i, col in enumerate(self.data.columns, 1):
                print(f"   {i:2d}. {col}")
            
            # Tipos de dados
            print(f"\n🔢 Tipos de dados:")
            print(self.data.dtypes.value_counts())
            
            # Valores faltantes
            missing_values = self.data.isnull().sum()
            if missing_values.sum() > 0:
                print(f"\n⚠️ Valores faltantes:")
                missing_info = missing_values[missing_values > 0]
                for col, count in missing_info.items():
                    pct = (count / len(self.data)) * 100
                    print(f"   {col}: {count:,} ({pct:.1f}%)")
            else:
                print(f"\n✅ Nenhum valor faltante encontrado")
            
            # Primeiras linhas
            print(f"\n👀 Primeiras 5 linhas:")
            print(self.data.head())
            
            # Salvar informações básicas
            self.results['dataset_info'] = {
                'shape': self.data.shape,
                'columns': list(self.data.columns),
                'dtypes': self.data.dtypes.to_dict(),
                'missing_values': missing_values.to_dict(),
                'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024**2
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {str(e)}")
            return False
    
    def preprocess_data(self):
        """Pré-processamento dos dados"""
        print("\n🔧 Realizando pré-processamento dos dados...")
        
        if self.data is None:
            print("❌ Dados não carregados. Execute load_and_explore_data() primeiro.")
            return False
        
        try:
            # Criar cópia dos dados para processamento
            processed = self.data.copy()
            
            # Identificar coluna alvo (assumindo que existe uma coluna de classe/label)
            target_candidates = ['label', 'target', 'class', 'threat_type', 'attack_type', 'malicious']
            target_col = None
            
            for col in target_candidates:
                if col in processed.columns:
                    target_col = col
                    break
            
            # Se não encontrou coluna alvo óbvia, usar a última coluna
            if target_col is None:
                target_col = processed.columns[-1]
                print(f"⚠️ Coluna alvo não identificada automaticamente. Usando: {target_col}")
            else:
                print(f"🎯 Coluna alvo identificada: {target_col}")
            
            # Separar features e target
            if target_col in processed.columns:
                X = processed.drop(columns=[target_col])
                y = processed[target_col]
                
                print(f"📊 Classes na coluna alvo:")
                class_counts = y.value_counts()
                print(class_counts)
                
                # Codificar variável alvo se necessário
                if y.dtype == 'object':
                    le_target = LabelEncoder()
                    y_encoded = le_target.fit_transform(y)
                    print(f"🔄 Variável alvo codificada: {dict(enumerate(le_target.classes_))}")
                else:
                    y_encoded = y.values
                    le_target = None
            else:
                X = processed
                y = None
                y_encoded = None
                le_target = None
                print("⚠️ Análise apenas não supervisionada (sem coluna alvo)")
            
            # Processar features categóricas
            categorical_cols = X.select_dtypes(include=['object']).columns
            numerical_cols = X.select_dtypes(include=[np.number]).columns
            
            print(f"📝 Colunas categóricas: {len(categorical_cols)}")
            print(f"🔢 Colunas numéricas: {len(numerical_cols)}")
            
            # Para colunas de texto (possível análise de texto)
            text_cols = []
            for col in categorical_cols:
                if X[col].str.len().mean() > 20:  # Assumindo que textos longos são para análise textual
                    text_cols.append(col)
            
            if text_cols:
                print(f"📄 Possíveis colunas de texto para análise: {text_cols}")
            
            # Codificar variáveis categóricas (exceto colunas de texto)
            X_processed = X.copy()
            label_encoders = {}
            
            for col in categorical_cols:
                if col not in text_cols:  # Não codificar colunas de texto por enquanto
                    le = LabelEncoder()
                    X_processed[col] = le.fit_transform(X[col].astype(str))
                    label_encoders[col] = le
            
            # Remover colunas de texto por enquanto (podem ser processadas separadamente)
            for col in text_cols:
                X_processed = X_processed.drop(columns=[col])
            
            # Normalizar dados numéricos
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_processed)
            
            # Salvar dados processados
            self.processed_data = {
                'X_original': X,
                'X_processed': X_processed,
                'X_scaled': X_scaled,
                'y': y,
                'y_encoded': y_encoded,
                'target_col': target_col,
                'categorical_cols': list(categorical_cols),
                'numerical_cols': list(numerical_cols),
                'text_cols': text_cols,
                'label_encoders': label_encoders,
                'target_encoder': le_target,
                'scaler': scaler,
                'feature_names': list(X_processed.columns)
            }
            
            print(f"✅ Pré-processamento concluído!")
            print(f"   Features processadas: {X_processed.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro no pré-processamento: {str(e)}")
            return False
    
    def supervised_analysis(self):
        """Análise supervisionada (classificação)"""
        if not self.processed_data or self.processed_data['y_encoded'] is None:
            print("⚠️ Análise supervisionada não disponível (sem variável alvo)")
            return False
        
        print("\n🎯 Iniciando Análise Supervisionada...")
        
        try:
            X = self.processed_data['X_scaled']
            y = self.processed_data['y_encoded']
            
            # Dividir dados
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"📊 Dados divididos - Treino: {X_train.shape[0]}, Teste: {X_test.shape[0]}")
            
            # Treinar modelo Random Forest
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf_model.fit(X_train, y_train)
            
            # Predições
            y_pred = rf_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"🏆 Acurácia do Random Forest: {accuracy:.4f}")
            
            # Relatório de classificação
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Matriz de confusão
            cm = confusion_matrix(y_test, y_pred)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': self.processed_data['feature_names'],
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔝 Top 10 Features Mais Importantes:")
            print(feature_importance.head(10))
            
            # Salvar resultados
            self.results['supervised_learning'] = {
                'accuracy': accuracy,
                'classification_report': class_report,
                'confusion_matrix': cm.tolist(),
                'feature_importance': feature_importance.to_dict('records'),
                'model_type': 'RandomForestClassifier'
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Erro na análise supervisionada: {str(e)}")
            return False
    
    def unsupervised_analysis(self):
        """Análise não supervisionada (clustering)"""
        print("\n🔄 Iniciando Análise Não Supervisionada...")
        
        if not self.processed_data:
            print("❌ Dados não processados. Execute preprocess_data() primeiro.")
            return False
        
        try:
            X_scaled = self.processed_data['X_scaled']
            
            # Usar o avaliador não supervisionado existente
            avaliador = AvaliadorNaoSupervisionado(X_scaled)
            
            # Avaliação de clustering
            print("🔍 Avaliando técnicas de clustering...")
            
            # Testar diferentes números de clusters
            silhouette_scores = []
            k_range = range(2, min(11, len(X_scaled)//10))  # Até 10 clusters ou dados/10
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                sil_score = silhouette_score(X_scaled, labels)
                silhouette_scores.append(sil_score)
                print(f"   K={k}: Silhouette Score = {sil_score:.4f}")
            
            # Melhor número de clusters
            best_k = k_range[np.argmax(silhouette_scores)]
            best_silhouette = max(silhouette_scores)
            
            print(f"🏆 Melhor K: {best_k} (Silhouette: {best_silhouette:.4f})")
            
            # Clustering final com melhor K
            final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            cluster_labels = final_kmeans.fit_predict(X_scaled)
            
            # Análise dos clusters
            cluster_analysis = pd.DataFrame(X_scaled)
            cluster_analysis['cluster'] = cluster_labels
            
            cluster_summary = cluster_analysis.groupby('cluster').agg(['mean', 'std', 'count'])
            
            print(f"\n📊 Resumo dos Clusters:")
            print(f"   Cluster 0: {sum(cluster_labels == 0)} pontos")
            for i in range(1, best_k):
                print(f"   Cluster {i}: {sum(cluster_labels == i)} pontos")
            
            # PCA para visualização
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            explained_variance = pca.explained_variance_ratio_
            
            print(f"📈 PCA - Variância explicada: {explained_variance[0]:.3f}, {explained_variance[1]:.3f}")
            print(f"   Total: {sum(explained_variance):.3f}")
            
            # Salvar resultados
            self.results['unsupervised_learning'] = {
                'best_k': best_k,
                'best_silhouette_score': best_silhouette,
                'silhouette_scores': dict(zip(k_range, silhouette_scores)),
                'cluster_counts': {i: int(sum(cluster_labels == i)) for i in range(best_k)},
                'pca_explained_variance': explained_variance.tolist(),
                'total_explained_variance': float(sum(explained_variance))
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Erro na análise não supervisionada: {str(e)}")
            return False
    
    def anomaly_detection(self):
        """Detecção de anomalias"""
        print("\n🚨 Iniciando Detecção de Anomalias...")
        
        if not self.processed_data:
            print("❌ Dados não processados. Execute preprocess_data() primeiro.")
            return False
        
        try:
            X_scaled = self.processed_data['X_scaled']
            
            # Isolation Forest
            print("🌲 Aplicando Isolation Forest...")
            iso_forest = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
            anomaly_labels_iso = iso_forest.fit_predict(X_scaled)
            
            # Local Outlier Factor
            print("📍 Aplicando Local Outlier Factor...")
            lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, n_jobs=-1)
            anomaly_labels_lof = lof.fit_predict(X_scaled)
            
            # Contar anomalias
            iso_anomalies = sum(anomaly_labels_iso == -1)
            lof_anomalies = sum(anomaly_labels_lof == -1)
            
            print(f"🔍 Anomalias detectadas:")
            print(f"   Isolation Forest: {iso_anomalies} ({iso_anomalies/len(X_scaled)*100:.1f}%)")
            print(f"   Local Outlier Factor: {lof_anomalies} ({lof_anomalies/len(X_scaled)*100:.1f}%)")
            
            # Consenso entre métodos
            consensus_anomalies = np.where((anomaly_labels_iso == -1) & (anomaly_labels_lof == -1))[0]
            print(f"   Consenso (ambos métodos): {len(consensus_anomalies)} ({len(consensus_anomalies)/len(X_scaled)*100:.1f}%)")
            
            # Salvar resultados
            self.results['anomaly_detection'] = {
                'isolation_forest_anomalies': int(iso_anomalies),
                'lof_anomalies': int(lof_anomalies),
                'consensus_anomalies': int(len(consensus_anomalies)),
                'isolation_forest_percentage': float(iso_anomalies/len(X_scaled)*100),
                'lof_percentage': float(lof_anomalies/len(X_scaled)*100),
                'consensus_percentage': float(len(consensus_anomalies)/len(X_scaled)*100)
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Erro na detecção de anomalias: {str(e)}")
            return False
    
    def generate_report(self):
        """Gera relatório completo da análise"""
        print("\n📋 Gerando Relatório Completo...")
        
        # Salvar relatório em arquivo
        output_dir = r"c:\Users\Administrador\Faculdade-Impacta\Iniciação-cientifica\doc\code\cybersecurity-datasets"
        os.makedirs(output_dir, exist_ok=True)
        
        report_file = os.path.join(output_dir, "cybersecurity_analysis_report.md")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Relatório de Análise - Dataset Cybersecurity\n\n")
            f.write(f"Data da análise: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Informações do dataset
            if 'dataset_info' in self.results:
                info = self.results['dataset_info']
                f.write("## 📊 Informações do Dataset\n\n")
                f.write(f"- **Dimensões**: {info['shape'][0]:,} linhas × {info['shape'][1]} colunas\n")
                f.write(f"- **Uso de memória**: {info['memory_usage_mb']:.2f} MB\n")
                f.write(f"- **Colunas**: {', '.join(info['columns'])}\n\n")
            
            # Análise supervisionada
            if 'supervised_learning' in self.results:
                sup = self.results['supervised_learning']
                f.write("## 🎯 Análise Supervisionada\n\n")
                f.write(f"- **Modelo**: {sup['model_type']}\n")
                f.write(f"- **Acurácia**: {sup['accuracy']:.4f}\n\n")
                
                f.write("### Top 5 Features Mais Importantes:\n")
                for i, feat in enumerate(sup['feature_importance'][:5], 1):
                    f.write(f"{i}. {feat['feature']}: {feat['importance']:.4f}\n")
                f.write("\n")
            
            # Análise não supervisionada
            if 'unsupervised_learning' in self.results:
                unsup = self.results['unsupervised_learning']
                f.write("## 🔄 Análise Não Supervisionada\n\n")
                f.write(f"- **Melhor K (clusters)**: {unsup['best_k']}\n")
                f.write(f"- **Silhouette Score**: {unsup['best_silhouette_score']:.4f}\n")
                f.write(f"- **Variância explicada (PCA)**: {unsup['total_explained_variance']:.3f}\n\n")
                
                f.write("### Distribuição dos Clusters:\n")
                for cluster, count in unsup['cluster_counts'].items():
                    f.write(f"- Cluster {cluster}: {count} pontos\n")
                f.write("\n")
            
            # Detecção de anomalias
            if 'anomaly_detection' in self.results:
                anom = self.results['anomaly_detection']
                f.write("## 🚨 Detecção de Anomalias\n\n")
                f.write(f"- **Isolation Forest**: {anom['isolation_forest_anomalies']} anomalias ({anom['isolation_forest_percentage']:.1f}%)\n")
                f.write(f"- **Local Outlier Factor**: {anom['lof_anomalies']} anomalias ({anom['lof_percentage']:.1f}%)\n")
                f.write(f"- **Consenso**: {anom['consensus_anomalies']} anomalias ({anom['consensus_percentage']:.1f}%)\n\n")
        
        print(f"✅ Relatório salvo em: {report_file}")
        
        # Salvar resultados em CSV para planilha
        results_file = os.path.join(output_dir, "cybersecurity_results.csv")
        
        # Preparar dados para CSV
        csv_data = []
        
        if 'supervised_learning' in self.results:
            sup = self.results['supervised_learning']
            csv_data.append({
                'Técnica': 'Classificação Supervisionada',
                'Algoritmo': sup['model_type'],
                'Métrica': 'Acurácia',
                'Valor': sup['accuracy'],
                'Observações': f"Features importantes: {', '.join([f['feature'] for f in sup['feature_importance'][:3]])}"
            })
        
        if 'unsupervised_learning' in self.results:
            unsup = self.results['unsupervised_learning']
            csv_data.append({
                'Técnica': 'Clustering K-Means',
                'Algoritmo': 'K-Means',
                'Métrica': 'Silhouette Score',
                'Valor': unsup['best_silhouette_score'],
                'Observações': f"K ótimo: {unsup['best_k']}, Variância PCA: {unsup['total_explained_variance']:.3f}"
            })
        
        if 'anomaly_detection' in self.results:
            anom = self.results['anomaly_detection']
            csv_data.append({
                'Técnica': 'Detecção de Anomalias',
                'Algoritmo': 'Isolation Forest',
                'Métrica': 'Percentual de Anomalias',
                'Valor': anom['isolation_forest_percentage'],
                'Observações': f"Consenso com LOF: {anom['consensus_percentage']:.1f}%"
            })
        
        pd.DataFrame(csv_data).to_csv(results_file, index=False, encoding='utf-8')
        print(f"📊 Resultados CSV salvos em: {results_file}")
        
        return True
    
    def run_complete_analysis(self):
        """Executa análise completa do dataset"""
        print("🚀 INICIANDO ANÁLISE COMPLETA DO DATASET CYBERSECURITY")
        print("=" * 60)
        
        # 1. Download
        if not self.download_dataset():
            return False
        
        # 2. Carregamento e exploração
        if not self.load_and_explore_data():
            return False
        
        # 3. Pré-processamento
        if not self.preprocess_data():
            return False
        
        # 4. Análise supervisionada
        self.supervised_analysis()
        
        # 5. Análise não supervisionada
        self.unsupervised_analysis()
        
        # 6. Detecção de anomalias
        self.anomaly_detection()
        
        # 7. Gerar relatório
        self.generate_report()
        
        print("\n" + "=" * 60)
        print("🎉 ANÁLISE COMPLETA FINALIZADA COM SUCESSO!")
        print("📋 Confira os arquivos gerados na pasta cybersecurity-datasets/")
        
        return True


def main():
    """Função principal para executar a análise"""
    try:
        # Configurar matplotlib para não mostrar gráficos
        plt.ioff()
        
        # Criar e executar analisador
        analyzer = CybersecurityAnalyzer()
        analyzer.run_complete_analysis()
        
    except Exception as e:
        print(f"❌ Erro durante a execução: {str(e)}")
        print("💡 Verifique se:")
        print("   1. As dependências estão instaladas (pip install -r requirements.txt)")
        print("   2. As credenciais da Kaggle estão configuradas")
        print("   3. Há conexão com a internet")


if __name__ == "__main__":
    main()
