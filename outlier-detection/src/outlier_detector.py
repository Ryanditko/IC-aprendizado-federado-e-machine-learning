"""
Módulo com implementações de técnicas de detecção de outliers.

Este módulo contém implementações de várias técnicas de detecção de outliers
para aprendizado não supervisionado:

1. Z-Score
2. IQR (Interquartile Range)
3. Isolation Forest
4. Local Outlier Factor (LOF)
5. DBSCAN
6. One-Class SVM
7. Elliptic Envelope
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class OutlierDetector:
    """Classe principal para detecção de outliers com múltiplas técnicas."""
    
    def __init__(self, data):
        """
        Inicializa o detector de outliers.
        
        Args:
            data (pd.DataFrame): DataFrame com os dados para análise
        """
        self.data = data.copy()
        self.numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        self.scaler = StandardScaler()
        self.results = {}
        
    def z_score_detection(self, threshold=3):
        """
        Detecção de outliers usando Z-Score.
        
        Args:
            threshold (float): Limite para considerar um ponto como outlier
            
        Returns:
            np.array: Array booleano indicando outliers
        """
        outliers = np.zeros(len(self.data), dtype=bool)
        
        for column in self.numeric_columns:
            z_scores = np.abs(stats.zscore(self.data[column]))
            outliers |= (z_scores > threshold)
        
        self.results['z_score'] = {
            'outliers': outliers,
            'method': 'Z-Score',
            'params': {'threshold': threshold},
            'n_outliers': np.sum(outliers)
        }
        
        return outliers
    
    def iqr_detection(self, multiplier=1.5):
        """
        Detecção de outliers usando IQR (Interquartile Range).
        
        Args:
            multiplier (float): Multiplicador para o IQR
            
        Returns:
            np.array: Array booleano indicando outliers
        """
        outliers = np.zeros(len(self.data), dtype=bool)
        
        for column in self.numeric_columns:
            Q1 = self.data[column].quantile(0.25)
            Q3 = self.data[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            column_outliers = (self.data[column] < lower_bound) | (self.data[column] > upper_bound)
            outliers |= column_outliers
        
        self.results['iqr'] = {
            'outliers': outliers,
            'method': 'IQR',
            'params': {'multiplier': multiplier},
            'n_outliers': np.sum(outliers)
        }
        
        return outliers
    
    def isolation_forest_detection(self, contamination=0.1, random_state=42):
        """
        Detecção de outliers usando Isolation Forest.
        
        Args:
            contamination (float): Proporção esperada de outliers
            random_state (int): Seed para reprodutibilidade
            
        Returns:
            np.array: Array booleano indicando outliers
        """
        # Normalizando os dados
        data_scaled = self.scaler.fit_transform(self.data[self.numeric_columns])
        
        # Aplicando Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        
        predictions = iso_forest.fit_predict(data_scaled)
        outliers = (predictions == -1)
        
        self.results['isolation_forest'] = {
            'outliers': outliers,
            'method': 'Isolation Forest',
            'params': {'contamination': contamination},
            'n_outliers': np.sum(outliers),
            'scores': iso_forest.score_samples(data_scaled)
        }
        
        return outliers
    
    def lof_detection(self, n_neighbors=20, contamination=0.1):
        """
        Detecção de outliers usando Local Outlier Factor (LOF).
        
        Args:
            n_neighbors (int): Número de vizinhos para considerar
            contamination (float): Proporção esperada de outliers
            
        Returns:
            np.array: Array booleano indicando outliers
        """
        # Normalizando os dados
        data_scaled = self.scaler.fit_transform(self.data[self.numeric_columns])
        
        # Aplicando LOF
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination
        )
        
        predictions = lof.fit_predict(data_scaled)
        outliers = (predictions == -1)
        
        self.results['lof'] = {
            'outliers': outliers,
            'method': 'Local Outlier Factor',
            'params': {'n_neighbors': n_neighbors, 'contamination': contamination},
            'n_outliers': np.sum(outliers),
            'scores': lof.negative_outlier_factor_
        }
        
        return outliers
    
    def dbscan_detection(self, eps=0.5, min_samples=5):
        """
        Detecção de outliers usando DBSCAN.
        
        Args:
            eps (float): Distância máxima entre pontos do mesmo cluster
            min_samples (int): Número mínimo de pontos para formar um cluster
            
        Returns:
            np.array: Array booleano indicando outliers
        """
        # Normalizando os dados
        data_scaled = self.scaler.fit_transform(self.data[self.numeric_columns])
        
        # Aplicando DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(data_scaled)
        
        # Pontos com label -1 são considerados outliers
        outliers = (labels == -1)
        
        self.results['dbscan'] = {
            'outliers': outliers,
            'method': 'DBSCAN',
            'params': {'eps': eps, 'min_samples': min_samples},
            'n_outliers': np.sum(outliers),
            'labels': labels,
            'n_clusters': len(set(labels)) - (1 if -1 in labels else 0)
        }
        
        return outliers
    
    def one_class_svm_detection(self, nu=0.1, gamma='scale'):
        """
        Detecção de outliers usando One-Class SVM.
        
        Args:
            nu (float): Limite superior na fração de erros de treinamento
            gamma (str): Coeficiente do kernel
            
        Returns:
            np.array: Array booleano indicando outliers
        """
        # Normalizando os dados
        data_scaled = self.scaler.fit_transform(self.data[self.numeric_columns])
        
        # Aplicando One-Class SVM
        oc_svm = OneClassSVM(nu=nu, gamma=gamma)
        predictions = oc_svm.fit_predict(data_scaled)
        outliers = (predictions == -1)
        
        self.results['one_class_svm'] = {
            'outliers': outliers,
            'method': 'One-Class SVM',
            'params': {'nu': nu, 'gamma': gamma},
            'n_outliers': np.sum(outliers)
        }
        
        return outliers
    
    def elliptic_envelope_detection(self, contamination=0.1):
        """
        Detecção de outliers usando Elliptic Envelope.
        
        Args:
            contamination (float): Proporção esperada de outliers
            
        Returns:
            np.array: Array booleano indicando outliers
        """
        # Normalizando os dados
        data_scaled = self.scaler.fit_transform(self.data[self.numeric_columns])
        
        # Aplicando Elliptic Envelope
        ee = EllipticEnvelope(contamination=contamination, random_state=42)
        predictions = ee.fit_predict(data_scaled)
        outliers = (predictions == -1)
        
        self.results['elliptic_envelope'] = {
            'outliers': outliers,
            'method': 'Elliptic Envelope',
            'params': {'contamination': contamination},
            'n_outliers': np.sum(outliers)
        }
        
        return outliers
    
    def run_all_methods(self):
        """
        Executa todas as técnicas de detecção de outliers.
        
        Returns:
            dict: Dicionário com resultados de todos os métodos
        """
        print("Executando detecção de outliers com múltiplas técnicas...")
        
        methods = [
            ('Z-Score', self.z_score_detection),
            ('IQR', self.iqr_detection),
            ('Isolation Forest', self.isolation_forest_detection),
            ('LOF', self.lof_detection),
            ('DBSCAN', self.dbscan_detection),
            ('One-Class SVM', self.one_class_svm_detection),
            ('Elliptic Envelope', self.elliptic_envelope_detection)
        ]
        
        for name, method in methods:
            print(f"Executando {name}...")
            try:
                method()
                print(f"✓ {name} concluído")
            except Exception as e:
                print(f"✗ Erro em {name}: {e}")
        
        return self.results
    
    def compare_methods(self):
        """
        Compara os resultados de todos os métodos.
        
        Returns:
            pd.DataFrame: DataFrame com comparação dos métodos
        """
        if not self.results:
            self.run_all_methods()
        
        comparison = []
        for method_name, result in self.results.items():
            comparison.append({
                'Método': result['method'],
                'N° Outliers': result['n_outliers'],
                'Porcentagem': f"{(result['n_outliers'] / len(self.data)) * 100:.2f}%",
                'Parâmetros': str(result['params'])
            })
        
        return pd.DataFrame(comparison)
    
    def plot_outliers(self, method_name, figsize=(15, 10)):
        """
        Plota os outliers detectados por um método específico.
        
        Args:
            method_name (str): Nome do método para plotar
            figsize (tuple): Tamanho da figura
        """
        if method_name not in self.results:
            print(f"Método {method_name} não foi executado ainda.")
            return
        
        outliers = self.results[method_name]['outliers']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Detecção de Outliers - {self.results[method_name]["method"]}', fontsize=16)
        
        # Plot 1: Scatter plot dos dados
        if len(self.numeric_columns) >= 2:
            col1, col2 = self.numeric_columns[0], self.numeric_columns[1]
            
            axes[0, 0].scatter(
                self.data[col1][~outliers], 
                self.data[col2][~outliers], 
                c='blue', alpha=0.6, label='Normal'
            )
            axes[0, 0].scatter(
                self.data[col1][outliers], 
                self.data[col2][outliers], 
                c='red', alpha=0.8, label='Outlier'
            )
            axes[0, 0].set_xlabel(col1)
            axes[0, 0].set_ylabel(col2)
            axes[0, 0].set_title('Scatter Plot')
            axes[0, 0].legend()
        
        # Plot 2: Boxplot da primeira coluna
        axes[0, 1].boxplot([
            self.data[self.numeric_columns[0]][~outliers],
            self.data[self.numeric_columns[0]][outliers]
        ], labels=['Normal', 'Outlier'])
        axes[0, 1].set_title(f'Boxplot - {self.numeric_columns[0]}')
        
        # Plot 3: Boxplot da segunda coluna (se existir)
        if len(self.numeric_columns) >= 2:
            axes[1, 0].boxplot([
                self.data[self.numeric_columns[1]][~outliers],
                self.data[self.numeric_columns[1]][outliers]
            ], labels=['Normal', 'Outlier'])
            axes[1, 0].set_title(f'Boxplot - {self.numeric_columns[1]}')
        
        # Plot 4: Histograma de outliers por método
        method_counts = [
            len(self.data) - np.sum(outliers),  # Normal
            np.sum(outliers)  # Outliers
        ]
        axes[1, 1].bar(['Normal', 'Outliers'], method_counts, color=['blue', 'red'])
        axes[1, 1].set_title('Contagem de Pontos')
        axes[1, 1].set_ylabel('Quantidade')
        
        plt.tight_layout()
        return fig
