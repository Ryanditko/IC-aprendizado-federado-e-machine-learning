"""
Módulo para avaliação de modelos de detecção de outliers.

Este módulo contém métricas e visualizações para avaliar a performance
dos diferentes métodos de detecção de outliers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
)
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Classe para avaliação de modelos de detecção de outliers."""
    
    def __init__(self, true_labels=None):
        """
        Inicializa o avaliador de modelos.
        
        Args:
            true_labels (np.array): Labels verdadeiros (se disponíveis)
        """
        self.true_labels = true_labels
        self.predictions = {}
        self.metrics = {}
        
    def add_predictions(self, method_name, predictions):
        """
        Adiciona predições de um método.
        
        Args:
            method_name (str): Nome do método
            predictions (np.array): Predições do método (booleano)
        """
        self.predictions[method_name] = predictions.astype(bool)
    
    def calculate_metrics(self):
        """
        Calcula métricas de avaliação para todos os métodos.
        
        Returns:
            pd.DataFrame: DataFrame com métricas por método
        """
        if self.true_labels is None:
            print("Aviso: Labels verdadeiros não disponíveis. Calculando apenas estatísticas descritivas.")
            return self._calculate_descriptive_metrics()
        
        metrics_list = []
        
        for method_name, predictions in self.predictions.items():
            metrics = {
                'Método': method_name,
                'Acurácia': accuracy_score(self.true_labels, predictions),
                'Precisão': precision_score(self.true_labels, predictions, zero_division=0),
                'Recall': recall_score(self.true_labels, predictions, zero_division=0),
                'F1-Score': f1_score(self.true_labels, predictions, zero_division=0),
                'N° Outliers Detectados': np.sum(predictions),
                'N° Outliers Reais': np.sum(self.true_labels),
                'Taxa Falsos Positivos': self._calculate_fpr(self.true_labels, predictions),
                'Taxa Falsos Negativos': self._calculate_fnr(self.true_labels, predictions)
            }
            metrics_list.append(metrics)
        
        self.metrics = pd.DataFrame(metrics_list)
        return self.metrics
    
    def _calculate_descriptive_metrics(self):
        """Calcula métricas descritivas quando não há labels verdadeiros."""
        metrics_list = []
        
        for method_name, predictions in self.predictions.items():
            total_points = len(predictions)
            outliers_detected = np.sum(predictions)
            
            metrics = {
                'Método': method_name,
                'Total de Pontos': total_points,
                'Outliers Detectados': outliers_detected,
                'Porcentagem de Outliers': f"{(outliers_detected / total_points) * 100:.2f}%",
                'Pontos Normais': total_points - outliers_detected
            }
            metrics_list.append(metrics)
        
        return pd.DataFrame(metrics_list)
    
    def _calculate_fpr(self, y_true, y_pred):
        """Calcula a Taxa de Falsos Positivos."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return fp / (fp + tn) if (fp + tn) > 0 else 0
    
    def _calculate_fnr(self, y_true, y_pred):
        """Calcula a Taxa de Falsos Negativos."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return fn / (fn + tp) if (fn + tp) > 0 else 0
    
    def plot_confusion_matrices(self, figsize=(15, 10)):
        """
        Plota matrizes de confusão para todos os métodos.
        
        Args:
            figsize (tuple): Tamanho da figura
        """
        if self.true_labels is None:
            print("Labels verdadeiros não disponíveis para matriz de confusão.")
            return
        
        n_methods = len(self.predictions)
        cols = min(3, n_methods)
        rows = (n_methods + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_methods == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (method_name, predictions) in enumerate(self.predictions.items()):
            row = idx // cols
            col = idx % cols
            
            if rows > 1:
                ax = axes[row, col]
            else:
                ax = axes[col]
            
            cm = confusion_matrix(self.true_labels, predictions)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'{method_name}')
            ax.set_xlabel('Predito')
            ax.set_ylabel('Real')
            ax.set_xticklabels(['Normal', 'Outlier'])
            ax.set_yticklabels(['Normal', 'Outlier'])
        
        # Ocultar subplots vazios
        for idx in range(n_methods, rows * cols):
            row = idx // cols
            col = idx % cols
            if rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.suptitle('Matrizes de Confusão por Método', fontsize=16)
        plt.tight_layout()
        return fig
    
    def plot_roc_curves(self, scores_dict=None, figsize=(10, 8)):
        """
        Plota curvas ROC para métodos que fornecem scores.
        
        Args:
            scores_dict (dict): Dicionário com scores de cada método
            figsize (tuple): Tamanho da figura
        """
        if self.true_labels is None:
            print("Labels verdadeiros não disponíveis para curvas ROC.")
            return
        
        if scores_dict is None:
            print("Scores não fornecidos. Usando predições binárias.")
            scores_dict = self.predictions
        
        plt.figure(figsize=figsize)
        
        for method_name, scores in scores_dict.items():
            try:
                # Se scores são binários, usar como está
                if len(np.unique(scores)) == 2:
                    fpr, tpr, _ = roc_curve(self.true_labels, scores)
                    auc = roc_auc_score(self.true_labels, scores)
                else:
                    # Se scores são contínuos, calcular ROC normalmente
                    fpr, tpr, _ = roc_curve(self.true_labels, scores)
                    auc = roc_auc_score(self.true_labels, scores)
                
                plt.plot(fpr, tpr, label=f'{method_name} (AUC = {auc:.3f})')
            except Exception as e:
                print(f"Erro ao calcular ROC para {method_name}: {e}")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('Taxa de Falsos Positivos')
        plt.ylabel('Taxa de Verdadeiros Positivos')
        plt.title('Curvas ROC')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_precision_recall_curves(self, scores_dict=None, figsize=(10, 8)):
        """
        Plota curvas Precision-Recall para métodos que fornecem scores.
        
        Args:
            scores_dict (dict): Dicionário com scores de cada método
            figsize (tuple): Tamanho da figura
        """
        if self.true_labels is None:
            print("Labels verdadeiros não disponíveis para curvas Precision-Recall.")
            return
        
        if scores_dict is None:
            scores_dict = self.predictions
        
        plt.figure(figsize=figsize)
        
        for method_name, scores in scores_dict.items():
            try:
                precision, recall, _ = precision_recall_curve(self.true_labels, scores)
                plt.plot(recall, precision, label=method_name)
            except Exception as e:
                print(f"Erro ao calcular Precision-Recall para {method_name}: {e}")
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Curvas Precision-Recall')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def analyze_method_agreement(self):
        """
        Analisa a concordância entre diferentes métodos.
        
        Returns:
            pd.DataFrame: Matriz de concordância entre métodos
        """
        methods = list(self.predictions.keys())
        n_methods = len(methods)
        
        # Matriz de concordância
        agreement_matrix = np.zeros((n_methods, n_methods))
        
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i == j:
                    agreement_matrix[i, j] = 1.0
                else:
                    # Calcula concordância (porcentagem de predições iguais)
                    pred1 = self.predictions[method1]
                    pred2 = self.predictions[method2]
                    agreement = np.mean(pred1 == pred2)
                    agreement_matrix[i, j] = agreement
        
        # Criando DataFrame
        agreement_df = pd.DataFrame(
            agreement_matrix, 
            index=methods, 
            columns=methods
        )
        
        return agreement_df
    
    def plot_method_agreement(self, figsize=(10, 8)):
        """
        Plota heatmap da concordância entre métodos.
        
        Args:
            figsize (tuple): Tamanho da figura
        """
        agreement_df = self.analyze_method_agreement()
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            agreement_df, 
            annot=True, 
            cmap='RdYlBu_r', 
            center=0.5,
            square=True,
            fmt='.3f'
        )
        plt.title('Concordância entre Métodos de Detecção de Outliers')
        plt.ylabel('Método 1')
        plt.xlabel('Método 2')
        plt.tight_layout()
        plt.show()
    
    def ensemble_prediction(self, threshold=0.5):
        """
        Cria uma predição ensemble baseada na votação dos métodos.
        
        Args:
            threshold (float): Threshold para considerar um ponto como outlier
            
        Returns:
            np.array: Predições ensemble
        """
        if not self.predictions:
            raise ValueError("Nenhuma predição foi adicionada ainda.")
        
        # Soma dos votos de cada método
        votes = np.zeros(len(list(self.predictions.values())[0]))
        
        for predictions in self.predictions.values():
            votes += predictions.astype(int)
        
        # Normaliza os votos
        vote_ratio = votes / len(self.predictions)
        
        # Aplica threshold
        ensemble_pred = vote_ratio >= threshold
        
        return ensemble_pred, vote_ratio
    
    def plot_ensemble_analysis(self, figsize=(15, 5)):
        """
        Plota análise do ensemble de métodos.
        
        Args:
            figsize (tuple): Tamanho da figura
        """
        ensemble_pred, vote_ratio = self.ensemble_prediction()
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot 1: Distribuição dos votos
        axes[0].hist(vote_ratio, bins=20, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Proporção de Votos')
        axes[0].set_ylabel('Frequência')
        axes[0].set_title('Distribuição de Votos do Ensemble')
        axes[0].axvline(x=0.5, color='red', linestyle='--', label='Threshold=0.5')
        axes[0].legend()
        
        # Plot 2: Comparação de outliers por método
        method_counts = [np.sum(pred) for pred in self.predictions.values()]
        method_names = list(self.predictions.keys())
        
        axes[1].bar(range(len(method_names)), method_counts, alpha=0.7)
        axes[1].set_xlabel('Métodos')
        axes[1].set_ylabel('N° de Outliers')
        axes[1].set_title('Outliers Detectados por Método')
        axes[1].set_xticks(range(len(method_names)))
        axes[1].set_xticklabels(method_names, rotation=45, ha='right')
        
        # Adicionar linha do ensemble
        ensemble_count = np.sum(ensemble_pred)
        axes[1].axhline(y=ensemble_count, color='red', linestyle='--', 
                       label=f'Ensemble: {ensemble_count}')
        axes[1].legend()
        
        # Plot 3: Votos vs índices dos pontos
        axes[2].scatter(range(len(vote_ratio)), vote_ratio, alpha=0.6, s=20)
        axes[2].set_xlabel('Índice do Ponto')
        axes[2].set_ylabel('Proporção de Votos')
        axes[2].set_title('Votos por Ponto de Dados')
        axes[2].axhline(y=0.5, color='red', linestyle='--', label='Threshold=0.5')
        axes[2].legend()
        
        plt.tight_layout()
        plt.show()
        
        return ensemble_pred, vote_ratio
