"""
Gera um PDF de apresentação com os gráficos e explicações.
Salva em: output-images2/presentation_nsl_kdd.pdf
"""
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, 'output-images2')
OUTPUT_PDF = os.path.join(IMAGES_DIR, 'presentation_nsl_kdd.pdf')

slides = [
    {
        'file': 'correlation_matrix.png',
        'title': 'Matriz de Correlação - Seleção de Features',
        'text': (
            'O que mostra: Heatmap das correlações entre features do NSL-KDD.\n'
            'Por que é importante: identifica features altamente correlacionadas (>0.95) para remoção.\n'
            'Requisito atendido: "Antes de usar todas as features, faça uma análise de correlação e descarte as irrelevantes.'
        )
    },
    {
        'file': 'metricas_acuracia_precisao_recall.png',
        'title': 'Acurácia, Precisão e Recall por Método',
        'text': (
            'O que mostra: comparação de Acurácia, Precisão e Recall para cada método.\n'
            'Por que é importante: Recall é a métrica principal (sensibilidade) — destaque no gráfico.\n'
            'Requisito atendido: "trazer gráfico: acurácia, precisão e recall".'
        )
    },
    {
        'file': 'matriz_confusao_detalhada.png',
        'title': 'Matriz de Confusão Detalhada (One-Class SVM)',
        'text': (
            'O que mostra: TP / FP / TN / FN para o melhor método (One-Class SVM).\n'
            'Por que é importante: demonstra trade-off entre detectar ataques (recall) e falsos positivos (precision).\n'
            'Requisito atendido: "trazer gráfico: matriz de confusão".'
        )
    },
    {
        'file': 'recall_comparativo.png',
        'title': 'Recall Comparativo e Trade-off com F1-Score',
        'text': (
            'O que mostra: Recall isolado e comparação Recall vs F1-Score.\n'
            'Por que é importante: evidencia a priorização da sensibilidade e o impacto no F1 e precision.\n'
            'Requisito atendido: Priorizar recall como métrica principal.'
        )
    },
    {
        'file': 'nsl_kdd_comparison.png',
        'title': 'Comparação Completa de Métodos',
        'text': (
            'O que mostra: painéis com Recall, F1-Score, Precision e tempo de execução.\n'
            'Por que é importante: visão consolidada da performance dos métodos.\n'
            'Relevância para a apresentação: resumo visual antes de discutir próximos passos.'
        )
    }
]

os.makedirs(IMAGES_DIR, exist_ok=True)

with PdfPages(OUTPUT_PDF) as pdf:
    # Página de título
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    fig.suptitle('NSL-KDD - Resultados (Normal vs U2R)\nResumo para apresentação', fontsize=20, fontweight='bold')
    plt.axis('off')
    plt.text(0.5, 0.45, 'Implementações: seleção de features por correlação, normalização condicional,\n'
             'detecção de outliers (IsolationForest, LOF, One-Class SVM, EllipticEnvelope).',
             ha='center', va='center', fontsize=12)
    plt.text(0.5, 0.25, 'Métrica principal: Recall (sensibilidade).\nGerado automaticamente.', ha='center', va='center', fontsize=10)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    for slide in slides:
        img_path = os.path.join(IMAGES_DIR, slide['file'])
        if not os.path.exists(img_path):
            # criar slide de erro
            fig = plt.figure(figsize=(11.69, 8.27))
            plt.text(0.5, 0.5, f"Arquivo não encontrado: {slide['file']}", ha='center', va='center', fontsize=16, color='red')
            plt.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            continue

        img = mpimg.imread(img_path)
        fig = plt.figure(figsize=(11.69, 8.27))
        ax = fig.add_axes([0.05, 0.18, 0.9, 0.75])
        ax.imshow(img)
        ax.axis('off')
        # Texto explicativo abaixo
        fig.text(0.5, 0.08, slide['text'], ha='center', va='center', wrap=True, fontsize=11)
        fig.suptitle(slide['title'], fontsize=16, fontweight='bold')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

print(f'PDF gerado: {OUTPUT_PDF}')
