import os
import shutil

# Definisci le sottocartelle
folders = {
    'confusion_matrix': 'confusion_matrix_',
    'metriche': [
        'barplot_confusion_percent_',
        'barplot_cv_f1score_podio',
        'barplot_cv_mse_tempo_giro',
        'barplot_distribuzione_classi',
        'barplot_precision_recall_specificity',
        'confronto_algoritmi_classificazione',
        'confronto_algoritmi_regressione',
        'metriche_classificazione',
        'metriche_regressione',
        'boxplot_errori_gap',
        'istogramma_errori_gap'
    ],
    'roc_pr': ['roc_curve_', 'precision_recall_curve_'],
    'feature_importance': ['feature_importance_', 'shap_summary_'],
    'scatter': ['scatter_gap_reale_vs_predetto', 'scatter_podio_reale_vs_predetto'],
    'risultati': ['risultati_', 'confronto_gap_reale_predetto', 'confronto_podio_reale_predetto'],
    'learning_validation': ['learning_curve_', 'validation_curve_']
}

# Crea le sottocartelle se non esistono
dir_modello = os.path.dirname(__file__)
for folder in folders:
    os.makedirs(os.path.join(dir_modello, folder), exist_ok=True)

# Sposta i file
for fname in os.listdir(dir_modello):
    fpath = os.path.join(dir_modello, fname)
    if not os.path.isfile(fpath):
        continue
    for folder, patterns in folders.items():
        if isinstance(patterns, str):
            patterns = [patterns]
        for pat in patterns:
            if fname.startswith(pat) or pat in fname:
                # Evita di spostare lo script stesso
                if fname == 'ordina_modello.py':
                    continue
                shutil.move(fpath, os.path.join(dir_modello, folder, fname))
                break

print('Ordinamento completato!')
