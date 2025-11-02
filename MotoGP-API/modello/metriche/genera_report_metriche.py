import os
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet


def add_title(story, text):
    styles = getSampleStyleSheet()
    story.append(Paragraph(f"<b>{text}</b>", styles['Title']))
    story.append(Spacer(1, 0.5 * cm))


def add_heading(story, text):
    styles = getSampleStyleSheet()
    story.append(Paragraph(f"<b>{text}</b>", styles['Heading2']))
    story.append(Spacer(1, 0.3 * cm))


def add_paragraph(story, text):
    styles = getSampleStyleSheet()
    story.append(Paragraph(text, styles['BodyText']))
    story.append(Spacer(1, 0.2 * cm))


def add_image(story, img_path, width=16 * cm):
    if os.path.exists(img_path):
        story.append(Image(img_path, width=width, height=width * 0.6))
        story.append(Spacer(1, 0.3 * cm))


def add_table(story, df: pd.DataFrame, title: str):
    add_heading(story, title)
    data = [df.columns.tolist()] + df.values.tolist()
    tbl = Table(data, hAlign='LEFT')
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightyellow]),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.4 * cm))


def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))  # modello/
    metriche_dir = os.path.join(base_dir, 'metriche')
    out_pdf = os.path.join(metriche_dir, 'report_metriche.pdf')

    story = []
    add_title(story, 'Report Metriche Modelli MotoGP')
    add_paragraph(story, 'Questo report raccoglie metriche e figure generate dagli script del progetto, includendo classificazione (podio) e regressione (gap/tempo).')

    # Tabelle metriche
    class_csv = os.path.join(metriche_dir, 'metriche_classificazione.csv')
    reg_csv = os.path.join(metriche_dir, 'metriche_regressione.csv')
    if os.path.exists(class_csv):
        df_class = pd.read_csv(class_csv)
        # Arrotonda numeri per leggibilità
        for col in df_class.columns:
            if col != 'Algoritmo':
                df_class[col] = df_class[col].astype(float).round(3)
        add_table(story, df_class, 'Metriche Classificazione (test split)')
    if os.path.exists(reg_csv):
        df_reg = pd.read_csv(reg_csv)
        for col in df_reg.columns:
            if col != 'Algoritmo':
                df_reg[col] = df_reg[col].astype(float).round(3)
        add_table(story, df_reg, 'Metriche Regressione (test split)')

    # Barplot CV
    add_heading(story, 'Validazione Incrociata (5-fold)')
    add_image(story, os.path.join(metriche_dir, 'barplot_cv_f1score_podio.png'))
    add_image(story, os.path.join(metriche_dir, 'barplot_cv_mse_tempo_giro.png'))

    # Classificazione: ROC/PR, Confusion Matrix, Confronti
    add_heading(story, 'Classificazione - Grafici')
    add_image(story, os.path.join(base_dir, 'roc_curve_confronto_algoritmi.png'))
    add_image(story, os.path.join(base_dir, 'precision_recall_curve_confronto_algoritmi.png'))
    add_image(story, os.path.join(base_dir, 'confusion_matrix_podio.png'))
    add_image(story, os.path.join(metriche_dir, 'confronto_algoritmi_classificazione.png'))
    add_image(story, os.path.join(metriche_dir, 'barplot_precision_recall_specificity.png'))

    # Regressione: errori e scatter
    add_heading(story, 'Regressione - Grafici')
    add_image(story, os.path.join(metriche_dir, 'confronto_algoritmi_regressione.png'))
    add_image(story, os.path.join(metriche_dir, 'boxplot_errori_gap.png'))
    add_image(story, os.path.join(metriche_dir, 'istogramma_errori_gap.png'))
    add_image(story, os.path.join(base_dir, 'scatter_gap_reale_vs_predetto.png'))
    add_image(story, os.path.join(base_dir, 'scatter_gap_reale_vs_predetto_confronto.png'))

    # Learning & Validation curves
    add_heading(story, 'Apprendimento e Validazione')
    add_image(story, os.path.join(base_dir, 'learning_curve_best_model.png'))
    add_image(story, os.path.join(base_dir, 'validation_curve_random_forest.png'))

    # Feature importance / SHAP
    add_heading(story, 'Interpretabilità')
    add_image(story, os.path.join(base_dir, 'feature_importance_random_forest.png'))
    add_image(story, os.path.join(base_dir, 'feature_importance_gradient_boosting.png'))
    add_image(story, os.path.join(base_dir, 'shap_summary_random_forest.png'))

    doc = SimpleDocTemplate(out_pdf, pagesize=A4, title='Report Metriche Modelli MotoGP')
    doc.build(story)
    print(f'Report generato: {out_pdf}')


if __name__ == '__main__':
    main()
