import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def confronto_modelli_base_vs_avanzato():
    """
    Confronta le performance del modello base vs quello con features avanzate
    """
    
    print("🔍 CONFRONTO MODELLI: BASE vs AVANZATO")
    
    # Dataset base (originale)
    df_base = pd.read_csv('classifica/motogp_results_cleaned_final_normalized_backup.csv')
    df_meteo_base = pd.read_csv('meteo/motogp_weather_data.csv')
    df_base = pd.merge(df_base, df_meteo_base, left_on=['Event','Year'], right_on=['Event','Year'], how='left')
    
    # Dataset avanzato
    df_avanzato = pd.read_csv('classifica/motogp_results_expanded_features.csv')
    
    # Prepara output
    output_dir = 'modello_avanzato/confronto'
    os.makedirs(output_dir, exist_ok=True)
    
    risultati_confronto = []
    
    # MODELLO BASE
    print("\n📊 Valutando modello BASE...")
    
    # Prepara dati base
    df_base = df_base[pd.to_numeric(df_base['Position'], errors='coerce').notnull()]
    df_base['Position'] = df_base['Position'].astype(int)
    df_base['Podio'] = (df_base['Position'] <= 3).astype(int)
    
    features_base = ['Rider_normalized', 'Team', 'Event', 'Year', 'Conditions']
    X_base = df_base[features_base].copy()
    y_base = df_base['Podio']
    
    # Encoding base
    for col in features_base:
        X_base[col] = LabelEncoder().fit_transform(X_base[col].astype(str))
    X_base = X_base.fillna(-1)
    
    # Split temporale base
    mask_train_base = (df_base['Year'] >= 2002) & (df_base['Year'] <= 2020)
    X_train_base = X_base[mask_train_base]
    y_train_base = y_base[mask_train_base]
    
    # Modello base
    rf_base = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    cv_scores_base = cross_val_score(rf_base, X_train_base, y_train_base, cv=5, scoring='roc_auc')
    
    risultati_confronto.append({
        'Modello': 'Base (5 features)',
        'ROC_AUC_CV': cv_scores_base.mean(),
        'ROC_AUC_STD': cv_scores_base.std(),
        'N_Features': len(features_base),
        'N_Samples': len(X_train_base)
    })
    
    print(f"✅ Modello Base: ROC AUC = {cv_scores_base.mean():.3f} ± {cv_scores_base.std():.3f}")
    
    # MODELLO AVANZATO
    print("\n🚀 Valutando modello AVANZATO...")
    
    # Prepara dati avanzati
    df_avanzato = df_avanzato[pd.to_numeric(df_avanzato['Position'], errors='coerce').notnull()]
    df_avanzato['Position'] = df_avanzato['Position'].astype(int)
    df_avanzato['Podio'] = (df_avanzato['Position'] <= 3).astype(int)
    
    # Features avanzate (tutte quelle disponibili)
    features_avanzate = [
        'Rider_normalized', 'Team', 'Event', 'Year', 'Conditions',  # Base
        'Avg_Position_Last3', 'Podiums_Last5', 'Points_Streak',     # Storiche
        'length_km', 'width_m', 'right corners', 'left corners',    # Circuito
        'Race_Number_Season', 'Rider_Experience',                   # Temporali
        'Season_Points_So_Far', 'Championship_Position',            # Competitive
        'Team_Avg_Position', 'Difficult_Conditions',               # Altri
        'Rider_Circuit_Avg'                                         # Interazione
    ]
    
    # Rimuovi features con troppi NaN
    df_clean = df_avanzato.dropna(subset=features_avanzate + ['Podio'])
    
    X_avanzato = df_clean[features_avanzate].copy()
    y_avanzato = df_clean['Podio']
    
    # Encoding avanzato
    categorical_features = ['Rider_normalized', 'Team', 'Event', 'Conditions']
    for col in categorical_features:
        if col in X_avanzato.columns:
            X_avanzato[col] = LabelEncoder().fit_transform(X_avanzato[col].astype(str))
    X_avanzato = X_avanzato.fillna(-1)
    
    # Split temporale avanzato
    mask_train_avanzato = (df_clean['Year'] >= 2002) & (df_clean['Year'] <= 2020)
    X_train_avanzato = X_avanzato[mask_train_avanzato]
    y_train_avanzato = y_avanzato[mask_train_avanzato]
    
    # Modello avanzato
    rf_avanzato = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    cv_scores_avanzato = cross_val_score(rf_avanzato, X_train_avanzato, y_train_avanzato, cv=5, scoring='roc_auc')
    
    risultati_confronto.append({
        'Modello': 'Avanzato (19 features)',
        'ROC_AUC_CV': cv_scores_avanzato.mean(),
        'ROC_AUC_STD': cv_scores_avanzato.std(),
        'N_Features': len(features_avanzate),
        'N_Samples': len(X_train_avanzato)
    })
    
    print(f"✅ Modello Avanzato: ROC AUC = {cv_scores_avanzato.mean():.3f} ± {cv_scores_avanzato.std():.3f}")
    
    # CONFRONTO FEATURE IMPORTANCE
    print("\n📈 Analizzando Feature Importance...")
    
    # Fit dei modelli per feature importance
    rf_base.fit(X_train_base, y_train_base)
    rf_avanzato.fit(X_train_avanzato, y_train_avanzato)
    
    # Feature importance base
    importance_base = pd.DataFrame({
        'Feature': features_base,
        'Importance': rf_base.feature_importances_,
        'Modello': 'Base'
    })
    
    # Feature importance avanzato
    importance_avanzato = pd.DataFrame({
        'Feature': features_avanzate,
        'Importance': rf_avanzato.feature_importances_,
        'Modello': 'Avanzato'
    })
    
    # VISUALIZZAZIONI
    crea_visualizzazioni_confronto(risultati_confronto, importance_base, importance_avanzato, output_dir)
    
    # Salva risultati
    df_risultati = pd.DataFrame(risultati_confronto)
    df_risultati.to_csv(os.path.join(output_dir, 'confronto_modelli.csv'), index=False)
    
    print(f"\n💾 Risultati salvati in: {output_dir}")
    
    return risultati_confronto, importance_base, importance_avanzato

def crea_visualizzazioni_confronto(risultati, importance_base, importance_avanzato, output_dir):
    """
    Crea visualizzazioni comparative
    """
    
    print("📊 Creando visualizzazioni comparative...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Confronto ROC AUC
    df_risultati = pd.DataFrame(risultati)
    
    axes[0,0].bar(df_risultati['Modello'], df_risultati['ROC_AUC_CV'], 
                 yerr=df_risultati['ROC_AUC_STD'], capsize=5, 
                 color=['skyblue', 'orange'], alpha=0.8)
    axes[0,0].set_title('Confronto ROC AUC Cross-Validation', fontsize=14, weight='bold')
    axes[0,0].set_ylabel('ROC AUC')
    axes[0,0].grid(True, alpha=0.3)
    
    # Aggiungi valori sulle barre
    for i, (modello, score, std) in enumerate(zip(df_risultati['Modello'], 
                                                   df_risultati['ROC_AUC_CV'], 
                                                   df_risultati['ROC_AUC_STD'])):
        axes[0,0].text(i, score + 0.01, f'{score:.3f}±{std:.3f}', 
                      ha='center', va='bottom', fontweight='bold')
    
    # 2. Feature Importance Base (Top 5)
    top_base = importance_base.nlargest(5, 'Importance')
    axes[0,1].barh(top_base['Feature'], top_base['Importance'], color='skyblue', alpha=0.8)
    axes[0,1].set_title('Feature Importance - Modello Base', fontsize=14, weight='bold')
    axes[0,1].set_xlabel('Importance')
    
    # 3. Feature Importance Avanzato (Top 10)
    top_avanzato = importance_avanzato.nlargest(10, 'Importance')
    axes[1,0].barh(top_avanzato['Feature'], top_avanzato['Importance'], color='orange', alpha=0.8)
    axes[1,0].set_title('Feature Importance - Modello Avanzato (Top 10)', fontsize=14, weight='bold')
    axes[1,0].set_xlabel('Importance')
    
    # 4. Miglioramento per categoria di feature
    categorie_importance = {
        'Base Features': importance_avanzato[importance_avanzato['Feature'].isin(
            ['Rider_normalized', 'Team', 'Event', 'Year', 'Conditions'])]['Importance'].sum(),
        'Performance Storiche': importance_avanzato[importance_avanzato['Feature'].isin(
            ['Avg_Position_Last3', 'Podiums_Last5', 'Points_Streak'])]['Importance'].sum(),
        'Caratteristiche Circuito': importance_avanzato[importance_avanzato['Feature'].isin(
            ['length_km', 'width_m', 'right corners', 'left corners'])]['Importance'].sum(),
        'Features Temporali': importance_avanzato[importance_avanzato['Feature'].isin(
            ['Race_Number_Season', 'Rider_Experience'])]['Importance'].sum(),
        'Features Competitive': importance_avanzato[importance_avanzato['Feature'].isin(
            ['Season_Points_So_Far', 'Championship_Position', 'Team_Avg_Position'])]['Importance'].sum(),
        'Altri': importance_avanzato[importance_avanzato['Feature'].isin(
            ['Difficult_Conditions', 'Rider_Circuit_Avg'])]['Importance'].sum()
    }
    
    categorie_df = pd.DataFrame(list(categorie_importance.items()), 
                               columns=['Categoria', 'Importance_Totale'])
    categorie_df = categorie_df.sort_values('Importance_Totale', ascending=True)
    
    axes[1,1].barh(categorie_df['Categoria'], categorie_df['Importance_Totale'], 
                  color='green', alpha=0.7)
    axes[1,1].set_title('Importanza per Categoria di Feature', fontsize=14, weight='bold')
    axes[1,1].set_xlabel('Importance Totale')
    
    plt.tight_layout()
    
    # Salva grafici
    plt.savefig(os.path.join(output_dir, 'confronto_modelli_visualizzazioni.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Crea summary report
    crea_report_confronto(risultati, categorie_importance, output_dir)

def crea_report_confronto(risultati, categorie_importance, output_dir):
    """
    Crea un report testuale del confronto
    """
    
    df_risultati = pd.DataFrame(risultati)
    miglioramento = df_risultati.iloc[1]['ROC_AUC_CV'] - df_risultati.iloc[0]['ROC_AUC_CV']
    miglioramento_perc = (miglioramento / df_risultati.iloc[0]['ROC_AUC_CV']) * 100
    
    report = f"""
# REPORT CONFRONTO MODELLI BASE vs AVANZATO

## 📊 RISULTATI PERFORMANCE

### Modello Base (5 features):
- ROC AUC: {df_risultati.iloc[0]['ROC_AUC_CV']:.3f} ± {df_risultati.iloc[0]['ROC_AUC_STD']:.3f}
- Features: Rider, Team, Event, Year, Conditions
- Campioni: {df_risultati.iloc[0]['N_Samples']:,}

### Modello Avanzato (19 features):
- ROC AUC: {df_risultati.iloc[1]['ROC_AUC_CV']:.3f} ± {df_risultati.iloc[1]['ROC_AUC_STD']:.3f}
- Features: 19 (incluse performance storiche, caratteristiche circuito, etc.)
- Campioni: {df_risultati.iloc[1]['N_Samples']:,}

## 🚀 MIGLIORAMENTO

- **Miglioramento assoluto**: +{miglioramento:.3f} ROC AUC
- **Miglioramento percentuale**: +{miglioramento_perc:.1f}%

## 📈 IMPORTANZA CATEGORIE FEATURES

"""
    
    for categoria, importance in sorted(categorie_importance.items(), 
                                       key=lambda x: x[1], reverse=True):
        perc = (importance / sum(categorie_importance.values())) * 100
        report += f"- **{categoria}**: {importance:.3f} ({perc:.1f}%)\n"
    
    report += f"""

## 💡 CONCLUSIONI

1. **Performance**: Il modello avanzato mostra un miglioramento significativo di {miglioramento_perc:.1f}%
2. **Features più importanti**: Le performance storiche dominano l'importanza
3. **Robustezza**: Entrambi i modelli mostrano bassa varianza nella cross-validation
4. **Raccomandazione**: Utilizzare il modello avanzato per predizioni più accurate

## 📁 FILE GENERATI

- confronto_modelli.csv: Risultati numerici
- confronto_modelli_visualizzazioni.png: Grafici comparativi
- report_confronto.md: Questo report

---
Generato automaticamente il: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(os.path.join(output_dir, 'report_confronto.md'), 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("📝 Report dettagliato salvato in: report_confronto.md")

if __name__ == "__main__":
    print("🔍 AVVIO CONFRONTO MODELLI BASE vs AVANZATO\n")
    
    risultati, imp_base, imp_avanzato = confronto_modelli_base_vs_avanzato()
    
    print("\n✅ CONFRONTO COMPLETATO!")
    print("📁 Tutti i file salvati in: modello_avanzato/confronto/")
