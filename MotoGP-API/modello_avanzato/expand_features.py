import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def espandi_features():
    """
    Espande il dataset con nuove features per migliorare le predizioni
    """
    
    # Carica i dati esistenti
    df = pd.read_csv('classifica/motogp_results_cleaned_final_normalized.csv')
    df_meteo = pd.read_csv('meteo/motogp_weather_data.csv')
    df_piste = pd.read_csv('piste/piste/dati_piste.csv')
    
    # Merge con dati meteo
    df = pd.merge(df, df_meteo, left_on=['Event','Year'], right_on=['Event','Year'], how='left')
    
    print("🔧 CREAZIONE NUOVE FEATURES...")
    
    # 1. FEATURES STORICHE DEL PILOTA
    print("📊 Aggiungendo performance storiche...")
    
    # Media posizioni ultime 3 gare
    df = df.sort_values(['Rider_normalized', 'Year', 'Event'])
    df['Avg_Position_Last3'] = df.groupby('Rider_normalized')['Position'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    
    # Numero di podi nelle ultime 5 gare
    df['Podiums_Last5'] = df.groupby('Rider_normalized')['Position'].transform(
        lambda x: (x.shift(1).rolling(window=5, min_periods=1).apply(
            lambda pos: (pos <= 3).sum()
        ))
    )
    
    # Streak di gare consecutive a punti
    df['Points_Streak'] = df.groupby('Rider_normalized')['Position'].transform(
        lambda x: (x <= 15).groupby((x > 15).cumsum()).cumcount()
    )
    
    # 2. FEATURES CIRCUITO
    print("🏁 Aggiungendo caratteristiche circuiti...")
    
    # Prepara dati piste (pulisci e normalizza)
    df_piste_clean = df_piste.copy()
    
    # Estrai lunghezza numerica
    df_piste_clean['length_km'] = df_piste_clean['length'].str.extract(r'([\d,]+)')[0].str.replace(',', '.').astype(float)
    
    # Estrai larghezza numerica
    df_piste_clean['width_m'] = df_piste_clean['width'].str.extract(r'(\d+)')[0].astype(float)
    
    # Mappa eventi ai circuiti
    event_mapping = {
        'ita': 'ita_circuit_information.pdf',
        'arg': 'arg_circuit_information.pdf',
        'esp': 'esp_circuit_information.pdf',
        'ger': 'ger_circuit_information.pdf',
        'aut': 'aut_circuit_information.pdf',
        'usa': 'usa_circuit_information.pdf',
        'hun': 'hun_circuit_information.pdf',
        'cze': 'cze_circuit_information.pdf',
        'qat': 'qat_circuit_information.pdf'
    }
    
    # Merge con caratteristiche circuiti
    df['circuit_file'] = df['Event'].map(event_mapping)
    df = pd.merge(df, df_piste_clean[['file', 'length_km', 'width_m', 'right corners', 'left corners']], 
                  left_on='circuit_file', right_on='file', how='left')
    
    # 3. FEATURES TEMPORALI
    print("📅 Aggiungendo features temporali...")
    
    # Gara numero nella stagione
    df['Race_Number_Season'] = df.groupby(['Year']).cumcount() + 1
    
    # Esperienza del pilota (anni in MotoGP)
    df['Rider_Experience'] = df.groupby('Rider_normalized')['Year'].transform(
        lambda x: x - x.min() + 1
    )
    
    # 4. FEATURES COMPETITIVE
    print("🏆 Aggiungendo features competitive...")
    
    # Punti totali stagione fino a quel momento
    df['Points_numeric'] = pd.to_numeric(df['Points'], errors='coerce').fillna(0)
    df['Season_Points_So_Far'] = df.groupby(['Rider_normalized', 'Year'])['Points_numeric'].transform('cumsum')
    
    # Posizione in classifica generale
    df['Championship_Position'] = df.groupby(['Year', 'Race_Number_Season'])['Season_Points_So_Far'].rank(
        method='dense', ascending=False
    )
    
    # Performance relativa del team nella gara
    df['Team_Avg_Position'] = df.groupby(['Team', 'Event', 'Year'])['Position'].transform('mean')
    
    # 5. FEATURES METEO AVANZATE
    print("🌤️ Aggiungendo features meteo avanzate...")
    
    # Flag condizioni difficili
    df['Difficult_Conditions'] = df['Conditions'].isin(['Rain', 'Wet', 'Mixed']).astype(int)
    
    # 6. FEATURES INTERAZIONE
    print("🔗 Aggiungendo features di interazione...")
    
    # Interazione pilota-circuito (performance storica su quel circuito)
    df['Rider_Circuit_Avg'] = df.groupby(['Rider_normalized', 'Event'])['Position'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    
    # Filtra e prepara per il modello
    features_complete = df.dropna(subset=['Position']).copy()
    features_complete['Position'] = features_complete['Position'].astype(int)
    features_complete['Podio'] = (features_complete['Position'] <= 3).astype(int)
    
    # Lista delle nuove features
    new_features = [
        'Rider_normalized', 'Team', 'Event', 'Year', 'Conditions',  # Originali
        'Avg_Position_Last3', 'Podiums_Last5', 'Points_Streak',     # Storiche
        'length_km', 'width_m', 'right corners', 'left corners',    # Circuito
        'Race_Number_Season', 'Rider_Experience',                   # Temporali
        'Championship_Position', 'Team_Avg_Position',               # Competitive
        'Difficult_Conditions',                                     # Meteo
        'Rider_Circuit_Avg'                                         # Interazione
    ]
    
    print(f"✅ Dataset espanso: {len(new_features)} features totali")
    print(f"📊 Righe con dati completi: {len(features_complete)}")
    
    return features_complete, new_features

def valuta_importance_features(df, features, target='Podio'):
    """
    Valuta l'importanza delle nuove features
    """
    print("\n🔍 ANALISI IMPORTANZA FEATURES...")
    
    # Prepara i dati
    X = df[features].copy()
    y = df[target]
    
    # Encoding categoriche
    categorical_features = ['Rider_normalized', 'Team', 'Event', 'Conditions']
    for col in categorical_features:
        if col in X.columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    # Rimuovi NaN
    X = X.fillna(-1)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Modello Random Forest per feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Feature importance (Gini) – mantiene compatibilità con file esistente
    importance = pd.DataFrame({
        'Feature': features,
        'Gini_Importance': rf.feature_importances_
    }).sort_values('Gini_Importance', ascending=False)

    # Permutation importance su test set (più interpretabile), con F1
    perm = permutation_importance(
        rf, X_test, y_test, n_repeats=25, random_state=42, scoring='f1'
    )
    perm_df = pd.DataFrame({
        'Feature': X_test.columns,
        'PI_Mean': perm.importances_mean,
        'PI_Std': perm.importances_std
    })
    # Ordina per impatto medio (clippa negativi a 0)
    perm_df['PI_Mean_Clipped'] = perm_df['PI_Mean'].clip(lower=0)
    perm_df = perm_df.sort_values('PI_Mean_Clipped', ascending=False)

    print("\n📈 TOP 10 PERMUTATION IMPORTANCE (ΔF1 medio):")
    print(perm_df[['Feature','PI_Mean']].head(10).to_string(index=False))

    # Grafico Gini Top 15 (come in origine) per il PNG principale
    # Assicura colonna 'Importance' per il plot principale
    importance_plot_df = importance.copy()
    if 'Importance' not in importance_plot_df.columns and 'Gini_Importance' in importance_plot_df.columns:
        importance_plot_df = importance_plot_df.rename(columns={'Gini_Importance': 'Importance'})
    gini_top15 = importance_plot_df.head(15).iloc[::-1]
    plt.figure(figsize=(12, 8))
    sns.barplot(data=gini_top15, x='Importance', y='Feature', color='#1f77b4')
    plt.title('Feature Importance - Top 15')
    plt.tight_layout()
    plt.savefig('modello/feature_importance_expanded.png', dpi=300)
    plt.close()

    # Grafico migliorato opzionale: permutation importance con barre d'errore
    top15 = perm_df.head(15).iloc[::-1]
    plt.figure(figsize=(12, 8))
    plt.barh(top15['Feature'], top15['PI_Mean_Clipped'], xerr=top15['PI_Std'], color='#1f77b4', alpha=0.9, ecolor='black', capsize=3)
    plt.xlabel('Decremento medio F1 (permutation)')
    plt.ylabel('Feature')
    plt.title('Permutation Feature Importance (test) – Top 15')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('modello/feature_importance_expanded_permutation.png', dpi=300)
    for y, v in enumerate(top15['PI_Mean_Clipped']):
        plt.text(v + max(top15['PI_Mean_Clipped'])*0.01, y, f"{v:.3f}", va='center')
    plt.savefig('modello/feature_importance_expanded_permutation_annotated.png', dpi=300)
    plt.close()
    
    # Accuratezza modello
    accuracy = rf.score(X_test, y_test)
    print(f"\n🎯 Accuratezza con features espanse: {accuracy:.3f}")
    
    # Salvataggi CSV
    importance.rename(columns={'Gini_Importance':'Importance'}, inplace=True)
    importance.to_csv('modello/feature_importance_gini.csv', index=False)
    perm_df[['Feature','PI_Mean','PI_Std']].to_csv('modello/feature_importance_permutation.csv', index=False)

    return rf, importance

if __name__ == "__main__":
    # Espandi il dataset
    df_expanded, features_list = espandi_features()
    
    # Salva il dataset espanso (nuovo file)
    df_expanded.to_csv('classifica/motogp_results_expanded_features.csv', index=False)
    print(f"\n💾 Dataset espanso salvato: classifica/motogp_results_expanded_features.csv")
    print(f"📁 Backup originale: classifica/motogp_results_cleaned_final_normalized_backup.csv")
    
    # Valuta le nuove features
    model, importance_df = valuta_importance_features(df_expanded, features_list)
    
    # Salva feature importance
    importance_df.to_csv('modello/feature_importance.csv', index=False)
    print(f"💾 Feature importance salvata: modello/feature_importance.csv")
