import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def predizione_podio_avanzata():
    """
    Modello di predizione podio con features espanse
    """
    
    # Carica dataset espanso
    df = pd.read_csv('classifica/motogp_results_expanded_features.csv')
    
    print("🚀 MODELLO AVANZATO CON FEATURES ESPANSE")
    print(f"📊 Dataset: {len(df)} righe, {len(df.columns)} colonne")
    
    # Filtra solo piloti classificati
    df = df[pd.to_numeric(df['Position'], errors='coerce').notnull()]
    df['Position'] = df['Position'].astype(int)
    df['Podio'] = (df['Position'] <= 3).astype(int)
    
    # Features per il modello (escludendo quelle con troppi NaN)
    features_numeriche = [
        'Year',
        'Avg_Position_Last3',
        'Podiums_Last5', 
        'Points_Streak',
        'length_km',
        'width_m',
        'right corners',
        'left corners',
        'Race_Number_Season',
        'Rider_Experience',
        'Season_Points_So_Far',
        'Championship_Position',
        'Team_Avg_Position',
        'Difficult_Conditions',
        'Rider_Circuit_Avg'
    ]
    
    features_categoriche = [
        'Rider_normalized',
        'Team', 
        'Event',
        'Conditions'
    ]
    
    features = features_categoriche + features_numeriche
    
    # Prepara i dati
    df_clean = df.dropna(subset=features + ['Podio']).copy()
    
    X = df_clean[features].copy()
    y = df_clean['Podio']
    
    # Encoding variabili categoriche
    for col in features_categoriche:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    X = X.fillna(-1)
    
    print(f"✅ Dati puliti: {len(df_clean)} righe")
    print(f"🎯 Distribuzione target: {y.value_counts().to_dict()}")
    
    # Split train/test
    mask_train = (df_clean['Year'] >= 2002) & (df_clean['Year'] <= 2020)
    X_train = X[mask_train]
    y_train = y[mask_train]
    X_test = X[~mask_train]
    y_test = y[~mask_train]
    
    print(f"🏃 Training: {len(X_train)} righe | Test: {len(X_test)} righe")
    
    # Bilanciamento con SMOTE
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    
    print(f"⚖️ Dopo SMOTE: {len(X_train_bal)} righe")
    
    # Modelli con ottimizzazione
    modelli = {}
    
    # Random Forest
    print("\n🌳 Training Random Forest...")
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced']
    }
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_params, cv=5, scoring='roc_auc', n_jobs=-1
    )
    rf_grid.fit(X_train_bal, y_train_bal)
    modelli['RandomForest'] = rf_grid.best_estimator_
    
    # LightGBM
    print("💡 Training LightGBM...")
    lgbm_params = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.1, 0.05]
    }
    lgbm_grid = RandomizedSearchCV(
        LGBMClassifier(random_state=42),
        lgbm_params, cv=5, scoring='roc_auc', n_jobs=-1, n_iter=10
    )
    lgbm_grid.fit(X_train_bal, y_train_bal)
    modelli['LightGBM'] = lgbm_grid.best_estimator_
    
    # XGBoost
    print("🚀 Training XGBoost...")
    xgb_params = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.1, 0.05]
    }
    xgb_grid = RandomizedSearchCV(
        XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        xgb_params, cv=5, scoring='roc_auc', n_jobs=-1, n_iter=10
    )
    xgb_grid.fit(X_train_bal, y_train_bal)
    modelli['XGBoost'] = xgb_grid.best_estimator_
    
    # Validazione incrociata
    print("\n📊 VALIDAZIONE INCROCIATA:")
    cv_results = {}
    for nome, model in modelli.items():
        cv_scores = cross_val_score(model, X_train_bal, y_train_bal, cv=5, scoring='roc_auc')
        cv_results[nome] = cv_scores.mean()
        print(f"{nome}: ROC AUC = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Test set evaluation
    print("\n🎯 PERFORMANCE SU TEST SET:")
    test_results = {}
    for nome, model in modelli.items():
        if len(X_test) > 0:
            test_score = model.score(X_test, y_test)
            test_results[nome] = test_score
            print(f"{nome}: Accuratezza = {test_score:.3f}")
    
    # Feature importance del miglior modello
    best_model_name = max(cv_results, key=cv_results.get)
    best_model = modelli[best_model_name]
    
    print(f"\n🏆 Miglior modello: {best_model_name}")
    
    return modelli, features, df_clean, cv_results, test_results

def genera_predizioni_avanzate(modelli, features, df_clean):
    """
    Genera predizioni usando il modello ensemble avanzato
    """
    
    print("\n🔮 GENERAZIONE PREDIZIONI AVANZATE...")
    
    # Crea output directory
    output_dir = 'modello_avanzato/output_predizioni'
    os.makedirs(output_dir, exist_ok=True)
    
    # GP Italia con nuove features
    eventi_italia = df_clean[df_clean['Event'].str.lower().str.contains('ita', na=False)]['Event'].unique()
    
    if len(eventi_italia) == 0:
        print("❌ Nessun evento Italia trovato")
        return
    
    event = eventi_italia[0]
    print(f"🏁 Evento selezionato: {event}")
    
    # Anni per analisi
    anni = list(range(2010, 2021))
    risultati = []
    
    # Encoding per predizioni
    le_rider = LabelEncoder().fit(df_clean['Rider_normalized'].astype(str))
    le_team = LabelEncoder().fit(df_clean['Team'].astype(str))
    le_event = LabelEncoder().fit(df_clean['Event'].astype(str))
    le_cond = LabelEncoder().fit(df_clean['Conditions'].astype(str))
    
    for year in anni:
        # Piloti per quell'anno
        piloti_anno = df_clean[
            (df_clean['Event'] == event) & 
            (df_clean['Year'] == year)
        ]['Rider_normalized'].dropna().unique()[:5]
        
        for rider in piloti_anno:
            # Trova dati reali per quel pilota/anno/evento
            riga = df_clean[
                (df_clean['Rider_normalized'] == rider) & 
                (df_clean['Event'] == event) & 
                (df_clean['Year'] == year)
            ]
            
            if not riga.empty:
                # Usa dati reali dove disponibili
                sample = riga.iloc[0]
                
                # Crea input per predizione
                X_input = pd.DataFrame({
                    'Rider_normalized': [le_rider.transform([sample['Rider_normalized']])[0]],
                    'Team': [le_team.transform([sample['Team']])[0]],
                    'Event': [le_event.transform([sample['Event']])[0]],
                    'Conditions': [le_cond.transform([sample['Conditions']])[0]],
                    'Year': [sample['Year']],
                    'Avg_Position_Last3': [sample.get('Avg_Position_Last3', 5.0)],
                    'Podiums_Last5': [sample.get('Podiums_Last5', 1.0)],
                    'Points_Streak': [sample.get('Points_Streak', 0.0)],
                    'length_km': [sample.get('length_km', 5.0)],
                    'width_m': [sample.get('width_m', 12.0)],
                    'right corners': [sample.get('right corners', 6)],
                    'left corners': [sample.get('left corners', 6)],
                    'Race_Number_Season': [sample.get('Race_Number_Season', 10)],
                    'Rider_Experience': [sample.get('Rider_Experience', 5)],
                    'Season_Points_So_Far': [sample.get('Season_Points_So_Far', 100)],
                    'Championship_Position': [sample.get('Championship_Position', 5)],
                    'Team_Avg_Position': [sample.get('Team_Avg_Position', 5.0)],
                    'Difficult_Conditions': [sample.get('Difficult_Conditions', 0)],
                    'Rider_Circuit_Avg': [sample.get('Rider_Circuit_Avg', 5.0)]
                })
                
                X_input = X_input.fillna(-1)
                
                # Predizioni ensemble (media dei tre modelli)
                predizioni = []
                for nome, model in modelli.items():
                    try:
                        proba = model.predict_proba(X_input)[0][1]
                        predizioni.append(proba)
                    except:
                        predizioni.append(0.5)  # Default
                
                prob_ensemble = np.mean(predizioni)
                
                risultati.append({
                    'Rider': sample['Rider_normalized'],
                    'Team': sample['Team'],
                    'Year': year,
                    'Conditions': sample['Conditions'],
                    'Prob_podio_avanzata': prob_ensemble,
                    'Posizione_reale': sample.get('Position', None)
                })
    
    if risultati:
        # Salva risultati
        df_risultati = pd.DataFrame(risultati)
        csv_path = os.path.join(output_dir, 'predizioni_avanzate_gp_italia.csv')
        df_risultati.to_csv(csv_path, index=False)
        
        # Visualizzazione
        visualizza_predizioni_avanzate(df_risultati, output_dir)
        
        print(f"💾 Predizioni salvate: {csv_path}")
        
        return df_risultati
    
    return None

def visualizza_predizioni_avanzate(df_risultati, output_dir):
    """
    Crea visualizzazioni avanzate delle predizioni
    """
    
    print("📊 Creando visualizzazioni avanzate...")
    
    # Prendi top 3 piloti
    top_piloti = df_risultati['Rider'].value_counts().head(3).index.tolist()
    df_plot = df_risultati[df_risultati['Rider'].isin(top_piloti)].copy()
    
    # Grafico probabilità vs posizione reale
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Probabilità nel tempo
    plt.subplot(2, 2, 1)
    for rider in top_piloti:
        rider_data = df_plot[df_plot['Rider'] == rider]
        plt.plot(rider_data['Year'], rider_data['Prob_podio_avanzata'], 
                marker='o', label=rider, linewidth=2, markersize=6)
    
    plt.title('Evoluzione Probabilità Podio (Modello Avanzato)', fontsize=14, weight='bold')
    plt.xlabel('Anno')
    plt.ylabel('Probabilità Podio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Confronto probabilità vs realtà
    plt.subplot(2, 2, 2)
    df_valid = df_plot.dropna(subset=['Posizione_reale'])
    df_valid['Podio_reale'] = (df_valid['Posizione_reale'] <= 3).astype(int)
    
    scatter_colors = ['red' if x == 1 else 'blue' for x in df_valid['Podio_reale']]
    plt.scatter(df_valid['Prob_podio_avanzata'], df_valid['Posizione_reale'], 
               c=scatter_colors, alpha=0.6, s=50)
    plt.axhline(y=3.5, color='green', linestyle='--', alpha=0.7, label='Soglia Podio')
    plt.title('Probabilità vs Posizione Reale', fontsize=14, weight='bold')
    plt.xlabel('Probabilità Podio Predetta')
    plt.ylabel('Posizione Reale')
    plt.legend(['Non Podio', 'Podio', 'Soglia Podio'])
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Distribuzione probabilità per pilota
    plt.subplot(2, 2, 3)
    df_plot.boxplot(column='Prob_podio_avanzata', by='Rider', ax=plt.gca())
    plt.title('Distribuzione Probabilità per Pilota', fontsize=14, weight='bold')
    plt.suptitle('')  # Rimuove il titolo automatico
    plt.xticks(rotation=45)
    
    # Subplot 4: Heatmap anno-pilota
    plt.subplot(2, 2, 4)
    pivot_data = df_plot.pivot(index='Rider', columns='Year', values='Prob_podio_avanzata')
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd', cbar_kws={'label': 'Prob. Podio'})
    plt.title('Heatmap Probabilità Podio', fontsize=14, weight='bold')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'analisi_predizioni_avanzate.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Grafici salvati: {plot_path}")

if __name__ == "__main__":
    print("🚀 AVVIO MODELLO AVANZATO CON FEATURES ESPANSE\n")
    
    # Addestra modelli
    modelli, features, df_clean, cv_results, test_results = predizione_podio_avanzata()
    
    # Genera predizioni
    risultati = genera_predizioni_avanzate(modelli, features, df_clean)
    
    print("\n✅ PROCESSO COMPLETATO!")
    print("📁 File generati nella cartella: modello_avanzato/output_predizioni/")
