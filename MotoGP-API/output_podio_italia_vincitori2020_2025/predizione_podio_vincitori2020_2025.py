import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Carica dataset classifica e meteo (percorsi relativi alla root del workspace)
df = pd.read_csv('classifica/motogp_results_cleaned_final_normalized.csv')
df_meteo = pd.read_csv('meteo/motogp_weather_data.csv')
df = pd.merge(df, df_meteo, left_on=['Event','Year'], right_on=['Event','Year'], how='left')

# Filtra solo piloti classificati
df = df[pd.to_numeric(df['Position'], errors='coerce').notnull()]
df['Position'] = df['Position'].astype(int)

# Target: podio (1 se posizione <=3, 0 altrimenti)
df['Podio'] = (df['Position'] <= 3).astype(int)

# Feature: pilota, team, pista, anno, condizioni meteo principali
features = ['Rider_normalized', 'Team', 'Event', 'Year', 'Conditions']
X = df[features].copy()
y = df['Podio']

# Encoding variabili categoriche
for col in ['Rider_normalized', 'Team', 'Event', 'Conditions']:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
X = X.fillna(-1)

# Addestra modello solo su dati dal 2002 al 2020
mask_train = (df['Year'] >= 2002) & (df['Year'] <= 2020)
X_train = X[mask_train]
y_train = y[mask_train]
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

def predizione_podio(rider, team, event, year, conditions):
    le_rider = LabelEncoder().fit(df['Rider_normalized'].astype(str))
    le_team = LabelEncoder().fit(df['Team'].astype(str))
    le_event = LabelEncoder().fit(df['Event'].astype(str))
    le_cond = LabelEncoder().fit(df['Conditions'].astype(str))
    if rider not in le_rider.classes_:
        raise ValueError(f"Rider '{rider}' non presente nei dati")
    if team not in le_team.classes_:
        raise ValueError(f"Team '{team}' non presente nei dati")
    if event not in le_event.classes_:
        raise ValueError(f"Event '{event}' non presente nei dati")
    if conditions not in le_cond.classes_:
        raise ValueError(f"Condizione meteo '{conditions}' non presente nei dati")
    X_input = pd.DataFrame({
        'Rider_normalized': [rider],
        'Team': [team],
        'Event': [event],
        'Year': [year],
        'Conditions': [conditions]
    })
    X_input['Rider_normalized'] = le_rider.transform(X_input['Rider_normalized'].astype(str))
    X_input['Team'] = le_team.transform(X_input['Team'].astype(str))
    X_input['Event'] = le_event.transform(X_input['Event'].astype(str))
    X_input['Conditions'] = le_cond.transform(X_input['Conditions'].astype(str))
    X_input = X_input.fillna(-1)
    proba = clf.predict_proba(X_input)[0][1]
    return proba

if __name__ == "__main__":
    # --- IMPORTANZA DELLE FEATURE DEL MODELLO ---
    print("\nImportanza delle feature nel modello Random Forest:")
    for feat, imp in zip(features, clf.feature_importances_):
        print(f"{feat}: {imp:.3f}")

    # --- RISULTATI STORICI DI OLIVEIRA AL GP D'ITALIA ---
    rider_oliv = 'Oliveira'
    eventi_italia = df[df['Event'].str.lower().str.contains('ita')]['Event'].unique()
    if len(eventi_italia) > 0:
        event_ita = eventi_italia[0]
        storico_oliv = df[(df['Rider_normalized'].str.contains(rider_oliv, case=False)) & (df['Event'] == event_ita)]
        if not storico_oliv.empty:
            print(f"\nRisultati storici di Oliveira al GP d'Italia:")
            print(storico_oliv[['Year','Team','Conditions','Position','Podio']].sort_values('Year'))
        else:
            print("\nNessun risultato storico di Oliveira al GP d'Italia trovato nei dati.")
    else:
        print("\nNessun evento Italia trovato nei dati.")
    # Trova i piloti con più vittorie (Position==1) dal 2020 al 2025
    mask_vittorie = (df['Year'] >= 2020) & (df['Year'] <= 2025) & (df['Position'] == 1)
    top_vincitori = df[mask_vittorie]['Rider_normalized'].value_counts().nlargest(3).index.tolist()
    print(f"Piloti con più vittorie 2020-2025: {top_vincitori}")
    # Analisi e previsione podio GP Italia 2020-2030 per questi piloti
    anni_pred = list(range(2020, 2031))
    eventi_italia = df[df['Event'].str.lower().str.contains('ita')]['Event'].unique()
    if len(eventi_italia) == 0:
        print("Nessun evento Italia trovato nei dati.")
        exit()
    event = eventi_italia[0]
    risultati_pred = []
    rng = np.random.default_rng(42)
    for rider in top_vincitori:
        storico = df[(df['Rider_normalized'] == rider) & (df['Event'] == event)]
        if storico.empty:
            continue
        team_possibili = storico['Team'].dropna().unique()
        if len(team_possibili) == 0:
            team_possibili = ['']
        condizioni_possibili = storico['Conditions'].dropna().unique()
        if len(condizioni_possibili) == 0:
            condizioni_possibili = ['Dry']
        for year in anni_pred:
            team = rng.choice(team_possibili)
            conditions = rng.choice(condizioni_possibili)
            try:
                proba = predizione_podio(rider, team, event, year, conditions)
                risultati_pred.append({
                    'Rider': rider,
                    'Team': team,
                    'Year': year,
                    'Conditions': conditions,
                    'Prob_podio': proba
                })
            except Exception as e:
                risultati_pred.append({
                    'Rider': rider,
                    'Team': team,
                    'Year': year,
                    'Conditions': conditions,
                    'Prob_podio': None
                })
    if risultati_pred:
        df_pred = pd.DataFrame(risultati_pred)
        output_dir = os.path.join('output_podio_italia_vincitori2020_2025')
        os.makedirs(output_dir, exist_ok=True)
        csv_pred_path = os.path.join(output_dir, 'previsione_probabilita_podio_gp_italia_2020_2030.csv')
        df_pred.to_csv(csv_pred_path, index=False)
        df_pred = df_pred.dropna(subset=['Prob_podio'])
        df_pred = df_pred.sort_values(['Year', 'Rider'])
        df_pred['label'] = df_pred['Rider'] + ' (' + df_pred['Year'].astype(str) + ')'
        piloti_unici_pred = df_pred['Rider'].unique()
        palette_pred = sns.color_palette('tab10', n_colors=len(piloti_unici_pred))
        pilota2col_pred = {pilota: palette_pred[i % len(palette_pred)] for i, pilota in enumerate(piloti_unici_pred)}
        bar_colors_pred = df_pred['Rider'].map(pilota2col_pred)
        plt.figure(figsize=(20, 7))
        bars = plt.bar(df_pred['label'], df_pred['Prob_podio'], color=bar_colors_pred, edgecolor='black')
        plt.ylabel('Probabilità podio', fontsize=13)
        plt.xlabel('Pilota (Anno)', fontsize=13)
        plt.title(f'Previsione probabilità podio GP Italia ({event}) 2020-2030', fontsize=15, weight="bold")
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=11)
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        for bar, val in zip(bars, df_pred['Prob_podio']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.0%}", ha='center', va='bottom', fontsize=9)
        from matplotlib.patches import Patch
        legend_handles = [Patch(color=pilota2col_pred[pilota], label=pilota) for pilota in piloti_unici_pred]
        plt.legend(handles=legend_handles, title='Pilota', bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.tight_layout()
        plot_pred_path = os.path.join(output_dir, 'grafico_previsione_probabilita_podio_gp_italia_2020_2030.png')
        plt.savefig(plot_pred_path, dpi=150)
        plt.show()
        print(f"Previsioni salvate in: {csv_pred_path}\nGrafico salvato in: {plot_pred_path}")
