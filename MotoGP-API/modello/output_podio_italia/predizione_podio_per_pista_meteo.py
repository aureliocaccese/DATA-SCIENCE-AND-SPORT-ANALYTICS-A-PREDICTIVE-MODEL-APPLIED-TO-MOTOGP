import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os

# Carica dataset classifica e meteo
# Percorsi relativi corretti

df = pd.read_csv('classifica/motogp_results_cleaned_final_normalized.csv')
df_meteo = pd.read_csv('meteo/motogp_weather_data.csv')

# Adatta i nomi delle colonne per il join e le feature
# 'Grand Prix' -> 'Event', 'Weather' -> 'Conditions'

df = pd.merge(df, df_meteo, left_on=['Event','Year'], right_on=['Event','Year'], how='left')

# Filtra solo piloti classificati
mask = pd.to_numeric(df['Position'], errors='coerce').notnull()
df = df[mask]
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

# Filtra solo dati dal 2002 al 2020 per l'addestramento
mask_train = (df['Year'] >= 2002) & (df['Year'] <= 2020)
X_train = X[mask_train]
y_train = y[mask_train]
# Addestra modello Random Forest solo su dati 2002-2020
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

def predizione_podio(rider, team, event, year, conditions):
    """
    Restituisce la probabilità di podio per una combinazione pilota, team, pista, anno, meteo.
    """
    # Ricostruisci encoder
    le_rider = LabelEncoder().fit(df['Rider_normalized'].astype(str))
    le_team = LabelEncoder().fit(df['Team'].astype(str))
    le_event = LabelEncoder().fit(df['Event'].astype(str))
    le_cond = LabelEncoder().fit(df['Conditions'].astype(str))
    # Controllo valori
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
    # Esempio: predizione podio per pilota reale, pista e meteo
    esempio = df[df['Team'].notnull() & (df['Team'] != '') & df['Conditions'].notnull()].iloc[0]
    rider = esempio['Rider_normalized']
    team = esempio['Team']
    event = esempio['Event']
    year = int(esempio['Year'])
    conditions = esempio['Conditions']
    print(f"Esempio: {rider}, {team}, {event}, {year}, {conditions}")
    proba = predizione_podio(rider, team, event, year, conditions)
    print(f"Probabilità podio: {proba:.2%}")
    
    # Probabilità di podio dal 2010 al 2020 al GP d'Italia SOLO per piloti che hanno effettivamente gareggiato in ciascun anno
    anni = list(range(2010, 2021))
    # Trova il nome esatto dell'evento GP Italia nei dati
    eventi_italia = df[df['Event'].str.lower().str.contains('ita')]['Event'].unique()
    risultati = []
    if len(eventi_italia) == 0:
        print("Nessun evento Italia trovato nei dati.")
    else:
        event = eventi_italia[0]  # Prende il primo nome valido
        print(f"Probabilità podio GP Italia ({event}) dal 2010 al 2020:")
        for year in anni:
            # Piloti che hanno corso quell'anno al GP Italia
            piloti_anno = df[(df['Event'] == event) & (df['Year'] == year)]['Rider_normalized'].dropna().unique()[:5]
            if len(piloti_anno) == 0:
                print(f"Anno {year}: nessun dato disponibile")
                continue
            for rider in piloti_anno:
                riga = df[(df['Rider_normalized'] == rider) & (df['Event'] == event) & (df['Year'] == year) & df['Team'].notnull() & (df['Team'] != '') & df['Conditions'].notnull()]
                if not riga.empty:
                    team = riga.iloc[0]['Team']
                    conditions = riga.iloc[0]['Conditions']
                    try:
                        proba = predizione_podio(rider, team, event, year, conditions)
                        risultati.append({
                            'Rider': rider,
                            'Team': team,
                            'Year': year,
                            'Conditions': conditions,
                            'Prob_podio': proba
                        })
                        print(f"{rider}, {team}, {year}, {conditions}: {proba:.2%}")
                    except Exception as e:
                        print(f"{rider}, {team}, {year}, {conditions}: Errore ({e})")
                else:
                    print(f"{rider}, --, {year}, --: dati non disponibili")

    # GRAFICO: barplot Rider/Year vs Probabilità podio
    if risultati:
        df_ris = pd.DataFrame(risultati)
        # Limita ai 3 piloti con più presenze nel periodo considerato
        top_n = 3
        piloti_freq = df_ris['Rider'].value_counts().nlargest(top_n).index.tolist()
        df_ris = df_ris[df_ris['Rider'].isin(piloti_freq)]
        # Crea cartella output se non esiste
        output_dir = os.path.join('modello', 'output_podio_italia')
        os.makedirs(output_dir, exist_ok=True)
        # Salva CSV
        csv_path = os.path.join(output_dir, 'probabilita_podio_gp_italia_2010_2020.csv')
        df_ris.to_csv(csv_path, index=False)
        # GRAFICO MIGLIORATO: colori per pilota
        import seaborn as sns
        # Ordina per anno e rider
        df_ris = df_ris.sort_values(['Year', 'Rider'])
        # Crea la colonna label DOPO l'ordinamento
        df_ris['label'] = df_ris['Rider'] + ' (' + df_ris['Year'].astype(str) + ')'
        # Colori diversi per pilota
        piloti_unici = df_ris['Rider'].unique()
        palette = sns.color_palette('tab10', n_colors=len(piloti_unici))
        pilota2col = {pilota: palette[i % len(palette)] for i, pilota in enumerate(piloti_unici)}
        bar_colors = df_ris['Rider'].map(pilota2col)
        # Plot
        plt.figure(figsize=(20, 7))
        bars = plt.bar(df_ris['label'], df_ris['Prob_podio'], color=bar_colors, edgecolor='black')
        plt.ylabel('Probabilità podio', fontsize=13)
        plt.xlabel('Pilota (Anno)', fontsize=13)
        plt.title(f'Probabilità podio GP Italia ({event}) dal 2010 al 2020', fontsize=15, weight="bold")
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=11)
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        # Valori sulle barre
        for bar, val in zip(bars, df_ris['Prob_podio']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.0%}", ha='center', va='bottom', fontsize=9)
        # Legenda per pilota
        from matplotlib.patches import Patch
        legend_handles = [Patch(color=pilota2col[pilota], label=pilota) for pilota in piloti_unici]
        plt.legend(handles=legend_handles, title='Pilota', bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.tight_layout()
        # Salva grafico
        plot_path = os.path.join(output_dir, 'grafico_probabilita_podio_gp_italia_2010_2020.png')
        plt.savefig(plot_path, dpi=150)
        plt.show()
        print(f"Risultati salvati in: {csv_path}\nGrafico salvato in: {plot_path}")

        # --- PREVISIONE DAL 2020 AL 2030 SUGLI STESSI PILOTI ---
        anni_pred = list(range(2020, 2031))
        piloti_pred = piloti_freq  # stessi piloti top_n
        risultati_pred = []
        import numpy as np
        rng = np.random.default_rng(42)
        for rider in piloti_pred:
            # Team e condizioni meteo storici per il pilota al GP Italia
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
                # Estrai team e condizione meteo casuali tra quelli storici
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
            # Salva CSV
            csv_pred_path = os.path.join(output_dir, 'previsione_probabilita_podio_gp_italia_2020_2030.csv')
            df_pred.to_csv(csv_pred_path, index=False)
            # GRAFICO
            import seaborn as sns
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
