# Estrai la parte del nome pilota dalla colonna Rider e aggiungi una colonna 'Nome'


import pandas as pd
import re

# Carica il file CSV
df = pd.read_csv("/Users/aurelio/Desktop/TESI/MotoGP-API/motogp_results_.csv")


# Rimuovi sempre la prima colonna (indipendentemente dal nome)
df = df.iloc[:, 1:]



# Stampa i nomi delle colonne per debug
print('Colonne trovate nel file:', list(df.columns))

# Assegna i nomi solo se il numero corrisponde
colonne_desiderate = ['Position', 'Race Number', 'Points', 'Rider', 'Time/Gap', 'Grand Prix', 'Year', 'Event']
if len(df.columns) == len(colonne_desiderate):
    df.columns = colonne_desiderate
    df = df[colonne_desiderate]
else:
    print(f"Attenzione: il file ha {len(df.columns)} colonne, ma ne sono attese {len(colonne_desiderate)}. Nomi non modificati.")

# Sostituisci i valori vuoti o NaN in 'Race Number' con 'ND'
df['Race Number'] = df['Race Number'].fillna('ND').replace('', 'ND')

# Funzione per pulire i nomi dei piloti
def clean_rider_name(name):
    if pd.isnull(name):
        return ''
    name = name.strip()
    match = re.match(r'^(.+?)(\1)+$', name)
    if match:
        name = match.group(1)
    half = len(name) // 2
    if name[:half] == name[half:]:
        name = name[:half]
    name = re.sub(r'\s+', ' ', name)
    return name.strip()

# Applica la pulizia alla colonna Rider
df['Rider'] = df['Rider'].apply(clean_rider_name)

# Salva il file pulito (sovrascrive il file originale)
import os
# Salva il file pulito (sovrascrive il file originale)
output_dir = 'classifica'
output_file = os.path.join(output_dir, 'motogp_results_cleaned_final.csv')
os.makedirs(output_dir, exist_ok=True)
df.to_csv(output_file, index=False)
import pandas as pd

# Percorso del file CSV
file_path = "classifica/motogp_results_cleaned_final.csv"

# Carica il dataset
df = pd.read_csv(file_path)

# 1. Elimina le righe "Event"
df = df[~df['Position'].astype(str).str.contains('Event', na=False)]

# 2. Rimuovi righe vuote tra le posizioni e i piloti non classificati (ND)
df = df[~df.isnull().all(axis=1)]

# 3. Aggiungi una sola riga vuota tra le gare
df['group'] = df['Grand Prix'] + '_' + df['Year'].astype(str)  # Identifica ogni gara
df['is_new_race'] = df['group'] != df['group'].shift()  # Trova l'inizio di una nuova gara



# Ripristino: una sola riga vuota tra le gare

# Inserisci una riga vuota con solo nome evento e anno prima di ogni classifica

# Unisci i non classificati subito dopo la classifica, senza righe vuote tra loro


# Raggruppa per evento: riga evento/anno, poi classificate, poi ND
rows = []
columns = df.columns.tolist()
grouped = df.groupby(['Event', 'Year'], sort=False)
for (event, year), group in grouped:
    # Riga vuota con solo Event e Year
    empty_row = [None]*len(columns)
    if 'Event' in columns:
        empty_row[columns.index('Event')] = event
    if 'Year' in columns:
        empty_row[columns.index('Year')] = year
    rows.append(empty_row)
    # Classificati
    classified = group[group['Position'].apply(lambda x: str(x).strip().isdigit())]
    for _, row in classified.iterrows():
        rows.append(row.values.tolist())
    # Non classificati (ND o vuoto)
    nd = group[~group['Position'].apply(lambda x: str(x).strip().isdigit())]
    for _, row in nd.iterrows():
        rows.append(row.values.tolist())
df_cleaned = pd.DataFrame(rows, columns=columns)
# Estrai la parte del nome pilota dalla colonna Rider e aggiungi una colonna 'Nome'
import re
def estrai_nome(rider):
    if isinstance(rider, str):
        # Cerca la parte con lettere seguite da numeri e punto finale (es. Marquez93M.)
        match = re.search(r'([A-Za-z]+\d+[A-Z]?\.)', rider)
        if match:
            return match.group(1)
    return ''
if 'Rider' in df_cleaned.columns:
    df_cleaned['Nome'] = df_cleaned['Rider'].apply(estrai_nome)
# Aggiorna i nomi delle colonne: la terza 'Rider', la quarta 'Team'
colonne_finali = df_cleaned.columns.tolist()
if len(colonne_finali) > 1:
    colonne_finali[1] = 'Points'
if len(colonne_finali) > 2:
    colonne_finali[2] = 'Rider'
if len(colonne_finali) > 3:
    colonne_finali[3] = 'Team'
df_cleaned.columns = colonne_finali

# Estrai la parte del nome pilota dalla colonna Rider e aggiungi una colonna 'Nome'
import re
if 'Rider' in df_cleaned.columns:
    def estrai_nome(rider):
        if isinstance(rider, str):
            # Cerca la parte con lettere seguite da numeri e punto finale (es. Marquez93M.)
            match = re.search(r'([A-Za-z]+\d+[A-Z]?\.)', rider)
            if match:
                return match.group(1)
        return ''
    df_cleaned['Nome'] = df_cleaned['Rider'].apply(estrai_nome)

# Salva il dataset aggiornato
output_path = "classifica/motogp_results_cleaned_final.csv"
df_cleaned.to_csv(output_path, index=False)

print(f"Dataset aggiornato salvato in: {output_path}")