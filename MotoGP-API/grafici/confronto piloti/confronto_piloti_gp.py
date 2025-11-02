import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Parametri: inserisci qui i piloti da confrontare
RIDERS = ['Rossi', 'Marquez', 'Bagnaia']  # nomi normalizzati
YEARS = list(range(2020, 2025))  # dal 2020 al 2024

# Carica il file normalizzato
df = pd.read_csv('classifica/motogp_results_cleaned_final_normalized.csv')

df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
df = df[df['Position'].notna()]
df['Year'] = df['Year'].astype(int)




# --- HEATMAP PILOTA-STAGIONE (VITTORIE) ---
import seaborn as sns
# Heatmap per gruppi di anni uguali su tutto il dataset
import math
anni_disponibili = sorted(df['Year'].dropna().unique())
anni_disponibili = [int(a) for a in anni_disponibili]
gruppo_dim = 5  # puoi cambiare la dimensione del gruppo qui
num_gruppi = math.ceil(len(anni_disponibili)/gruppo_dim)

for g in range(num_gruppi):
	anni_gruppo = anni_disponibili[g*gruppo_dim:(g+1)*gruppo_dim]
	if not anni_gruppo:
		continue
	df_vittorie = df[(df['Year'].isin(anni_gruppo)) & (df['Position'] == 1)]
	conteggio = df_vittorie.groupby(['Year', 'Rider_normalized']).size().reset_index(name='Vittorie')
	totali = conteggio.groupby('Rider_normalized')['Vittorie'].sum()
	piloti_top = totali[totali >= 2].sort_values(ascending=False).index.tolist()
	if not piloti_top:
		continue
	tabella = conteggio[conteggio['Rider_normalized'].isin(piloti_top)].pivot(index='Rider_normalized', columns='Year', values='Vittorie').fillna(0).reindex(piloti_top)
	plt.figure(figsize=(max(12, len(piloti_top)*0.6), 0.5*len(piloti_top)+4))
	sns.heatmap(tabella, annot=True, fmt='.0f', cmap='YlOrRd', linewidths=0.5, cbar_kws={'label': 'Vittorie'})
	plt.title(f'Heatmap per pilota con più di 2 vittorie\nAnni: {anni_gruppo[0]} - {anni_gruppo[-1]}', fontsize=18, fontweight='bold')
	plt.xlabel('Stagione', fontsize=14)
	plt.ylabel('Pilota', fontsize=14)
	plt.tight_layout()
	filename = f'grafici/heatmap/vittorie_pilota_stagione_heatmap_{anni_gruppo[0]}_{anni_gruppo[-1]}.png'
	plt.savefig(filename, dpi=300)
	plt.show()
	print(f"Heatmap salvata: {filename}")
