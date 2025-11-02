
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as mticker

# Carica il file CSV normalizzato con controllo di esistenza
file_path = 'classifica/motogp_results_cleaned_final_normalized.csv'
if not os.path.exists(file_path):
	print(f"ERRORE: File non trovato: {file_path}\nEsegui prima lo script 'normalize_rider_names.py' dalla root del progetto MotoGP-API.")
	exit(1)
df = pd.read_csv(file_path)

# Filtra solo le posizioni di podio (1, 2, 3)
podio = df[df['Position'].isin([1, 2, 3])]


# Conta podi oro/argento/bronzo per i primi 10 piloti
top10 = podio['Rider_normalized'].value_counts().sort_values(ascending=False).head(10).index.tolist()
podio_top10 = podio[podio['Rider_normalized'].isin(top10)]
podio_pivot = podio_top10.pivot_table(index='Rider_normalized', columns='Position', values='Grand Prix', aggfunc='count', fill_value=0)
# Ordina i piloti per totale podi
podio_pivot = podio_pivot.loc[top10]

# Rinomina colonne per chiarezza
col_map = {1.0: 'Oro', 2.0: 'Argento', 3.0: 'Bronzo'}
podio_pivot = podio_pivot.rename(columns=col_map)
for col in ['Oro', 'Argento', 'Bronzo']:
	if col not in podio_pivot.columns:
		podio_pivot[col] = 0
podio_pivot = podio_pivot[['Oro', 'Argento', 'Bronzo']]

plt.figure(figsize=(13,7))
ax = podio_pivot.plot(
	kind='bar',
	stacked=True,
	color=['#FFD700', '#C0C0C0', '#CD7F32'],
	zorder=2,
	edgecolor='black',
	ax=plt.gca()
)
plt.title('Top 10 piloti per numero di podi (Oro/Argento/Bronzo)', fontsize=16, fontweight='bold')
plt.xlabel('Pilota', fontsize=13)
plt.ylabel('Numero di podi', fontsize=13)
plt.xticks(rotation=35, ha='right', fontsize=12)
plt.yticks(fontsize=12)
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.set_ylim(0, podio_pivot.sum(axis=1).max() * 1.15)
ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=1)

# Etichette su ogni segmento
for idx, pilota in enumerate(podio_pivot.index):
	bottom = 0
	for col, color in zip(['Oro', 'Argento', 'Bronzo'], ['#FFD700', '#C0C0C0', '#CD7F32']):
		val = podio_pivot.loc[pilota, col]
		if val > 0:
			ax.text(
				idx, bottom + val/2, str(int(val)),
				ha='center', va='center', fontsize=11, fontweight='bold', color='black',
				bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.18', alpha=0.18)
			)
		bottom += val

plt.legend(title='Posizione', fontsize=12, title_fontsize=13, loc='upper right')
plt.tight_layout()
plt.savefig('grafici/podi_per_pilota_stacked.png')
plt.show()
