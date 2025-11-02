import pandas as pd
import matplotlib.pyplot as plt
import os

# Carica il file CSV normalizzato
file_path = 'classifica/motogp_results_cleaned_final_normalized.csv'
if not os.path.exists(file_path):
    print(f"ERRORE: File non trovato: {file_path}\nEsegui prima lo script 'normalize_rider_names.py' dalla root del progetto MotoGP-API.")
    exit(1)
df = pd.read_csv(file_path)

# Filtra solo i primi posti
primi = df[df['Position'] == 1]

# Conta i primi posti per pilota normalizzato e prendi i primi 10
primi_per_pilota = primi['Rider_normalized'].value_counts().sort_values(ascending=False).head(10)

# Crea il grafico
plt.figure(figsize=(12,6))
ax = primi_per_pilota.plot(kind='bar', color='gold')
plt.title('Top 10 piloti per numero di primi posti')
plt.xlabel('Pilota')
plt.ylabel('Numero di vittorie')
plt.xticks(rotation=45, ha='right')
for i, v in enumerate(primi_per_pilota.values):
    ax.text(i, v + 0.1, str(v), ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig('grafici/primi_pilota.png')
plt.show()
