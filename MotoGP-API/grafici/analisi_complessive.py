import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as path_effects

# Carica il file normalizzato
csv_path = 'classifica/motogp_results_cleaned_final_normalized.csv'
df: pd.DataFrame = pd.read_csv(csv_path)

# Filtra solo le righe con posizione numerica valida (float, esclude ND e NaN)
df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
df = df[df['Position'].notna()]


# Analisi 1: Numero totale di gare e piloti per anno (grafico a barre)
gare_per_anno = df.groupby('Year')['Grand Prix'].nunique()
piloti_per_anno = df.groupby('Year')['Rider_normalized'].nunique()

# Grafico gare/piloti per anno migliorato
fig1, ax1 = plt.subplots(figsize=(14,7))
bar_width = 0.4
anni = gare_per_anno.index.astype(int)
bar1 = ax1.bar(anni-bar_width/2, gare_per_anno.values, width=bar_width, label='Gare', color='#377eb8', edgecolor='black', linewidth=1.2)
bar2 = ax1.bar(anni+bar_width/2, piloti_per_anno.values, width=bar_width, label='Piloti unici', color='#e41a1c', edgecolor='black', linewidth=1.2)
ax1.set_xlabel('Anno', fontsize=15, fontweight='bold')
ax1.set_ylabel('Numero', fontsize=15, fontweight='bold')
ax1.set_title('Numero di gare e piloti unici per anno', fontsize=20, fontweight='bold')
ax1.legend(fontsize=13, loc='upper left')
plt.xticks(anni, rotation=45, fontsize=12, fontweight='bold')
plt.yticks(fontsize=12)
ax1.grid(axis='y', linestyle='--', alpha=0.6)
# Etichette numeriche sopra le barre
for bar in bar1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, int(bar.get_height()), ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')
for bar in bar2:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, int(bar.get_height()), ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')
plt.tight_layout()
plt.savefig('grafici/gare_piloti_per_anno.png', dpi=300)
plt.close(fig1)


# Analisi 2: Podi totali per ogni pilota (grafico a barre, top 15)
podi = df[df['Position'].isin([1.0, 2.0, 3.0])]
podi_count = podi.groupby(['Rider_normalized', 'Position']).size().unstack(fill_value=0)
podi_count.columns = ['Oro', 'Argento', 'Bronzo'][:len(podi_count.columns)]
podi_count = podi_count.sort_values('Oro', ascending=False).head(15)

# Grafico podi top 15 più leggibile e dettagliato
fig2, ax2 = plt.subplots(figsize=(15,8))
bar_colors = ['#FFD700', '#A8A8A8', '#CD7F32']  # oro, argento, bronzo
podi_count[['Oro', 'Argento', 'Bronzo']].plot(
    kind='bar', stacked=True, ax=ax2, color=bar_colors, edgecolor='black', linewidth=1.2)
ax2.set_title('Podi totali (oro, argento, bronzo) - Top 15 piloti', fontsize=20, fontweight='bold')
ax2.set_xlabel('Pilota', fontsize=15, fontweight='bold')
# Asse y automatico e griglia tratteggiata (versione precedente)
ax2.set_ylabel('Numero podi', fontsize=15, fontweight='bold')
plt.xticks(rotation=30, ha='right', fontsize=12, fontweight='bold')
plt.yticks(fontsize=12)
ax2.legend(title='Tipo podio', fontsize=13, title_fontsize=14, loc='upper right')
ax2.grid(axis='y', linestyle='--', alpha=0.5)
# Etichette numeriche su ogni barra
for c_idx, col in enumerate(['Oro', 'Argento', 'Bronzo']):
    for idx, val in enumerate(podi_count[col]):
        if val > 0:
            # Somma cumulata delle colonne precedenti per la stessa riga
            y_base = sum([podi_count.iloc[idx][c] for c in ['Oro', 'Argento', 'Bronzo'][:c_idx]])
            y = y_base + val/2
            ax2.text(idx, y, int(val), ha='center', va='center', fontsize=11, fontweight='bold', color='black')
plt.tight_layout()
plt.savefig('grafici/podi_top15.png', dpi=300)
plt.close(fig2)




# Analisi 3: Calcolo classifica piloti per punti totali per anno (necessario per analisi successiva, nessun grafico prodotto)
df['Points'] = pd.to_numeric(df['Points'], errors='coerce').fillna(0)
classifica = df.groupby(['Year', 'Rider_normalized'])['Points'].sum().reset_index()

# Analisi 4: Andamento punti dei migliori piloti nel tempo
top_piloti = classifica.groupby('Rider_normalized')['Points'].sum().sort_values(ascending=False).head(5).index
trend = classifica[classifica['Rider_normalized'].isin(top_piloti)]



# Figura 1: Andamento punti dei migliori 5 piloti nel tempo (colori contrasto, etichette nere)
plt.figure(figsize=(14,8))
# Palette ad alto contrasto (colori ben distinguibili)
contrast_palette = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#984ea3']
ax = sns.lineplot(data=trend, x='Year', y='Points', hue='Rider_normalized', marker='o', linewidth=2.5, markersize=9, palette=contrast_palette)
plt.title('Andamento punti dei migliori 5 piloti nel tempo', fontsize=18)
plt.ylabel('Punti totali stagionali', fontsize=14)
plt.xlabel('Anno', fontsize=14)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.xticks(sorted(trend['Year'].unique()), rotation=45)
plt.legend(title='Pilota', fontsize=12, title_fontsize=13, loc='upper left')


# Etichette punti su ogni marker (in nero, grassetto, con bordo bianco per leggibilità)
for pilota in top_piloti:
    dati = trend[trend['Rider_normalized'] == pilota]
    for _, row in dati.iterrows():
        ax.text(
            row['Year'], row['Points']+2, int(row['Points']),
            color='black', fontsize=11, ha='center', va='bottom', fontweight='bold',
            path_effects=[
                path_effects.withStroke(linewidth=2.5, foreground='white')
            ]
        )

plt.tight_layout()
plt.savefig('grafici/andamento_punti_top5.png', dpi=300)
plt.show()
