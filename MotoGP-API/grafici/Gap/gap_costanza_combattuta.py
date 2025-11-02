
# IMPORT
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random




# Carica il file CSV normalizzato
file_path = 'classifica/motogp_results_cleaned_final_normalized.csv'
if not os.path.exists(file_path):
    print(f"ERRORE: File non trovato: {file_path}")
    exit(1)
df = pd.read_csv(file_path)

# Parametri: usa tutti gli anni disponibili nel dataset
anni = sorted(df['Year'].dropna().unique())
output_dir = 'grafici/Gap/'
os.makedirs(output_dir, exist_ok=True)

# Carica il file CSV normalizzato
file_path = 'classifica/motogp_results_cleaned_final_normalized.csv'
if not os.path.exists(file_path):
    print(f"ERRORE: File non trovato: {file_path}")
    exit(1)
df = pd.read_csv(file_path)

# Funzione per convertire il gap in secondi
def parse_gap(val):
    if isinstance(val, str):
        if 'Laps' in val or 'ND' in val or val.strip() == '' or val.startswith('Event'):
            return None
        if val.startswith('+'):
            try:
                return float(val.replace('+','').replace(',','.'))
            except:
                return None
        if ':' in val:
            try:
                m, s = val.split(':')
                return int(m)*60 + float(s.replace(',','.'))
            except:
                return None
    return None

# Prepara il dataframe
df = df[df['Time/Gap'].notnull() & df['Position'].notnull()].copy()
df['Gap_sec'] = df['Time/Gap'].apply(parse_gap)
df = df[df['Gap_sec'].notnull()]



# === COSTANZA PILOTI MULTI-STAGIONE (tutto il dataset, TOP 10) ===
costanza = []
for anno in anni:
    df_anno = df[df['Year'] == anno].copy()
    piloti = df_anno['Rider_normalized'].value_counts().loc[lambda x: x>=8].index.tolist()
    for pilota in piloti:
        dev = df_anno[df_anno['Rider_normalized']==pilota]['Gap_sec'].std()
        costanza.append({'Pilota': pilota, 'Anno': int(anno), 'Deviazione_std': dev})
df_costanza = pd.DataFrame(costanza)

# Seleziona i 10 piloti con più stagioni valide nel periodo
piloti_top = df_costanza['Pilota'].value_counts().sort_values(ascending=False).head(10).index.tolist()
df_costanza = df_costanza[df_costanza['Pilota'].isin(piloti_top)]


# Ordinamento piloti per costanza media (dal più costante)
media_costanza = df_costanza.groupby('Pilota')['Deviazione_std'].mean().sort_values()
piloti_top_sorted = [p for p in media_costanza.index if p in piloti_top]



# --- Migliora visibilità punti: marker grandi, bordo nero, trasparenza ---
plt.figure(figsize=(15,8))
colori = plt.cm.get_cmap('tab10', len(piloti_top_sorted))
for idx, pilota in enumerate(piloti_top_sorted):
    dati = df_costanza[df_costanza['Pilota']==pilota].sort_values('Anno')
    # Jitter verticale per punti con gap < 0.6 (molto ravvicinati)
    y_jitter = []
    for y in dati['Deviazione_std']:
        if y < 0.6:
            y_jitter.append(y + random.uniform(-0.07, 0.07))
        else:
            y_jitter.append(y)
    plt.plot(dati['Anno'], y_jitter, label=pilota, linewidth=2.5, color=colori(idx), alpha=0.85)
    plt.scatter(dati['Anno'], y_jitter, s=160, color=colori(idx), edgecolor='black', linewidth=1.2, alpha=0.95, zorder=3)
    # Etichette spostate: sopra se y>0.7, sotto se y<=0.7
    for x, y, yj in zip(dati['Anno'], dati['Deviazione_std'], y_jitter):
        if y > 0.7:
            plt.text(x, yj+0.13, f"{y:.1f}", ha='center', va='bottom', fontsize=10, color=colori(idx))
        else:
            plt.text(x, yj-0.13, f"{y:.1f}", ha='center', va='top', fontsize=10, color=colori(idx))
plt.xlabel('Stagione', fontsize=16, fontweight='bold')
plt.ylabel('Deviazione standard gap dal vincitore (s)', fontsize=16, fontweight='bold')
plt.title('Costanza dei top 10 piloti MotoGP (2010-2020)\nDeviazione standard del gap dal vincitore per stagione', fontsize=19, fontweight='bold', pad=18)
plt.suptitle('Solo piloti con almeno 3 stagioni complete nel periodo', fontsize=13, y=0.92, color='dimgray')
plt.legend(title='Pilota (ordinati per costanza media)', fontsize=12, title_fontsize=13, bbox_to_anchor=(1.01, 1), loc='upper left', frameon=True)
plt.grid(axis='y', linestyle=':', alpha=0.7, linewidth=1)
plt.ylim(bottom=0, top=max(df_costanza['Deviazione_std'].max()+0.5, 2))
plt.tight_layout(rect=[0,0,0.85,1])
plt.savefig(os.path.join(output_dir, 'gap_costanza_piloti_2010_2020.png'), dpi=250)
plt.close()


bars = []

# === GARE PIÙ COMBATTUTE MULTI-STAGIONE (tutto il dataset, TOP 3 PER ANNO) ===
combattute = []
for anno in anni:
    df_anno = df[df['Year'] == anno].copy()
    df_anno['GP_short'] = df_anno['Grand Prix'].str.replace('Grand Prix', '').str.strip()
    gp_order = df_anno['GP_short'].drop_duplicates().tolist()
    for gp in gp_order:
        dati_gp = df_anno[df_anno['GP_short']==gp]
        if dati_gp['Position'].isin([1,3]).sum() == 2:
            gap1 = dati_gp[dati_gp['Position']==1]['Gap_sec'].values[0]
            gap3 = dati_gp[dati_gp['Position']==3]['Gap_sec'].values[0]
            combattute.append({'Anno': int(anno), 'GP': gp, 'Gap13': abs(gap3-gap1)})
df_combattute = pd.DataFrame(combattute)

# Prendi le 3 gare più combattute per stagione
top_comb = df_combattute.groupby('Anno').apply(lambda x: x.nsmallest(3, 'Gap13')).reset_index(drop=True)

# Migliora visualizzazione colonne: mostra solo le gare effettivamente presenti in almeno un anno, ordina per frequenza, limita a max 12 eventi per leggibilità
MAX_EVENTI = 12
# Ordina gli eventi per gap medio 1°-3° crescente (gare più combattute per prime)
event_stats = top_comb.groupby('GP')['Gap13'].mean().sort_values()
eventi = event_stats.head(MAX_EVENTI).index.tolist()
anni_sorted = sorted(anni)
bar_width = 0.13
bar_positions = {ev: i for i, ev in enumerate(eventi)}
anno_palette = plt.cm.get_cmap('tab20', len(anni_sorted))

# Costruisci una matrice (evento x anno) con i gap, NaN se mancante
gap_matrix = np.full((len(eventi), len(anni_sorted)), np.nan)
for i_ev, ev in enumerate(eventi):
    for i_an, an in enumerate(anni_sorted):
        match = top_comb[(top_comb['GP']==ev) & (top_comb['Anno']==an)]
        if not match.empty:
            gap_matrix[i_ev, i_an] = match.iloc[0]['Gap13']

plt.figure(figsize=(max(16, len(eventi)*0.9), 9))
bars = []
for i_an, an in enumerate(anni_sorted):
    pos = np.arange(len(eventi)) + (i_an-len(anni_sorted)/2)*bar_width + bar_width/2
    bar = plt.bar(pos, gap_matrix[:,i_an], width=bar_width, color=anno_palette(i_an), alpha=0.92, edgecolor='black', linewidth=0.7, label=str(an))
    bars.append(bar)
    # Etichette disattivate per evitare affollamento; il valore è leggibile tramite griglia e legenda

plt.ylabel('Gap tra 1° e 3° classificato (s)', fontsize=16, fontweight='bold')
plt.xlabel('Gran Premio', fontsize=16, fontweight='bold')
plt.title('Top 3 gare più combattute MotoGP per evento (tutto il dataset)\nGap tra 1° e 3° classificato, colonne affiancate per anno', fontsize=18, fontweight='bold', pad=14)
# Nota come didascalia in basso per evitare sovrapposizioni con il titolo
plt.gcf().text(0.5, 0.01, 'Colori diversi per stagione. Valori più bassi = gare più combattute', ha='center', va='bottom', fontsize=11, color='dimgray')
plt.xticks(np.arange(len(eventi)), eventi, rotation=45, ha='right', fontsize=13)
plt.ylim(bottom=0, top=max(np.nanmax(gap_matrix)+0.5, 2))
plt.legend(title='Stagione', fontsize=12, title_fontsize=13, bbox_to_anchor=(1.01, 1), loc='upper left', frameon=True, ncol=2)
plt.grid(axis='y', linestyle=':', alpha=0.7, linewidth=1)
plt.tight_layout(rect=[0,0.03,0.85,1])
plt.savefig(os.path.join(output_dir, 'gap_gare_combattute_tutto_dataset.png'), dpi=250)
plt.close()

# --- Visualizzazione alternativa 1: Heatmap eventi (righe) x anni (colonne) ---
try:
    import matplotlib
    plt.figure(figsize=(max(12, len(anni_sorted)*0.5), max(8, len(eventi)*0.5)))
    # Usa masked array per NaN
    data = np.ma.masked_invalid(gap_matrix)
    cmap = plt.cm.RdYlGn_r
    im = plt.imshow(data.T, aspect='auto', interpolation='nearest', cmap=cmap)
    cbar = plt.colorbar(im)
    cbar.set_label('Gap 1°-3° (ms)')
    plt.xticks(ticks=np.arange(len(eventi)), labels=eventi, rotation=45, ha='right', fontsize=11)
    plt.yticks(ticks=np.arange(len(anni_sorted)), labels=[str(a) for a in anni_sorted], fontsize=11)
    plt.title('Gare più combattute (TOP3 per anno) - Heatmap', fontsize=16, fontweight='bold')
    plt.xlabel('Gran Premio')
    plt.ylabel('Stagione')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gap_gare_combattute_heatmap.png'), dpi=250)
    plt.close()
except Exception as e:
    print(f"Avviso: impossibile creare la heatmap ({e})")

# --- Visualizzazione alternativa 2: Barh dei GP più ricorrenti (media gap) ---
try:
    agg = top_comb.groupby('GP')['Gap13'].mean().sort_values().head(15)
    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(agg))
    bars = plt.barh(y_pos, agg.values, color='teal', alpha=0.9, edgecolor='black')
    plt.gca().invert_yaxis()
    plt.yticks(y_pos, agg.index, fontsize=12)
    plt.xlabel('Gap medio 1°-3° (ms)')
    plt.title('Gare più combattute (TOP3 per anno)\nMedia gap 1°-3° per evento', fontsize=16, fontweight='bold')
    for rect, val in zip(bars, agg.values):
        plt.text(val + max(agg.values)*0.01, rect.get_y()+rect.get_height()/2, f"{val:.2f}", va='center', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gap_gare_combattute_top_eventi_media.png'), dpi=250)
    plt.close()
except Exception as e:
    print(f"Avviso: impossibile creare il barh aggregato ({e})")

print('Grafici Gap comparativi generati in grafici/Gap/')
