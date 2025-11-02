import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Carica il dataset dei risultati
df = pd.read_csv('/Users/aurelio/Desktop/TESI/MotoGP-API/classifica/motogp_results_cleaned_final.csv')



# Nessun filtro: usa tutti gli anni disponibili

# Estrai i circuiti unici (nome GP)
circuiti = df['Grand Prix'].drop_duplicates().sort_values().tolist()



# Tabella presenza circuito per anno (tutti gli anni)
anni = sorted(df['Year'].dropna().astype(int).unique())
circuiti_clean = [c for c in circuiti if pd.notna(c)]
tabella = pd.DataFrame(index=circuiti_clean, columns=anni)
for c in circuiti_clean:
    for a in anni:
        presente = not df[(df['Grand Prix']==c) & (df['Year']==a)].empty
        tabella.loc[c, a] = '✓' if presente else ''


# Heatmap presenza circuiti per anno

# Heatmap con quadrati separati e colori più chiari
plt.figure(figsize=(len(anni)*0.6, len(circuiti_clean)*0.5))
data = (tabella == '✓').astype(int).values
im = plt.imshow(data, aspect='auto', cmap='BuGn', interpolation='nearest')
plt.xticks(np.arange(len(anni)), [str(a) for a in anni], rotation=90, fontsize=10)
plt.yticks(np.arange(len(circuiti_clean)), circuiti_clean, fontsize=11)
plt.xlabel('Anno', fontsize=14, fontweight='bold')
plt.ylabel('Circuito', fontsize=14, fontweight='bold')
plt.title('Presenza dei circuiti in MotoGP', fontsize=16, fontweight='bold', pad=16)
plt.colorbar(im, label='Presenza (1=presente, 0=assente)')
# Griglia bianca per separare i quadrati
plt.grid(which='both', color='white', linewidth=1.5)
plt.gca().set_xticks(np.arange(-.5, len(anni), 1), minor=True)
plt.gca().set_yticks(np.arange(-.5, len(circuiti_clean), 1), minor=True)
plt.gca().tick_params(which='minor', bottom=False, left=False)
plt.tight_layout()
plt.savefig('heatmap_presenza_circuiti_tutti_gli_anni.png', dpi=200)
plt.close()
print('Heatmap salvata come heatmap_presenza_circuiti_tutti_gli_anni.png')
