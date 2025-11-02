import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# Carica i dati storici
df_storico = pd.read_csv('modello/output_podio_italia/probabilita_podio_gp_italia_2010_2020.csv')
df_storico['Tipo'] = 'Storico'
# Carica le previsioni future
df_future = pd.read_csv('modello/output_podio_italia/previsione_probabilita_podio_gp_italia_2020_2030.csv')
df_future['Tipo'] = 'Previsione'
# Unisci i dati
df = pd.concat([df_storico, df_future], ignore_index=True)
# Ordina per anno e poi per pilota
df = df.sort_values(['Year', 'Rider'])
df['label'] = df['Rider'].astype(str) + ' (' + df['Year'].astype(str) + ')'
piloti_unici = df['Rider'].astype(str).unique()
palette = sns.color_palette('Set1', n_colors=len(piloti_unici))
pilota2col = {pilota: palette[i % len(palette)] for i, pilota in enumerate(piloti_unici)}
bar_colors = df['Rider'].astype(str).map(pilota2col)
plt.figure(figsize=(24, 10))
bars = plt.bar(df['label'], df['Prob_podio'], color=bar_colors, edgecolor='black', alpha=0.85, width=0.7)
plt.ylabel('Probabilità podio', fontsize=16)
plt.xlabel('Pilota (Anno)', fontsize=16)
plt.title('Probabilità podio GP Italia: storico 2010-2020 e previsioni 2020-2030', fontsize=20, weight='bold')
plt.xticks(rotation=60, ha='right', fontsize=13)
plt.yticks(fontsize=14)
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.3)
for bar, val, tipo in zip(bars, df['Prob_podio'], df['Tipo']):
    if val > 0:
        if tipo == 'Previsione' and val < 0.08:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{val:.0%}", ha='center', va='bottom', fontsize=14, fontweight='bold', color='black')
        else:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.07, f"{val:.0%}", ha='center', va='top', fontsize=14, fontweight='bold', color='black')
legend_handles = [Patch(color=pilota2col[pilota], label=pilota) for pilota in piloti_unici]
plt.legend(handles=legend_handles, title='Pilota', loc='upper right', fontsize=13)
plt.tight_layout()
plt.savefig('modello/output_podio_italia/grafico_probabilita_podio_gp_italia_storico_previsione.png', dpi=300)
plt.show()
