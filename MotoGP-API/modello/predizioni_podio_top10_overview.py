import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    root = os.path.join('modello', 'predizioni_podio_top10_all_gp')
    agg_csv = os.path.join(root, 'all_events_top10_pred_2020_2030.csv')
    if not os.path.exists(agg_csv):
        raise FileNotFoundError(f"File non trovato: {agg_csv}. Esegui prima predizione_podio_top10_all_gp.py")
    df = pd.read_csv(agg_csv)
    out_dir = os.path.join(root, '_overview')
    os.makedirs(out_dir, exist_ok=True)

    # 1) Heatmap Event x Year: media delle probabilità (Top10 mediati)
    pivot_event_year = df.groupby(['Event', 'Year'], as_index=False)['Prob_podio'].mean()
    pv = pivot_event_year.pivot(index='Event', columns='Year', values='Prob_podio').fillna(0)
    plt.figure(figsize=(14, max(6, 0.3 * len(pv))))
    sns.heatmap(pv, annot=True, fmt='.0%', cmap='YlOrRd', vmin=0, vmax=1, cbar_kws={'label': 'Probabilità media podio (Top10)'})
    plt.title('Probabilità media podio (Top10) – Eventi × Anni (2020–2030)', fontsize=16, weight='bold')
    plt.xlabel('Anno')
    plt.ylabel('Evento')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'overview_event_year_heatmap.png'), dpi=200)
    plt.close()

    # 2) Boxplot per evento: distribuzione probabilità (Top10, 2020–2030)
    plt.figure(figsize=(14, 6))
    order = df.groupby('Event')['Prob_podio'].mean().sort_values(ascending=False).index
    sns.boxplot(data=df, x='Event', y='Prob_podio', order=order, whis=1.5)
    sns.stripplot(data=df, x='Event', y='Prob_podio', order=order, color='black', size=2, alpha=0.25)
    plt.ylim(0, 1)
    plt.title('Distribuzione probabilità podio per evento (Top10, 2020–2030)', fontsize=16, weight='bold')
    plt.xlabel('Evento')
    plt.ylabel('Probabilità podio')
    plt.xticks(rotation=30, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'overview_event_boxplot.png'), dpi=200)
    plt.close()

    # 3) Media per pilota su tutti i GP e anni
    mean_by_rider = df.groupby('Rider', as_index=False)['Prob_podio'].mean().sort_values('Prob_podio', ascending=False)
    plt.figure(figsize=(12, 6))
    bars = plt.bar(mean_by_rider['Rider'], mean_by_rider['Prob_podio'], color=sns.color_palette('Set2', n_colors=len(mean_by_rider)), edgecolor='black', alpha=0.9)
    for b, v in zip(bars, mean_by_rider['Prob_podio']):
        plt.text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.0%}', ha='center', va='bottom', fontsize=10, weight='bold')
    plt.ylim(0, 1)
    plt.title('Probabilità media podio per pilota (Top10, 2020–2030, su tutti i GP)', fontsize=16, weight='bold')
    plt.xlabel('Pilota')
    plt.ylabel('Probabilità media podio')
    plt.xticks(rotation=30, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'overview_rider_mean.png'), dpi=200)
    plt.close()

    print(f"Grafici overview creati in: {out_dir}")


if __name__ == '__main__':
    main()
