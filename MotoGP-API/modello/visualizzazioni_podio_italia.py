import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


OUT_DIR = 'modello/output_podio_italia_expanded/visualizzazioni_alternative'
PIVOT_CSV = 'modello/output_podio_italia_expanded/prob_podio_italia_by_year_2030.csv'
DATASET = 'classifica/motogp_results_expanded_features.csv'


def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def load_pivot_and_meta(top_n=8):
    if not os.path.exists(PIVOT_CSV):
        raise FileNotFoundError(f"Pivot CSV non trovato: {PIVOT_CSV}. Esegui prima predizione_podio_italia_expanded.py")
    pivot = pd.read_csv(PIVOT_CSV).set_index('Year')
    pivot.index = pivot.index.astype(int)
    # anno massimo osservato dal dataset per il GP d'Italia
    if os.path.exists(DATASET):
        raw = pd.read_csv(DATASET)
        raw['Event'] = raw['Event'].astype(str).str.lower().str.strip()
        ita = raw[raw['Event'] == 'ita']
        ita['Year'] = pd.to_numeric(ita['Year'], errors='coerce')
        obs_max_year = int(ita['Year'].dropna().max()) if not ita.empty else int(pivot.index.min())
    else:
        obs_max_year = int(pivot.index.min())

    # selezione top piloti per media nel periodo osservato
    observed_slice = pivot.loc[pivot.index <= obs_max_year]
    means = observed_slice.mean(axis=0).sort_values(ascending=False)
    riders = means.head(top_n).index.tolist()
    pivot = pivot[riders]
    return pivot, riders, obs_max_year


def plot_lines_with_projection(pivot, riders, obs_max_year):
    year_min, year_max = int(pivot.index.min()), int(pivot.index.max())
    colors = plt.cm.tab10.colors
    plt.figure(figsize=(12, 7))
    for i, r in enumerate(riders):
        s = pivot[r]
        color = colors[i % len(colors)]
        # storico
        years_hist = [y for y in pivot.index if y <= obs_max_year]
        if years_hist:
            plt.plot(years_hist, s.loc[years_hist], color=color, linewidth=2, label=r)
        # proiezione
        years_proj = [y for y in pivot.index if y > obs_max_year]
        if years_proj:
            plt.plot(years_proj, s.loc[years_proj], color=color, linestyle='--', linewidth=2)
            # etichetta finale
            plt.text(year_max + 0.1, float(s.loc[year_max]), r, color=color, va='center')

    # shading periodo di proiezione
    if obs_max_year < year_max:
        plt.axvspan(obs_max_year + 0.5, year_max + 0.5, color='grey', alpha=0.15, label='Proiezione')
    plt.xlim(year_min, year_max + 1)
    plt.ylim(0, 1)
    plt.xlabel('Anno')
    plt.ylabel('Probabilità di podio')
    plt.title("Probabilità podio GP d'Italia – linee (storico e proiezione)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'linee_proiezione.png')
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_grouped_bars_by_year(pivot, riders):
    colors = plt.cm.tab10.colors
    years = pivot.index.tolist()
    n_years, n_riders = len(years), len(riders)
    width = 0.8 / max(n_riders, 1)
    x = np.arange(n_years)
    plt.figure(figsize=(min(20, 1 + n_years * 0.6), 8))
    for i, r in enumerate(riders):
        vals = pivot[r].values
        plt.bar(x + i * width - (n_riders - 1) * width / 2, vals, width=width,
                color=colors[i % len(colors)], label=r)
    plt.xticks(x, years, rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.xlabel('Anno')
    plt.ylabel('Probabilità di podio')
    plt.title("Probabilità podio GP d'Italia – barre raggruppate per anno")
    plt.legend(ncol=min(4, len(riders)))
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'barre_raggruppate_per_anno.png')
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_heatmap(pivot, riders):
    plt.figure(figsize=(max(8, len(riders) * 1.2), 8))
    sns.heatmap(pivot.T, cmap='viridis', vmin=0.0, vmax=1.0, cbar_kws={'label': 'Probabilità'})
    plt.xlabel('Anno')
    plt.ylabel('Pilota')
    plt.title("Probabilità podio GP d'Italia – heatmap")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'heatmap.png')
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_small_multiples(pivot, riders, cols=3):
    rows = int(np.ceil(len(riders) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.5), sharex=True, sharey=True)
    axes = np.array(axes).reshape(rows, cols)
    year_min, year_max = int(pivot.index.min()), int(pivot.index.max())
    colors = plt.cm.tab10.colors
    for idx, r in enumerate(riders):
        ax = axes[idx // cols][idx % cols]
        ax.plot(pivot.index, pivot[r], color=colors[idx % len(colors)], linewidth=2)
        ax.set_title(r)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(year_min, year_max)
        ax.set_ylim(0, 1)
    # nascondi subplot vuoti
    for j in range(len(riders), rows * cols):
        axes[j // cols][j % cols].axis('off')
    fig.suptitle("Probabilità podio GP d'Italia – small multiples", y=0.995)
    fig.supxlabel('Anno')
    fig.supylabel('Probabilità')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'small_multiples.png')
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_bump_chart(pivot, riders):
    # ranking: 1 = probabilità più alta
    ranks = pivot[riders].rank(axis=1, ascending=False, method='first')
    year_min, year_max = int(pivot.index.min()), int(pivot.index.max())
    colors = plt.cm.tab10.colors
    plt.figure(figsize=(12, 7))
    for i, r in enumerate(riders):
        plt.plot(pivot.index, ranks[r], color=colors[i % len(colors)], linewidth=2, label=r)
        # etichette agli estremi
        plt.text(year_min - 0.2, ranks.loc[year_min, r], r, ha='right', va='center', color=colors[i % len(colors)])
        plt.text(year_max + 0.2, ranks.loc[year_max, r], r, ha='left', va='center', color=colors[i % len(colors)])
    plt.gca().invert_yaxis()
    plt.yticks(range(1, len(riders) + 1))
    plt.xlabel('Anno')
    plt.ylabel('Posizione (ranking)')
    plt.title("Ranking probabilità podio – bump chart")
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'bump_chart.png')
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def main(top_n=8):
    ensure_out_dir()
    pivot, riders, obs_max_year = load_pivot_and_meta(top_n=top_n)
    paths = {}
    paths['linee'] = plot_lines_with_projection(pivot, riders, obs_max_year)
    paths['barre'] = plot_grouped_bars_by_year(pivot, riders)
    paths['heatmap'] = plot_heatmap(pivot, riders)
    paths['small_multiples'] = plot_small_multiples(pivot, riders)
    paths['bump_chart'] = plot_bump_chart(pivot, riders)
    print("Visualizzazioni salvate:")
    for k, v in paths.items():
        print(f"- {k}: {v}")


if __name__ == '__main__':
    main(top_n=8)
