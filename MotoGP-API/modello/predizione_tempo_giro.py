# Previsione tempi sul giro (regressione)
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import numpy as np
import seaborn as sns
import os
from pandas.api.types import is_numeric_dtype


# Carica il dataset
file_path = '/Users/aurelio/Desktop/TESI/MotoGP-API/classifica/motogp_results_cleaned_final_normalized.csv'
df = pd.read_csv(file_path)

# FILTRO 1: solo piloti arrivati al traguardo (Position numerica e Time/Gap valido)
df = df[pd.to_numeric(df['Position'], errors='coerce').notnull()]
df['Position'] = df['Position'].astype(int)

# FILTRO 2: escludi gare bagnate (se disponibile la colonna 'Event' o simile)
if 'Event' in df.columns:
    df = df[~df['Event'].str.lower().str.contains('wet|rain|bagnato', na=False)]


# Feature engineering: unione meteo (Track Conditions, Humidity, Ground Temp)
# Carica meteo e normalizza colonne per il join
weather_path = '/Users/aurelio/Desktop/TESI/MotoGP-API/meteo/motogp_weather_data.csv'
dfw = pd.read_csv(weather_path)
dfw['Event'] = dfw['Event'].astype(str).str.lower().str.strip()
dfw['Year'] = pd.to_numeric(dfw['Year'], errors='coerce').astype('Int64')

# Pulisci colonne meteo: rimuovi simboli e converti a numerico
def to_float_pct(x):
    if pd.isna(x) or str(x).upper() == 'N/A':
        return np.nan
    s = str(x).strip().replace('%','').replace(',','.')
    try:
        return float(s)
    except:
        return np.nan

def to_float_deg(x):
    if pd.isna(x) or str(x).upper() == 'N/A':
        return np.nan
    s = str(x).strip().replace('º','').replace('°','').replace(',','.')
    try:
        return float(s)
    except:
        return np.nan

dfw['Humidity_num'] = dfw.get('Humidity').apply(to_float_pct)
dfw['GroundTemp_num'] = dfw.get('Ground Temp').apply(to_float_deg)
dfw['TrackConditions_cat'] = dfw.get('Track Conditions').astype(str).str.title().replace({'N/A': np.nan})

# Prepara chiavi nella base risultati
df['Event'] = df['Event'].astype(str).str.lower().str.strip()
df['Year_int'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)

# Join sinistro su Year+Event
df = df.merge(
    dfw[['Year','Event','TrackConditions_cat','Humidity_num','GroundTemp_num']],
    left_on=['Year_int','Event'], right_on=['Year','Event'], how='left'
)

# Costruisci matrice delle feature, includendo meteo
# Permetti override via env `MOTOGP_FEATURES` (lista comma-separata)
DEFAULT_FEATURES = [
    'Rider_normalized', 'Team', 'Grand Prix', 'Year_int',
    'TrackConditions_cat', 'Humidity_num', 'GroundTemp_num'
]
env_features = os.environ.get('MOTOGP_FEATURES')
if env_features:
    requested = [c.strip() for c in env_features.split(',') if c.strip()]
    # Tieni solo le colonne effettivamente presenti
    available_cols = set(df.columns)
    features = [c for c in requested if c in available_cols]
    if not features:
        print(f"[WARN] MOTOGP_FEATURES non valide: {requested}. Uso default: {DEFAULT_FEATURES}")
        features = DEFAULT_FEATURES
else:
    features = DEFAULT_FEATURES

print(f"[INFO] Variabili utilizzate (features): {features}")
X_full = df[features].copy()


# Target: tempo/Gap rispetto al vincitore, convertito in secondi
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




df['Gap_sec'] = df['Time/Gap'].apply(parse_gap)
# FILTRO 7: escludi gap troppo piccoli (<0.1s)
# FILTRO 10: gap tra 5° e 95° percentile
gap_valid = df['Gap_sec'].notnull() & (df['Gap_sec'] < 60) & (df['Gap_sec'] >= 0.1)
q_low = df.loc[gap_valid, 'Gap_sec'].quantile(0.05)
q_high = df.loc[gap_valid, 'Gap_sec'].quantile(0.95)
mask = gap_valid & (df['Gap_sec'] >= q_low) & (df['Gap_sec'] <= q_high)
df = df[mask]
X = X_full.loc[mask].copy()
y = df['Gap_sec']

# Encoding variabili categoriche (dinamico in base alle feature selezionate)
for col in X.columns:
    if not is_numeric_dtype(X[col]):
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Gestione NaN: imputazione numeriche con mediana, categoriche già label-encodate
for col in ['Humidity_num','GroundTemp_num']:
    if col in X.columns:
        med = X[col].median()
        X[col] = X[col].fillna(med)
X = X.fillna(-1)
y = y.fillna(y.mean())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Confronto tra più algoritmi di regressione
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

# Selezione modelli configurabile via env `MOTOGP_MODELS` (es: "rf,lr,svr,knn,gbr,dt")
MODEL_FACTORIES = {
    'rf': ('Random Forest', lambda: RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)),
    'lr': ('Linear Regression', lambda: LinearRegression()),
    'svr': ('SVR', lambda: SVR(C=2.0, epsilon=0.2)),
    'knn': ('KNN (k=7)', lambda: KNeighborsRegressor(n_neighbors=7)),
    'gbr': ('Gradient Boosting', lambda: GradientBoostingRegressor(n_estimators=200, learning_rate=0.08, random_state=42)),
    'dt': ('Decision Tree', lambda: DecisionTreeRegressor(max_depth=8, random_state=42)),
}

env_models = os.environ.get('MOTOGP_MODELS')
if env_models:
    keys = [k.strip().lower() for k in env_models.split(',') if k.strip()]
else:
    keys = ['rf', 'lr', 'svr', 'knn', 'gbr']  # default storici

models = {}
for k in keys:
    if k in MODEL_FACTORIES:
        name, factory = MODEL_FACTORIES[k]
        models[name] = factory()
    else:
        print(f"[WARN] Modello '{k}' non riconosciuto. Valori validi: {list(MODEL_FACTORIES.keys())}")

print(f"[INFO] Modelli attivi: {list(models.keys())}")

# Modalità grafici: default solo istogramma errori per modello
PLOT_MODE = os.environ.get('MOTOGP_PLOTS', 'errors-only').lower()

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{name}: MAE = {mae:.2f}, RMSE = {rmse:.2f}")
    results[name] = {'mae': mae, 'rmse': rmse, 'y_pred': y_pred}

labels = list(results.keys())
mae_vals = [results[k]['mae'] for k in labels]
rmse_vals = [results[k]['rmse'] for k in labels]
best_idx = rmse_vals.index(min(rmse_vals))

if PLOT_MODE != 'errors-only':
    # Grafici comparativi + salvataggio metriche
    plt.figure(figsize=(8,5))
    bars1 = plt.bar(labels, mae_vals, color=['gold' if i==best_idx else 'royalblue' for i in range(len(labels))])
    plt.title('MAE - Confronto algoritmi regressione gap')
    plt.ylabel('Errore')
    plt.xticks(range(len(labels)), labels, rotation=20)
    max_mae = max(mae_vals)
    plt.ylim(0, max_mae * 1.10)
    for i, bar in enumerate(bars1):
        y = bar.get_height() + max_mae * 0.02
        plt.text(bar.get_x() + bar.get_width()/2, y, f"{mae_vals[i]:.2f}", ha='center', va='bottom', fontsize=11,
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))
    plt.tight_layout()
    plt.savefig('modello/confronto_algoritmi_regressione_mae.png', dpi=150)
    plt.close()

    plt.figure(figsize=(8,5))
    bars2 = plt.bar(labels, rmse_vals, color=['gold' if i==best_idx else 'orange' for i in range(len(labels))])
    plt.title('RMSE - Confronto algoritmi regressione gap')
    plt.ylabel('Errore')
    plt.xticks(range(len(labels)), labels, rotation=20)
    max_rmse = max(rmse_vals)
    plt.ylim(0, max_rmse * 1.10)
    for i, bar in enumerate(bars2):
        y = bar.get_height() + max_rmse * 0.02
        plt.text(bar.get_x() + bar.get_width()/2, y, f"{rmse_vals[i]:.2f}", ha='center', va='bottom', fontsize=11,
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))
    plt.tight_layout()
    plt.savefig('modello/confronto_algoritmi_regressione_rmse.png', dpi=150)
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(13,5))
    bars1 = axs[0].bar(labels, mae_vals, color=['gold' if i==best_idx else 'royalblue' for i in range(len(labels))])
    axs[0].set_title('MAE')
    axs[0].set_ylabel('Errore')
    for i, bar in enumerate(bars1):
        axs[0].text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.01, f"{mae_vals[i]:.2f}", ha='center', va='bottom', fontsize=10)
    bars2 = axs[1].bar(labels, rmse_vals, color=['gold' if i==best_idx else 'orange' for i in range(len(labels))])
    axs[1].set_title('RMSE')
    for i, bar in enumerate(bars2):
        axs[1].text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.01, f"{rmse_vals[i]:.2f}", ha='center', va='bottom', fontsize=10)
    for ax in axs:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20)
    plt.suptitle('Confronto algoritmi regressione gap')
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig('modello/confronto_algoritmi_regressione.png', dpi=150)
    plt.close()

    df_metrics = pd.DataFrame({'Algoritmo': labels,'MAE': mae_vals,'RMSE': rmse_vals})
    df_metrics.to_csv('modello/metriche_regressione.csv', index=False)



if PLOT_MODE != 'errors-only':
    # Scatter complessivo
    plt.figure(figsize=(9,9))
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#984ea3']
    marker = 'o'
    y_min = min(y_test.min(), *(results[name]['y_pred'].min() for name in labels))
    y_max = max(y_test.max(), *(results[name]['y_pred'].max() for name in labels))
    plt_min = 10.0
    plt_max = 50.0
    for i, name in enumerate(labels):
        y_pred = results[name]['y_pred']
        plt.scatter(y_test, y_pred, alpha=0.25, label=name, color=colors[i%len(colors)], marker=marker, s=36, edgecolor='black', linewidth=0.7)
    plt.plot([plt_min, plt_max], [plt_min, plt_max], 'k--', lw=2, label='Perfetto')
    plt.xlabel('Gap reale (s)')
    plt.ylabel('Gap predetto (s)')
    plt.title('Scatterplot predizione gap in relazione al meteo')
    plt.xlim(plt_min, plt_max)
    plt.ylim(plt_min, plt_max)
    plt.legend()
    plt.tight_layout()
    plt.savefig('modello/scatter_gap_reale_vs_predetto_confronto.png', dpi=150)
    plt.close()

if PLOT_MODE != 'errors-only':
    for name in labels:
        y_pred = results[name]['y_pred']
        plt.figure(figsize=(7,7))
        plt.scatter(y_test, y_pred, alpha=0.3)
        x_min_s, x_max_s = 10.0, 50.0
        plt.plot([x_min_s, x_max_s], [x_min_s, x_max_s], 'r--', lw=2)
        plt.xlim(x_min_s, x_max_s)
        plt.ylim(x_min_s, x_max_s)
        plt.xlabel('Gap reale (s)')
        plt.ylabel('Gap predetto (s)')
        plt.title(f'{name} - Gap reale vs predetto')
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        plt.tight_layout()
        fname = name.lower().replace(' ','_').replace('(','').replace(')','').replace('/','_')
        plt.savefig(f'modello/scatter_gap_reale_vs_predetto_{fname}.png', dpi=150)
        plt.close()

if PLOT_MODE != 'errors-only':
    dt_model = DecisionTreeRegressor(max_depth=8, random_state=42)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    grid_names = labels + ['Decision Tree']
    grid_preds = [results[name]['y_pred'] for name in labels] + [dt_pred]
    y_min_grid = min(y_test.min(), *(pred.min() for pred in grid_preds))
    y_max_grid = max(y_test.max(), *(pred.max() for pred in grid_preds))
    plt_min_g = 10.0
    plt_max_g = 50.0
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
    palette6 = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#984ea3', '#a65628']
    for i, ax in enumerate(axes.ravel()):
        if i >= len(grid_names):
            ax.axis('off')
            continue
        name = grid_names[i]
        y_pred_i = grid_preds[i]
        ax.scatter(y_test, y_pred_i, alpha=0.28, s=24, edgecolor='black', linewidth=0.5, color=palette6[i % len(palette6)])
        ax.plot([plt_min_g, plt_max_g], [plt_min_g, plt_max_g], 'k--', lw=1.2)
        ax.set_title(name, fontsize=11)
        ax.set_xlim(plt_min_g, plt_max_g)
        ax.set_ylim(plt_min_g, plt_max_g)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    for ax in axes[1, :]:
        ax.set_xlabel('Gap reale (s)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Gap predetto (s)')
    fig.suptitle('scatterplot predizione gap in relazione al meteo', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('modello/scatter_gap_6_algoritmi.png', dpi=150)
    plt.close()

best_name = labels[best_idx]
best_pred = results[best_name]['y_pred']
if PLOT_MODE != 'errors-only':
    df_results = pd.DataFrame({'y_true': y_test, 'y_pred': best_pred})
    df_results.to_csv('modello/risultati_regressione_gap.csv', index=False)
    plt.figure(figsize=(7,7))
    plt.scatter(y_test, best_pred, alpha=0.3)
    min_s, max_s = 10.0, 50.0
    plt.plot([min_s, max_s], [min_s, max_s], 'r--', lw=2)
    plt.xlim(min_s, max_s)
    plt.ylim(min_s, max_s)
    plt.xlabel('Gap reale (s)')
    plt.ylabel('Gap predetto (s)')
    plt.title(f'{best_name} - Gap reale vs predetto')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    plt.tight_layout()
    plt.savefig('modello/scatter_gap_reale_vs_predetto.png', dpi=150)
    plt.close()

if PLOT_MODE != 'errors-only':
    # Istogramma avanzato degli errori (miglior modello)
    errors = y_test - best_pred

    # Statistiche chiave
    err_mean = float(np.mean(errors))
    err_median = float(np.median(errors))
    err_std = float(np.std(errors, ddof=1))
    mae = float(mean_absolute_error(y_test, best_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, best_pred)))
    p5, p95 = np.percentile(errors, [5, 95])

    # Bins simmetrici attorno a 0 con regola di Freedman–Diaconis (fallback a 30)
    iqr = np.subtract(*np.percentile(errors, [75, 25]))
    n = max(len(errors), 1)
    bin_width = 2 * iqr * (n ** (-1/3)) if iqr > 0 else None
    max_abs = float(np.percentile(np.abs(errors), 99))
    if bin_width and bin_width > 0:
        nbins = int(np.clip((2 * max_abs) / bin_width, 12, 80))
    else:
        nbins = 30
    bins = np.linspace(-max_abs, max_abs, nbins + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    counts, bin_edges, _ = ax.hist(errors, bins=bins, color='skyblue', edgecolor='black', alpha=0.85)

    # Linee di riferimento e intervallo 90%
    ax.axvline(0, color='black', linestyle='--', linewidth=1, label='Errore 0')
    ax.axvline(err_mean, color='#d62728', linestyle='-', linewidth=1.5, label=f"Media {err_mean:.2f}s")
    ax.axvline(err_median, color='#1f77b4', linestyle=':', linewidth=1.5, label=f"Mediana {err_median:.2f}s")
    ax.axvspan(p5, p95, color='grey', alpha=0.15, label='Intervallo 90%')

    ax.set_xlabel('Errore (Gap reale - predetto) [s]')
    ax.set_ylabel('Frequenza')
    ax.set_title('Distribuzione degli errori di regressione')

    # KDE su asse gemello (densità)
    ax2 = ax.twinx()
    sns.kdeplot(x=errors, ax=ax2, color='crimson', linewidth=2, label='KDE (densità)')
    ax2.set_ylabel('Densità stimata')

    # Legenda combinata (entrambi assi)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Riquadro metriche
    txt = f"MAE: {mae:.2f}s\nRMSE: {rmse:.2f}s\nσ: {err_std:.2f}s"
    ax.text(0.98, 0.95, txt, transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='grey', alpha=0.9))

    plt.tight_layout()
    plt.savefig('modello/istogramma_errori_gap.png', dpi=150)
    plt.savefig('modello/istogramma_errori_gap_migliorato.png', dpi=150)
    plt.close()

if PLOT_MODE != 'errors-only':
    plt.figure(figsize=(4,6))
    plt.boxplot(errors, vert=True, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
    plt.ylabel('Errore (Gap reale - predetto) [s]')
    plt.title('Boxplot errori di regressione')
    plt.tight_layout()
    plt.savefig('modello/boxplot_errori_gap.png', dpi=150)
    plt.close()

# Tabella confronto reali/predetti (prime 20 righe)
if PLOT_MODE != 'errors-only':
    df_results.head(20).to_csv('modello/confronto_gap_reale_predetto.csv', index=False)

# (Funzione di predizione per combinazione rimossa su richiesta)

# === Nuovi grafici: errori per ciascun modello e variabili alternative ===
out_err_dir = 'modello/errori'
os.makedirs(out_err_dir, exist_ok=True)

def freedman_bins(x, symmetric=False, clip_p=99, min_bins=12, max_bins=80):
    x = np.asarray(x)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    n = max(len(x), 1)
    bw = 2 * iqr * (n ** (-1/3)) if iqr > 0 else None
    if symmetric:
        max_abs = float(np.percentile(np.abs(x), clip_p))
        if bw and bw > 0:
            nb = int(np.clip((2 * max_abs) / bw, min_bins, max_bins))
        else:
            nb = 30
        bins = np.linspace(-max_abs, max_abs, nb + 1)
    else:
        lo, hi = np.percentile(x, [100-clip_p, clip_p])
        if bw and bw > 0:
            nb = int(np.clip((hi - lo) / bw, min_bins, max_bins))
        else:
            nb = 30
        bins = np.linspace(lo, hi, nb + 1)
    return bins

def plot_error_hist_generic(values, xlabel, title, out_path, show_zero=False):
    vals = np.asarray(values)
    symmetric = show_zero
    bins = freedman_bins(vals, symmetric=symmetric)
    mean_v = float(np.mean(vals))
    median_v = float(np.median(vals))
    std_v = float(np.std(vals, ddof=1))
    p5, p95 = np.percentile(vals, [5, 95])
    fig, ax = plt.subplots(figsize=(9,5))
    ax.hist(vals, bins=bins, color='skyblue', edgecolor='black', alpha=0.85)
    if show_zero:
        ax.axvline(0, color='black', linestyle='--', linewidth=1, label='0')
    ax.axvline(mean_v, color='#d62728', linestyle='-', linewidth=1.5, label=f"Media {mean_v:.2f}")
    ax.axvline(median_v, color='#1f77b4', linestyle=':', linewidth=1.5, label=f"Mediana {median_v:.2f}")
    ax.axvspan(p5, p95, color='grey', alpha=0.15, label='Intervallo 90%')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequenza')
    ax.set_title(title)
    ax2 = ax.twinx()
    sns.kdeplot(x=vals, ax=ax2, color='crimson', linewidth=2, label='KDE (densità)')
    ax2.set_ylabel('Densità stimata')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_error_hist_advanced(resid, mae, rmse, out_path, title='Distribuzione degli errori di regressione'):
    resid = np.asarray(resid)
    # Statistiche chiave
    err_mean = float(np.mean(resid))
    err_median = float(np.median(resid))
    err_std = float(np.std(resid, ddof=1))
    p5, p95 = np.percentile(resid, [5, 95])
    # Bins simmetrici attorno a 0 con regola di Freedman–Diaconis
    iqr = np.subtract(*np.percentile(resid, [75, 25]))
    n = max(len(resid), 1)
    bin_width = 2 * iqr * (n ** (-1/3)) if iqr > 0 else None
    max_abs = float(np.percentile(np.abs(resid), 99))
    if bin_width and bin_width > 0:
        nbins = int(np.clip((2 * max_abs) / bin_width, 12, 80))
    else:
        nbins = 30
    bins = np.linspace(-max_abs, max_abs, nbins + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(resid, bins=bins, color='skyblue', edgecolor='black', alpha=0.85)
    ax.axvline(0, color='black', linestyle='--', linewidth=1, label='Errore 0')
    ax.axvline(err_mean, color='#d62728', linestyle='-', linewidth=1.5, label=f"Media {err_mean:.2f}s")
    ax.axvline(err_median, color='#1f77b4', linestyle=':', linewidth=1.5, label=f"Mediana {err_median:.2f}s")
    ax.axvspan(p5, p95, color='grey', alpha=0.15, label='Intervallo 90%')
    ax.set_xlabel('Errore (Gap reale - predetto) [s]')
    ax.set_ylabel('Frequenza')
    ax.set_title(title)
    # KDE su asse gemello
    ax2 = ax.twinx()
    sns.kdeplot(x=resid, ax=ax2, color='crimson', linewidth=2, label='KDE (densità)')
    ax2.set_ylabel('Densità stimata')
    # Legenda combinata
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    # Riquadro metriche
    txt = f"MAE: {mae:.2f}s\nRMSE: {rmse:.2f}s\nσ: {err_std:.2f}s"
    ax.text(0.98, 0.95, txt, transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='grey', alpha=0.9))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_multi_kde(series_dict, xlabel, title, out_path, symmetric=False, clip_p=99, palette=None):
    # series_dict: { model_name: np.array([...]) }
    if palette is None:
        palette = sns.color_palette('tab10')
    # Calcola limiti asse x comuni
    all_vals = np.concatenate([np.asarray(v) for v in series_dict.values() if len(v) > 0])
    if symmetric:
        max_abs = float(np.percentile(np.abs(all_vals), clip_p))
        xlim = (-max_abs, max_abs)
    else:
        lo = float(np.percentile(all_vals, 100-clip_p))
        hi = float(np.percentile(all_vals, clip_p))
        xlim = (max(0.0, lo), hi)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, vals) in enumerate(series_dict.items()):
        if len(vals) == 0:
            continue
        sns.kdeplot(x=np.asarray(vals), ax=ax, label=name, linewidth=2, color=palette[i % len(palette)])
    if symmetric:
        ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.8)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Densità stimata')
    ax.set_title(title)
    ax.legend(title='Modello')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_error_hist_panel(ax, resid, mae, rmse, title, bins=None):
    resid = np.asarray(resid)
    if bins is None:
        bins = freedman_bins(resid, symmetric=True)
    err_mean = float(np.mean(resid))
    err_median = float(np.median(resid))
    err_std = float(np.std(resid, ddof=1))
    p5, p95 = np.percentile(resid, [5, 95])
    ax.hist(resid, bins=bins, color='skyblue', edgecolor='black', alpha=0.85)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.axvline(err_mean, color='#d62728', linestyle='-', linewidth=1.2)
    ax.axvline(err_median, color='#1f77b4', linestyle=':', linewidth=1.2)
    ax.axvspan(p5, p95, color='grey', alpha=0.12)
    ax.set_title(title, fontsize=13)
    # KDE su asse gemello con etichette minime
    ax2 = ax.twinx()
    sns.kdeplot(x=resid, ax=ax2, color='crimson', linewidth=1.6)
    ax2.set_yticks([])
    ax2.set_ylabel('')
    # Box metriche compatto
    txt = f"MAE {mae:.1f}s\nRMSE {rmse:.1f}s"
    ax.text(0.98, 0.95, txt, transform=ax.transAxes, ha='right', va='top',
            fontsize=9, bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='grey', alpha=0.9))

for name in labels:
    y_pred = results[name]['y_pred']
    resid = y_test - y_pred
    # Salva solo l'istogramma errori avanzato per ogni modello, in sottocartella dedicata
    mae_m = float(mean_absolute_error(y_test, y_pred))
    rmse_m = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    model_dir = os.path.join(out_err_dir, name.lower().replace(' ','_').replace('(','').replace(')',''))
    os.makedirs(model_dir, exist_ok=True)
    out_path = os.path.join(model_dir, 'istogramma_errori_gap.png')
    plot_error_hist_advanced(resid, mae=mae_m, rmse=rmse_m, out_path=out_path,
                             title='Distribuzione degli errori di regressione')

# Grafici di confronto unici per tipo di errore (KDE sovrapposti)
resid_dict = {}
abs_dict = {}
rel_dict = {}
for name in labels:
    y_pred = results[name]['y_pred']
    resid = (y_test - y_pred).to_numpy() if hasattr(y_test, 'to_numpy') else (y_test - y_pred)
    resid_dict[name] = resid
    abs_dict[name] = np.abs(resid)
    mask = np.abs(y_test) > 1e-6
    rel_vals = np.zeros_like(resid)
    rel_vals[mask] = (resid[mask] / y_test[mask]) * 100.0
    rel_dict[name] = rel_vals[mask]

plot_multi_kde(
    resid_dict,
    xlabel='Errore (Gap reale - predetto) [s]',
    title='Confronto distribuzione residui per modello',
    out_path=os.path.join(out_err_dir, 'confronto_residui_kde.png'),
    symmetric=True
)

plot_multi_kde(
    abs_dict,
    xlabel='Errore assoluto |Gap reale - predetto| [s]',
    title='Confronto distribuzione errore assoluto per modello',
    out_path=os.path.join(out_err_dir, 'confronto_errore_assoluto_kde.png'),
    symmetric=False
)

plot_multi_kde(
    rel_dict,
    xlabel='Errore relativo (%)',
    title='Confronto distribuzione errore relativo (%) per modello',
    out_path=os.path.join(out_err_dir, 'confronto_errore_relativo_kde.png'),
    symmetric=True
)

# Griglie con 3 o 6 grafici "istogramma errori" per immagine
try:
    GRID_N = int(os.environ.get('MOTOGP_GRID', '6'))
except ValueError:
    GRID_N = 6
GRID_N = 6 if GRID_N not in (3, 6) else GRID_N

grid_dir = os.path.join(out_err_dir, 'griglie')
os.makedirs(grid_dir, exist_ok=True)

all_resid = [ (name, (y_test - results[name]['y_pred'])) for name in labels ]
shared_bins = freedman_bins(np.concatenate([r.values if hasattr(r, 'values') else np.asarray(r) for _, r in all_resid]), symmetric=True)

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

rows, cols = (2, 3) if GRID_N == 6 else (1, 3)
page = 1
for group in chunks(all_resid, GRID_N):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10) if GRID_N==6 else (15, 5), sharex=True, sharey=True)
    axes_arr = axes.ravel() if GRID_N==6 else axes if isinstance(axes, np.ndarray) else [axes]
    for idx in range(rows*cols):
        if idx < len(group):
            name, resid = group[idx]
            mae_m = float(results[name]['mae'])
            rmse_m = float(results[name]['rmse'])
            plot_error_hist_panel(axes_arr[idx], resid, mae=mae_m, rmse=rmse_m, title=name, bins=shared_bins)
        else:
            axes_arr[idx].axis('off')
    fig.supxlabel('Errore (s)')
    fig.supylabel('Frequenza')
    fig.suptitle('Istogramma errori per modello', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_grid = os.path.join(grid_dir, f'istogramma_errori_gap_grid{GRID_N}_p{page}.png')
    plt.savefig(out_grid, dpi=150)
    plt.close()
    page += 1

if PLOT_MODE != 'errors-only':
    all_resid = np.concatenate([y_test - results[name]['y_pred'] for name in labels])
    shared_bins = freedman_bins(all_resid, symmetric=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for ax, name in zip(axes.ravel(), labels + ['']):
        if name == '':
            ax.axis('off')
            continue
        resid = y_test - results[name]['y_pred']
        ax.hist(resid, bins=shared_bins, color='skyblue', edgecolor='black', alpha=0.85)
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        ax.set_title(name, fontsize=11)
    fig.suptitle('Confronto residui per modello (bins condivisi)', fontsize=20)
    fig.supxlabel('Errore (s)')
    fig.supylabel('Frequenza')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(out_err_dir, 'residui_confronto_griglia.png'), dpi=150)
    plt.close()
