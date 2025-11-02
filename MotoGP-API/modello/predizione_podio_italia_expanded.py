import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier


def load_data():
    df = pd.read_csv('classifica/motogp_results_expanded_features.csv')
    # Pulisci tipi
    df['Event'] = df['Event'].astype(str).str.lower().str.strip()
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
    # Tieni solo posizioni valide
    df = df[pd.to_numeric(df['Position'], errors='coerce').notnull()].copy()
    df['Position'] = df['Position'].astype(int)
    # Target Podio (già presente come 0/1, ma lo rigeneriamo per sicurezza)
    if 'Podio' in df.columns:
        df['Podio'] = pd.to_numeric(df['Podio'], errors='coerce').fillna(0).astype(int)
    else:
        df['Podio'] = (df['Position'] <= 3).astype(int)
    return df


def _to_float_pct(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace('%','').replace(',','.')
    try:
        return float(s)
    except:
        return np.nan

def _to_float_deg(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace('º','').replace('°','').replace(',','.')
    try:
        return float(s)
    except:
        return np.nan

def _cond_simple(s):
    st = str(s).lower()
    if 'rain' in st or 'wet' in st:
        return 'wet/rain'
    if 'cloud' in st or 'overcast' in st:
        return 'cloudy'
    if 'sun' in st or 'clear' in st:
        return 'clear'
    return 'other'

def _trackcond_simple(s):
    st = str(s).lower()
    if 'wet' in st:
        return 'wet'
    if 'dry' in st:
        return 'dry'
    return 'other'


def build_features(df):
    # Selezione di feature utili dal dataset "expanded"
    # Ingegnerizzazione meteo numerica e flag
    if 'Temperature' in df.columns:
        df['Temp_num'] = df['Temperature'].apply(_to_float_deg)
    else:
        df['Temp_num'] = np.nan
    if 'Humidity' in df.columns:
        df['Humidity_num'] = df['Humidity'].apply(_to_float_pct)
    else:
        df['Humidity_num'] = np.nan
    if 'Ground Temp' in df.columns:
        df['GroundTemp_num'] = df['Ground Temp'].apply(_to_float_deg)
    else:
        df['GroundTemp_num'] = np.nan
    df['TempGroundDiff'] = df['GroundTemp_num'] - df['Temp_num']

    # Flag meteo semplificati
    df['Cond_Simple'] = df.get('Conditions', '').apply(_cond_simple)
    df['TrackCond_Simple'] = df.get('Track Conditions', '').apply(_trackcond_simple)

    feat_cat = [
        'Rider_normalized','Team','Grand Prix','Event',
        'Cond_Simple','TrackCond_Simple'
    ]
    feat_num = [
        'Year','Humidity_num','GroundTemp_num','Temp_num','TempGroundDiff',
        'Avg_Position_Last3','Podiums_Last5',
        'Points_Streak','length_km','width_m','right corners','left corners',
        'Race_Number_Season','Rider_Experience','Season_Points_So_Far',
        'Team_Avg_Position','Difficult_Conditions','Rider_Circuit_Avg'
    ]
    # Filtra alle colonne esistenti
    feat_cat = [c for c in feat_cat if c in df.columns]
    feat_num = [c for c in feat_num if c in df.columns]
    features = feat_cat + feat_num

    X = df[features].copy()
    y = df['Podio'].astype(int)

    # Tipi e NaN
    for c in feat_cat:
        X[c] = X[c].astype(str).fillna('Unknown')
    for c in feat_num:
        X[c] = pd.to_numeric(X[c], errors='coerce')

    return X, y, feat_cat, feat_num


def train_cv_predict(X, y, feat_cat, feat_num):
    cat_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    num_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler())
    ])
    pre = ColumnTransformer([
        ('cat', cat_pipe, feat_cat),
        ('num', num_pipe, feat_num)
    ])
    clf = GradientBoostingClassifier(random_state=42)
    pipe = Pipeline([('pre', pre), ('clf', clf)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba = cross_val_predict(pipe, X, y, cv=cv, method='predict_proba')[:,1]
    return proba


def plot_italy(df, proba, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    dfp = df.copy()
    dfp['proba_podio'] = proba
    ita = dfp[dfp['Event'] == 'ita']
    agg = ita.groupby('Rider_normalized')['proba_podio'].mean().sort_values(ascending=False)
    top = agg.head(20)
    years_by_rider = (
        ita.groupby('Rider_normalized')['Year']
           .apply(lambda s: sorted(set(pd.to_numeric(s, errors='coerce').dropna().astype(int).tolist())))
           .to_dict()
    )

    def _fmt_years(ys):
        if not ys:
            return ''
        ys = sorted(set(int(y) for y in ys))
        if len(ys) == 1:
            return f"{ys[0]}"
        return f"{ys[0]}–{ys[-1]}"
    # salva
    ita[['Rider_normalized','Team','Year','Conditions','Track Conditions','proba_podio']].to_csv(
        os.path.join(out_dir, 'prob_podio_italia_entries.csv'), index=False
    )
    top_df = top.reset_index()
    top_df['Years'] = top_df['Rider_normalized'].map(lambda r: ','.join(str(y) for y in years_by_rider.get(r, [])))
    top_df['Year_Range'] = top_df['Rider_normalized'].map(lambda r: _fmt_years(years_by_rider.get(r, [])))
    top_df.to_csv(os.path.join(out_dir, 'prob_podio_italia_top20.csv'), index=False)
    # grafico
    plt.figure(figsize=(12,6))
    x = np.arange(len(top))
    labels = [f"{r} ({_fmt_years(years_by_rider.get(r, []))})" for r in top.index]
    bars = plt.bar(x, top.values, color='crimson')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.ylabel('Probabilità di podio')
    plt.title("Probabilità di podio – GP d'Italia (dataset expanded, CV 5-fold)")
    for b, v in zip(bars, top.values):
        plt.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))
    plt.tight_layout()
    out_png = os.path.join(out_dir, 'barplot_probabilita_podio_italia_expanded.png')
    plt.savefig(out_png, dpi=150)
    plt.close()
    return out_png


def plot_italy_by_year(df, proba, out_dir, top_k=6, last_n_years=10):
    os.makedirs(out_dir, exist_ok=True)
    dfp = df.copy()
    dfp['proba_podio'] = proba
    ita = dfp[dfp['Event'] == 'ita'].copy()
    if ita.empty:
        return None
    # Media per anno e pilota (storico)
    g = ita.groupby(['Year', 'Rider_normalized'])['proba_podio'].mean().reset_index()
    # Seleziona finestra recente
    years_sorted = sorted(g['Year'].dropna().unique().astype(int))
    if not years_sorted:
        return None
    last_years = years_sorted[-last_n_years:]
    g_recent = g[g['Year'].isin(last_years)]
    # Top_k piloti per media nel periodo recente
    top_riders = (
        g_recent.groupby('Rider_normalized')['proba_podio']
                .mean()
                .sort_values(ascending=False)
                .head(top_k)
                .index.tolist()
    )
    g_recent = g_recent[g_recent['Rider_normalized'].isin(top_riders)]
    if g_recent.empty:
        return None
    # Pivot anni x piloti (solo storico recente)
    pivot = g_recent.pivot_table(index='Year', columns='Rider_normalized', values='proba_podio', aggfunc='mean')
    pivot = pivot.sort_index()
    # Salva CSV storico recente
    csv_path = os.path.join(out_dir, 'prob_podio_italia_by_year_storico_recente.csv')
    pivot.to_csv(csv_path, index_label='Year')
    # Plot linee storico (senza proiezione)
    plt.figure(figsize=(12, 7))
    colors = plt.cm.tab10.colors
    for i, rider in enumerate(pivot.columns):
        series = pivot[rider]
        color = colors[i % len(colors)]
        plt.plot(series.index, series.values, label=rider, color=color, linewidth=2, marker='o', ms=4)
        # etichetta ultimo punto
        plt.text(series.index.max() + 0.2, float(series.iloc[-1]), rider, color=color, va='center')
    plt.ylim(0, 1)
    plt.xlim(min(last_years), max(last_years) + 1)
    plt.xlabel('Anno')
    plt.ylabel('Probabilità di podio')
    plt.title("Probabilità podio GP d'Italia – storico recente (senza proiezione)")
    plt.grid(True, alpha=0.3)
    plt.legend(title='Pilota', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    out_png = os.path.join(out_dir, 'line_prob_podio_italia_per_anno_storico_recente.png')
    plt.savefig(out_png, dpi=150)
    plt.close()
    return out_png


if __name__ == '__main__':
    out_dir = 'modello/output_podio_italia_expanded'
    df = load_data()
    X, y, feat_cat, feat_num = build_features(df)
    proba = train_cv_predict(X, y, feat_cat, feat_num)
    png = plot_italy(df, proba, out_dir)
    print(f"Grafico salvato in: {png}")
    png_year = plot_italy_by_year(df, proba, out_dir, top_k=6, last_n_years=10)
    if png_year:
        print(f"Grafico per anno (storico) salvato in: {png_year}")
    else:
        print("Nessun dato per GP d'Italia per generare il grafico per anno (storico).")
