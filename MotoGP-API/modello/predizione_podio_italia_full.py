import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


def load_and_merge_data():
    df = pd.read_csv('classifica/motogp_results_cleaned_final_normalized.csv')
    dfw = pd.read_csv('meteo/motogp_weather_data.csv')

    # Normalizza chiavi e tipi per il join
    df['Event'] = df['Event'].astype(str).str.lower().str.strip()
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
    dfw['Event'] = dfw['Event'].astype(str).str.lower().str.strip()
    dfw['Year'] = pd.to_numeric(dfw['Year'], errors='coerce').astype('Int64')

    # Pulisci meteo
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
    # Mantieni anche condizioni testuali
    if 'Track Conditions' not in dfw.columns:
        dfw['Track Conditions'] = np.nan
    if 'Conditions' not in dfw.columns:
        dfw['Conditions'] = np.nan

    # Merge
    df = df.merge(
        dfw[['Year','Event','Conditions','Track Conditions','Humidity_num','GroundTemp_num']],
        on=['Year','Event'], how='left'
    )

    return df


def build_features(df):
    # Tieni solo righe con posizione numerica
    df = df[pd.to_numeric(df['Position'], errors='coerce').notnull()].copy()
    df['Position'] = df['Position'].astype(int)
    # Target: Podio
    df['Podio'] = (df['Position'] <= 3).astype(int)

    # Feature considerate "più utili" disponibili nel dataset
    feat_cat = ['Rider_normalized', 'Team', 'Grand Prix', 'Event', 'Conditions', 'Track Conditions']
    feat_num = ['Year', 'Humidity_num', 'GroundTemp_num']
    features = feat_cat + feat_num

    X = df[features].copy()
    y = df['Podio'].astype(int)

    # Gestione NaN: categorie a stringa, numeriche imputate con mediana (via pipeline)
    for col in feat_cat:
        X[col] = X[col].astype(str).fillna('Unknown')
    for col in feat_num:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    return X, y, feat_cat, feat_num, df


def train_and_predict(X, y, feat_cat, feat_num):
    # Pipeline: OneHot sulle categoriche, StandardScaler sulle numeriche, RF classifier
    pre = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), feat_cat),
            ('num', StandardScaler(), feat_num),
        ]
    )
    clf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, class_weight='balanced')
    pipe = Pipeline(steps=[('pre', pre), ('clf', clf)])

    # CV pred proba su tutto il dataset per evitare overfitting
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_proba = cross_val_predict(pipe, X, y, cv=cv, method='predict_proba')[:, 1]
    return y_proba


def plot_italy_probabilities(df, y_proba, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df_pred = df.copy()
    df_pred['proba_podio'] = y_proba

    # Filtra GP d'Italia (event code 'ita')
    ita = df_pred[df_pred['Event'].astype(str).str.lower() == 'ita']

    # Aggrega per pilota (media delle apparizioni al GP d'Italia)
    agg = ita.groupby('Rider_normalized')['proba_podio'].mean().sort_values(ascending=False)
    top = agg.head(20)

    # Salva CSV completo e top
    ita[['Rider_normalized','Team','Year','Conditions','Track Conditions','Humidity_num','GroundTemp_num','proba_podio']].to_csv(
        os.path.join(output_dir, 'prob_podio_italia_per_entry.csv'), index=False
    )
    top.reset_index().to_csv(os.path.join(output_dir, 'prob_podio_italia_top20.csv'), index=False)

    # Barplot
    plt.figure(figsize=(10,6))
    bars = plt.bar(top.index, top.values, color='teal')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.ylabel('Probabilità di podio')
    plt.title('Probabilità di podio – GP d\'Italia (media CV su tutto il dataset)')
    # Etichette sopra le barre
    for b, v in zip(bars, top.values):
        plt.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))
    plt.tight_layout()
    out_png = os.path.join(output_dir, 'barplot_probabilita_podio_italia_full.png')
    plt.savefig(out_png, dpi=150)
    plt.close()
    return out_png


if __name__ == '__main__':
    out_dir = 'modello/output_podio_italia'
    df = load_and_merge_data()
    X, y, feat_cat, feat_num, df_full = build_features(df)
    y_proba = train_and_predict(X, y, feat_cat, feat_num)
    png = plot_italy_probabilities(df_full, y_proba, out_dir)
    print(f"Grafico salvato in: {png}")
