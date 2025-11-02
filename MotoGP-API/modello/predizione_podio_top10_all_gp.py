import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    HAS_XGB = True
    HAS_LGBM = True
except Exception:
    HAS_XGB = False
    HAS_LGBM = False


def prepare_data():
    df = pd.read_csv('classifica/motogp_results_expanded_features.csv')
    df = df[pd.to_numeric(df['Position'], errors='coerce').notnull()].copy()
    df['Position'] = df['Position'].astype(int)
    df['Podio'] = (df['Position'] <= 3).astype(int)
    # Feature base + caratteristiche tecniche circuiti
    features = ['Rider_normalized', 'Team', 'Event', 'Year', 'Conditions', 
                'length_km', 'width_m', 'right corners', 'left corners', 
                'Rider_Experience', 'Team_Avg_Position']
    X = df[features].copy()
    y = df['Podio'].copy()

    encoders = {}
    categorical_cols = ['Rider_normalized', 'Team', 'Event', 'Conditions']
    for col in categorical_cols:
        le = LabelEncoder().fit(X[col].astype(str))
        encoders[col] = le
        X[col] = le.transform(X[col].astype(str))
    X = X.fillna(-1)

    mask_train = (df['Year'] >= 2002) & (df['Year'] <= 2020)
    X_train = X[mask_train]
    y_train = y[mask_train]

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    models = []
    # RandomForest baseline (robusta e veloce)
    models.append(
        ('rf', RandomForestClassifier(n_estimators=300, max_depth=20, class_weight='balanced', random_state=42))
    )
    if HAS_LGBM:
        models.append(('lgbm', LGBMClassifier(n_estimators=400, random_state=42)))
    if HAS_XGB:
        models.append(('xgb', XGBClassifier(n_estimators=400, max_depth=6, subsample=0.9, colsample_bytree=0.9,
                                            random_state=42, use_label_encoder=False, eval_metric='logloss')))

    for name, model in models:
        model.fit(X_train_bal, y_train_bal)

    return df, encoders, models


def predict_proba_podium(row_df, encoders, models):
    X_input = row_df.copy()
    for col in ['Rider_normalized', 'Team', 'Event', 'Conditions']:
        le = encoders[col]
        val = X_input.iloc[0][col]
        if val not in le.classes_:
            # unseen category -> add temporarily
            le_classes = le.classes_.tolist() + [val]
            le.classes_ = np.array(le_classes)
        X_input[col] = le.transform(X_input[col].astype(str))
    # Seleziona tutte le feature utilizzate nel training, incluse quelle tecniche
    feature_cols = ['Rider_normalized', 'Team', 'Event', 'Year', 'Conditions', 
                   'length_km', 'width_m', 'right corners', 'left corners', 
                   'Rider_Experience', 'Team_Avg_Position']
    X_input = X_input[feature_cols].fillna(-1)
    probs = []
    for name, model in models:
        if hasattr(model, 'predict_proba'):
            probs.append(model.predict_proba(X_input)[0][1])
    if not probs:
        return None
    return float(np.mean(probs))


def pick_team_conditions(df, rider, event, year):
    # Preferisci dati dell'anno specifico
    df_re = df[(df['Rider_normalized'] == rider) & (df['Event'] == event)]
    df_year = df_re[df_re['Year'] == year]
    if not df_year.empty:
        team = df_year['Team'].dropna().mode()
        cond = df_year['Conditions'].dropna().mode()
        if len(team) and len(cond):
            return team.iloc[0], cond.iloc[0]
    # Altrimenti prendi il più frequente sullo stesso evento
    if not df_re.empty:
        team = df_re['Team'].dropna().mode()
        cond = df_re['Conditions'].dropna().mode()
        if len(team) and len(cond):
            return team.iloc[0], cond.iloc[0]
    # Fallback: più frequente globale per rider
    df_r = df[df['Rider_normalized'] == rider]
    if not df_r.empty:
        team = df_r['Team'].dropna().mode()
        cond = df_r['Conditions'].dropna().mode()
        if len(team) and len(cond):
            return team.iloc[0], cond.iloc[0]
    return None, None


def plot_event_outputs(df_event, event_slug, out_dir, event_full_name=None):
    os.makedirs(out_dir, exist_ok=True)
    # Heatmap Rider x Year
    pivot = df_event.pivot(index='Rider', columns='Year', values='Prob_podio').fillna(0)
    plt.figure(figsize=(14, 6))
    sns.heatmap(pivot, annot=True, fmt='.0%', cmap='YlGnBu', vmin=0, vmax=1, cbar_kws={'label': 'Probabilità podio'})
    plt.title(f'Probabilità podio (Top 10) – {event_full_name or event_slug} – 2020-2030', fontsize=16, weight='bold')
    plt.ylabel('Pilota')
    plt.xlabel('Anno')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{event_slug}_heatmap_top10_2020_2030.png'), dpi=200)
    plt.close()

    # Barplot media per pilota (ordinato)
    mean_by_rider = df_event.groupby('Rider', as_index=False)['Prob_podio'].mean()
    mean_by_rider = mean_by_rider.sort_values('Prob_podio', ascending=False)
    plt.figure(figsize=(12, 6))
    bars = plt.bar(mean_by_rider['Rider'], mean_by_rider['Prob_podio'], color=sns.color_palette('Set2', n_colors=len(mean_by_rider)), edgecolor='black', alpha=0.9)
    for bar, val in zip(bars, mean_by_rider['Prob_podio']):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.0%}', ha='center', va='bottom', fontsize=10, weight='bold')
    plt.ylim(0, 1)
    plt.ylabel('Probabilità media podio (2020–2030)')
    title_name = event_full_name or event_slug
    plt.title(f'Media probabilità podio (Top 10) – {title_name} – 2020-2030', fontsize=16, weight='bold')
    plt.xticks(rotation=30, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{event_slug}_bar_mean_top10_2020_2030.png'), dpi=200)
    plt.close()


def main():
    df, encoders, models = prepare_data()
    out_root = os.path.join('modello', 'predizioni_podio_top10_all_gp')
    os.makedirs(out_root, exist_ok=True)

    # Top 15 piloti per frequenza globale nel dataset
    top10 = df['Rider_normalized'].value_counts().nlargest(15).index.tolist()
    events = sorted(df['Event'].dropna().unique())
    years = list(range(2020, 2031))

    all_records = []
    # Mappa Event -> nome completo (Grand Prix) se disponibile
    event_full_map = {}
    if 'Grand Prix' in df.columns:
        for ev, sub in df.groupby('Event'):
            gp = sub['Grand Prix'].dropna()
            event_full_map[ev] = gp.mode().iloc[0] if not gp.empty else ev
    else:
        event_full_map = {ev: ev for ev in events}

    for event in events:
        event_slug = str(event).lower().replace(' ', '_')
        event_full_name = event_full_map.get(event, str(event))
        recs_event = []
        for rider in top10:
            for year in years:
                team, cond = pick_team_conditions(df, rider, event, year)
                if team is None or cond is None:
                    continue
                
                # Ottieni caratteristiche tecniche del circuito dal dataset
                circuit_data = df[df['Event'] == event].iloc[0]
                
                # Calcola esperienza del pilota e media team (approssimazione)
                rider_exp = len(df[(df['Rider_normalized'] == rider) & (df['Year'] < year)])
                team_avg = df[(df['Team'] == team) & (df['Year'] < year)]['Position'].mean()
                if pd.isna(team_avg):
                    team_avg = 10.0  # fallback
                
                row = pd.DataFrame({
                    'Rider_normalized': [rider], 
                    'Team': [team], 
                    'Event': [event], 
                    'Year': [year], 
                    'Conditions': [cond],
                    'length_km': [circuit_data.get('length_km', 5.0)],
                    'width_m': [circuit_data.get('width_m', 15.0)],
                    'right corners': [circuit_data.get('right corners', 8.0)],
                    'left corners': [circuit_data.get('left corners', 7.0)],
                    'Rider_Experience': [rider_exp],
                    'Team_Avg_Position': [team_avg]
                })
                proba = predict_proba_podium(row, encoders, models)
                if proba is None:
                    continue
                rec = {'Event': event, 'Rider': rider, 'Team': team, 'Year': year, 'Conditions': cond, 'Prob_podio': proba}
                recs_event.append(rec)
                all_records.append(rec)
        if recs_event:
            df_event = pd.DataFrame(recs_event)
            event_dir = os.path.join(out_root, event_slug)
            os.makedirs(event_dir, exist_ok=True)
            csv_path = os.path.join(event_dir, f'{event_slug}_top10_pred_2020_2030.csv')
            df_event.to_csv(csv_path, index=False)
            plot_event_outputs(df_event, event_slug, event_dir, event_full_name=event_full_name)

    # Salva aggregato complessivo
    if all_records:
        df_all = pd.DataFrame(all_records)
        df_all.to_csv(os.path.join(out_root, 'all_events_top10_pred_2020_2030.csv'), index=False)


if __name__ == '__main__':
    main()
