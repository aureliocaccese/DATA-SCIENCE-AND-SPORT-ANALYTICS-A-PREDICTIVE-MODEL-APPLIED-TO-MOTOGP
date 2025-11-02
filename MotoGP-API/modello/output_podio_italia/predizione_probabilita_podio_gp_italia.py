import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
import seaborn as sns
import os
import numpy as np

# Carica dataset classifica e meteo
# Percorsi relativi

df = pd.read_csv('classifica/motogp_results_cleaned_final_normalized.csv')
df_meteo = pd.read_csv('meteo/motogp_weather_data.csv')
df = pd.merge(df, df_meteo, left_on=['Event','Year'], right_on=['Event','Year'], how='left')

# Filtra solo piloti classificati
df = df[pd.to_numeric(df['Position'], errors='coerce').notnull()]
df['Position'] = df['Position'].astype(int)

# Target: podio (1 se posizione <=3, 0 altrimenti)
df['Podio'] = (df['Position'] <= 3).astype(int)

features = ['Rider_normalized', 'Team', 'Event', 'Year', 'Conditions']
X = df[features].copy()
y = df['Podio']

for col in ['Rider_normalized', 'Team', 'Event', 'Conditions']:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
X = X.fillna(-1)

mask_train = (df['Year'] >= 2002) & (df['Year'] <= 2020)
X_train = X[mask_train]
y_train = y[mask_train]
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# XGBoost
xgb_params = {'n_estimators': [100, 200], 'max_depth': [3, 6, 10]}
xgb_grid = RandomizedSearchCV(XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'), xgb_params, cv=5, scoring='roc_auc', n_jobs=-1)
xgb_grid.fit(X_train_bal, y_train_bal)
clf_xgb = xgb_grid.best_estimator_
# XGBoost
xgb_params = {'n_estimators': [100, 200], 'max_depth': [3, 6, 10]}
xgb_grid = RandomizedSearchCV(XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'), xgb_params, cv=5, scoring='roc_auc', n_jobs=-1)
xgb_grid.fit(X_train_bal, y_train_bal)
clf_xgb = xgb_grid.best_estimator_
smote = SMOTE(random_state=42)

X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)


# Random Forest con ottimizzazione iperparametri
rf_params = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'class_weight': ['balanced']}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='roc_auc', n_jobs=-1)
rf_grid.fit(X_train_bal, y_train_bal)
clf_rf = rf_grid.best_estimator_

# LightGBM
lgbm_params = {'n_estimators': [100, 200], 'max_depth': [3, 6, 10]}
lgbm_grid = RandomizedSearchCV(LGBMClassifier(random_state=42), lgbm_params, cv=5, scoring='roc_auc', n_jobs=-1)
lgbm_grid.fit(X_train_bal, y_train_bal)
clf_lgbm = lgbm_grid.best_estimator_

# Validazione incrociata
rf_cv = cross_val_score(clf_rf, X_train_bal, y_train_bal, cv=5, scoring='roc_auc')
lgbm_cv = cross_val_score(clf_lgbm, X_train_bal, y_train_bal, cv=5, scoring='roc_auc')
xgb_cv = cross_val_score(clf_xgb, X_train_bal, y_train_bal, cv=5, scoring='roc_auc')
print(f"Random Forest CV ROC AUC: {rf_cv.mean():.3f}")
print(f"LightGBM CV ROC AUC: {lgbm_cv.mean():.3f}")
print(f"XGBoost CV ROC AUC: {xgb_cv.mean():.3f}")

def predizione_podio(rider, team, event, year, conditions):
    le_rider = LabelEncoder().fit(df['Rider_normalized'].astype(str))
    le_team = LabelEncoder().fit(df['Team'].astype(str))
    le_event = LabelEncoder().fit(df['Event'].astype(str))
    le_cond = LabelEncoder().fit(df['Conditions'].astype(str))
    if rider not in le_rider.classes_:
        raise ValueError(f"Rider '{rider}' non presente nei dati")
    if team not in le_team.classes_:
        raise ValueError(f"Team '{team}' non presente nei dati")
    if event not in le_event.classes_:
        raise ValueError(f"Event '{event}' non presente nei dati")
    if conditions not in le_cond.classes_:
        raise ValueError(f"Condizione meteo '{conditions}' non presente nei dati")
    X_input = pd.DataFrame({
        'Rider_normalized': [rider],
        'Team': [team],
        'Event': [event],
        'Year': [year],
        'Conditions': [conditions]
    })
    X_input['Rider_normalized'] = le_rider.transform(X_input['Rider_normalized'].astype(str))
    X_input['Team'] = le_team.transform(X_input['Team'].astype(str))
    X_input['Event'] = le_event.transform(X_input['Event'].astype(str))
    X_input['Conditions'] = le_cond.transform(X_input['Conditions'].astype(str))
    X_input = X_input.fillna(-1)
    # Ensemble: media delle probabilità dei tre modelli
    proba_rf = clf_rf.predict_proba(X_input)[0][1]
    proba_lgbm = clf_lgbm.predict_proba(X_input)[0][1]
    proba_xgb = clf_xgb.predict_proba(X_input)[0][1]
    proba = np.mean([proba_rf, proba_lgbm, proba_xgb])
    return proba

if __name__ == "__main__":
    output_dir = os.path.join('modello', 'output_podio_italia')
    os.makedirs(output_dir, exist_ok=True)
    anni = list(range(2010, 2021))
    eventi_italia = df[df['Event'].str.lower().str.contains('ita')]['Event'].unique()
    risultati = []
    if len(eventi_italia) == 0:
        print("Nessun evento Italia trovato nei dati.")
    else:
        event = eventi_italia[0]
        for year in anni:
            piloti_anno = df[(df['Event'] == event) & (df['Year'] == year)]['Rider_normalized'].dropna().unique()[:5]
            for rider in piloti_anno:
                riga = df[(df['Rider_normalized'] == rider) & (df['Event'] == event) & (df['Year'] == year) & df['Team'].notnull() & (df['Team'] != '') & df['Conditions'].notnull()]
                if not riga.empty:
                    team = riga.iloc[0]['Team']
                    conditions = riga.iloc[0]['Conditions']
                    try:
                        proba = predizione_podio(rider, team, event, year, conditions)
                        risultati.append({
                            'Rider': rider,
                            'Team': team,
                            'Year': year,
                            'Conditions': conditions,
                            'Prob_podio': proba
                        })
                    except Exception as e:
                        risultati.append({
                            'Rider': rider,
                            'Team': team,
                            'Year': year,
                            'Conditions': conditions,
                            'Prob_podio': None
                        })
    if risultati:
        df_ris = pd.DataFrame(risultati)
        top_n = 3
        piloti_freq = df_ris['Rider'].value_counts().nlargest(top_n).index.tolist()
        df_ris = df_ris[df_ris['Rider'].isin(piloti_freq)]
        csv_path = os.path.join(output_dir, 'probabilita_podio_gp_italia_2010_2020.csv')
        df_ris.to_csv(csv_path, index=False)
        # Grafico migliorato
        df_ris = df_ris.sort_values(['Year', 'Rider'])
        df_ris['label'] = df_ris['Rider'] + ' (' + df_ris['Year'].astype(str) + ')'
        piloti_unici = df_ris['Rider'].unique()
        palette = sns.color_palette('Set1', n_colors=len(piloti_unici))
        pilota2col = {pilota: palette[i % len(palette)] for i, pilota in enumerate(piloti_unici)}
        bar_colors = df_ris['Rider'].map(pilota2col)
        plt.figure(figsize=(20, 8))
        bars = plt.bar(df_ris['label'], df_ris['Prob_podio'], color=bar_colors, edgecolor='black', alpha=0.85, width=0.7)
        plt.ylabel('Probabilità podio', fontsize=16)
        plt.xlabel('Pilota (Anno)', fontsize=16)
        plt.title(f'Probabilità podio GP Italia ({event}) dal 2010 al 2020', fontsize=18, weight="bold")
        plt.xticks(rotation=60, ha='right', fontsize=13)
        plt.yticks(fontsize=14)
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        # Valori sulle barre (solo >0%)
        for bar, val in zip(bars, df_ris['Prob_podio']):
            if val > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.07, f"{val:.0%}", ha='center', va='top', fontsize=14, fontweight='bold', color='black')
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=pilota2col[pilota], label=pilota) for pilota in piloti_unici]
    plt.legend(handles=legend_handles, title='Pilota', loc='upper right', fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'grafico_probabilita_podio_gp_italia_2010_2020.png')
    plt.savefig(plot_path, dpi=300)
    plt.show()
    print(f"Risultati salvati in: {csv_path}\nGrafico salvato in: {plot_path}")

        # --- PREVISIONE DAL 2020 AL 2030 SUGLI STESSI PILOTI ---
    anni_pred = list(range(2020, 2031))
    piloti_pred = piloti_freq
    risultati_pred = []
    rng = np.random.default_rng(42)
    for rider in piloti_pred:
            storico = df[(df['Rider_normalized'] == rider) & (df['Event'] == event)]
            for year in anni_pred:
                storico_anno = storico[storico['Year'] == year]
                team_possibili = storico_anno['Team'].dropna().unique()
                condizioni_possibili = storico_anno['Conditions'].dropna().unique()
                # Se non ci sono dati per quell'anno, usa i dati storici generali
                if len(team_possibili) == 0:
                    team_possibili = storico['Team'].dropna().unique()
                if len(condizioni_possibili) == 0:
                    condizioni_possibili = storico['Conditions'].dropna().unique()
                # Se ancora non ci sono dati, salta la previsione
                if len(team_possibili) == 0 or len(condizioni_possibili) == 0:
                    continue
                team = rng.choice(team_possibili)
                conditions = rng.choice(condizioni_possibili)
                try:
                    proba = predizione_podio(rider, team, event, year, conditions)
                    risultati_pred.append({
                        'Rider': rider,
                        'Team': team,
                        'Year': year,
                        'Conditions': conditions,
                        'Prob_podio': proba
                    })
                except Exception as e:
                    risultati_pred.append({
                        'Rider': rider,
                        'Team': team,
                        'Year': year,
                        'Conditions': conditions,
                        'Prob_podio': None
                    })
if risultati_pred:
            df_pred = pd.DataFrame(risultati_pred)
            csv_pred_path = os.path.join(output_dir, 'previsione_probabilita_podio_gp_italia_2020_2030.csv')
            df_pred.to_csv(csv_pred_path, index=False)
            df_pred = df_pred.dropna(subset=['Prob_podio'])
            df_pred = df_pred.sort_values(['Year', 'Rider'])
            df_pred['label'] = df_pred['Rider'] + ' (' + df_pred['Year'].astype(str) + ')'
            piloti_unici_pred = df_pred['Rider'].unique()
            palette_pred = sns.color_palette('Set1', n_colors=len(piloti_unici_pred))
            pilota2col_pred = {pilota: palette_pred[i % len(palette_pred)] for i, pilota in enumerate(piloti_unici_pred)}
            bar_colors_pred = df_pred['Rider'].map(pilota2col_pred)
            plt.figure(figsize=(20, 8))
            bars = plt.bar(df_pred['label'], df_pred['Prob_podio'], color=bar_colors_pred, edgecolor='black', alpha=0.85, width=0.7)
            plt.ylabel('Probabilità podio', fontsize=16)
            plt.xlabel('Pilota (Anno)', fontsize=16)
            plt.title(f'Previsione probabilità podio GP Italia ({event}) 2020-2030', fontsize=18, weight="bold")
            plt.xticks(rotation=60, ha='right', fontsize=13)
            plt.yticks(fontsize=14)
            plt.ylim(0, 1)
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            for bar, val in zip(bars, df_pred['Prob_podio']):
                if val > 0:
                    if val < 0.08:
                        # Percentuale troppo bassa: posiziona sopra la barra
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{val:.0%}", ha='center', va='bottom', fontsize=14, fontweight='bold', color='black')
                    else:
                        # Percentuale normale: posiziona dentro la barra
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.07, f"{val:.0%}", ha='center', va='top', fontsize=14, fontweight='bold', color='black')
            from matplotlib.patches import Patch
            legend_handles = [Patch(color=pilota2col_pred[pilota], label=pilota) for pilota in piloti_unici_pred]
            plt.legend(handles=legend_handles, title='Pilota', loc='upper right', fontsize=13)
            plt.tight_layout()
            plot_pred_path = os.path.join(output_dir, 'grafico_previsione_probabilita_podio_gp_italia_2020_2030.png')
            plt.savefig(plot_pred_path, dpi=300)
            plt.show()
            print(f"Previsioni salvate in: {csv_pred_path}\nGrafico salvato in: {plot_pred_path}")
