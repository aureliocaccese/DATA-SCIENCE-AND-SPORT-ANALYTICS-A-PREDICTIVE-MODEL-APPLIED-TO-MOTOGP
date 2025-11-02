def predizione_podio_logreg(rider, team, event, year, conditions):
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
    # Logistic Regression: addestra e predici
    clf_logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
    mask_train = (df['Year'] >= 2002) & (df['Year'] <= 2020)
    X_train = X[mask_train]
    y_train = y[mask_train]
    clf_logreg.fit(X_train, y_train)
    proba = clf_logreg.predict_proba(X_input)[0][1]
    return proba
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Carica dataset classifica e meteo (percorsi relativi alla root del workspace)
df = pd.read_csv('classifica/motogp_results_cleaned_final_normalized.csv')
df_meteo = pd.read_csv('meteo/motogp_weather_data.csv')
df = pd.merge(df, df_meteo, left_on=['Event','Year'], right_on=['Event','Year'], how='left')

# Filtra solo piloti classificati
df = df[pd.to_numeric(df['Position'], errors='coerce').notnull()]
df['Position'] = df['Position'].astype(int)

# Target: podio (1 se posizione <=3, 0 altrimenti)
df['Podio'] = (df['Position'] <= 3).astype(int)

# Feature: pilota, team, pista, anno, condizioni meteo principali
features = ['Rider_normalized', 'Team', 'Event', 'Year', 'Conditions']
X = df[features].copy()
y = df['Podio']

# Encoding variabili categoriche
for col in ['Rider_normalized', 'Team', 'Event', 'Conditions']:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
X = X.fillna(-1)

# Addestra modello solo su dati dal 2002 al 2020
mask_train = (df['Year'] >= 2002) & (df['Year'] <= 2020)
X_train = X[mask_train]
y_train = y[mask_train]
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

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
    proba = clf.predict_proba(X_input)[0][1]
    return proba

if __name__ == "__main__":
    # --- SCENARI MULTIPLI E ESPORTAZIONE ---
    scenari = [
        {'Rider': 'Bagnaia', 'Team': 'Pramac Racing', 'Event': 'ita', 'Year': 2020, 'Conditions': 'Clear'},
        {'Rider': 'Quartararo', 'Team': 'Petronas Yamaha SRT', 'Event': 'ita', 'Year': 2020, 'Conditions': 'Clear'},
        {'Rider': 'Martin', 'Team': 'Pramac Racing', 'Event': 'ita', 'Year': 2021, 'Conditions': 'Clear'},
        {'Rider': 'Marquez', 'Team': 'Repsol Honda Team', 'Event': 'ita', 'Year': 2019, 'Conditions': 'Partly Cloudy'},
        {'Rider': 'Bastianini', 'Team': 'Avintia Esponsorama', 'Event': 'ita', 'Year': 2021, 'Conditions': 'Clear'}
    ]
    risultati = []
    for s in scenari:
        try:
            prob_rf = predizione_podio(s['Rider'], s['Team'], s['Event'], s['Year'], s['Conditions'])
            prob_logreg = predizione_podio_logreg(s['Rider'], s['Team'], s['Event'], s['Year'], s['Conditions'])
            risultati.append({
                'Rider': s['Rider'],
                'Team': s['Team'],
                'Event': s['Event'],
                'Year': s['Year'],
                'Conditions': s['Conditions'],
                'Prob_podio_RF': prob_rf,
                'Prob_podio_LogReg': prob_logreg
            })
            print(f"{s['Rider']} ({s['Team']}, {s['Event']}, {s['Year']}, {s['Conditions']}): RF={prob_rf:.2%}, LogReg={prob_logreg:.2%}")
        except Exception as e:
            print(f"Errore per scenario {s}: {e}")
    # Esporta risultati
    df_risultati = pd.DataFrame(risultati)
    output_dir = 'output_podio_italia_vincitori2020_2025'
    os.makedirs(output_dir, exist_ok=True)
    df_risultati.to_csv(f'{output_dir}/previsioni_scenari.csv', index=False)
    print("\nRisultati esportati in output_podio_italia_vincitori2020_2025/previsioni_scenari.csv")
    # Esempio di utilizzo delle funzioni di scenario
    print("\nEsempio di previsione podio per Bagnaia/Pramac Racing/ita/2020/Clear:")
    prob_rf = predizione_podio('Bagnaia', 'Pramac Racing', 'ita', 2020, 'Clear')
    print(f"Random Forest: Probabilità podio = {prob_rf:.2%}")
    prob_logreg = predizione_podio_logreg('Bagnaia', 'Pramac Racing', 'ita', 2020, 'Clear')
    print(f"Logistic Regression: Probabilità podio = {prob_logreg:.2%}")
    # --- SPLIT DATI E MODELLI ---
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    X_train, X_test, y_train, y_test = train_test_split(X[mask_train], y[mask_train], test_size=0.2, random_state=42, stratify=y[mask_train])
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced')
    }

    # --- VALIDAZIONE INCROCIATA RANDOM FOREST ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("\nCross-validation (5-fold) Random Forest:")
    rf_cv_acc = cross_val_score(models['Random Forest'], X_train, y_train, cv=cv, scoring='accuracy')
    rf_cv_f1 = cross_val_score(models['Random Forest'], X_train, y_train, cv=cv, scoring='f1')
    rf_cv_roc = cross_val_score(models['Random Forest'], X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"Accuracy: {rf_cv_acc.mean():.3f} ± {rf_cv_acc.std():.3f}")
    print(f"F1-score: {rf_cv_f1.mean():.3f} ± {rf_cv_f1.std():.3f}")
    print(f"ROC AUC: {rf_cv_roc.mean():.3f} ± {rf_cv_roc.std():.3f}")

    # --- METRICHE SU TEST SET RANDOM FOREST ---
    models['Random Forest'].fit(X_train, y_train)
    y_proba_rf = models['Random Forest'].predict_proba(X_test)[:,1]
    y_pred_rf = models['Random Forest'].predict(X_test)
    print("\nMetriche Random Forest (test set):")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.3f}")
    print(f"F1-score: {f1_score(y_test, y_pred_rf):.3f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba_rf):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred_rf):.3f}")
    print(f"Recall: {recall_score(y_test, y_pred_rf):.3f}")
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    print(f"Confusion matrix:\n{cm_rf}")
    # Salva confusion matrix
    plt.figure()
    ConfusionMatrixDisplay(cm_rf).plot()
    plt.title('Confusion Matrix - Random Forest')
    plt.savefig(f'{output_dir}/confusion_matrix_rf.png')
    plt.close()
    # ROC curve
    plt.figure()
    RocCurveDisplay.from_estimator(models['Random Forest'], X_test, y_test)
    plt.title('ROC Curve - Random Forest')
    plt.savefig(f'{output_dir}/roc_curve_rf.png')
    plt.close()
    # Precision-Recall curve
    plt.figure()
    PrecisionRecallDisplay.from_estimator(models['Random Forest'], X_test, y_test)
    plt.title('Precision-Recall Curve - Random Forest')
    plt.savefig(f'{output_dir}/pr_curve_rf.png')
    plt.close()

    # --- INTERPRETABILITÀ: FEATURE IMPORTANCE GRAFICA ---
    importances = models['Random Forest'].feature_importances_
    plt.figure(figsize=(8,4))
    plt.bar(features, importances, color='royalblue')
    plt.ylabel('Importanza')
    plt.title('Feature Importance - Random Forest')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance_rf.png')
    plt.close()

    # --- INTERPRETABILITÀ: SHAP (se disponibile) ---
    try:
        import shap
        explainer = shap.TreeExplainer(models['Random Forest'])
        shap_values = explainer.shap_values(X_test)
        plt.figure()
        shap.summary_plot(shap_values, X_test, feature_names=features, show=False)
        plt.savefig(f'{output_dir}/shap_summary_rf.png')
        plt.close()
        print('SHAP plot salvato.')
    except ImportError:
        print('SHAP non installato, interpretabilità avanzata non disponibile.')
    except Exception as e:
        print(f'Errore SHAP: {e}')

    # --- GESTIONE OUTLIER: stampa valori anomali su Position ---
    outlier_pos = df[(df['Position'] < 1) | (df['Position'] > 25)]
    if not outlier_pos.empty:
        print('\nAttenzione: valori anomali su Position trovati:')
        print(outlier_pos[['Rider_normalized','Year','Event','Position']])
    else:
        print('\nNessun outlier su Position.')

    # --- CONFRONTO MODELLI ---
    print("\nConfronto modelli (train/test split 2002-2020):")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"{name}: Accuracy={acc:.3f}, F1-score={f1:.3f}")

    # --- IMPORTANZA DELLE FEATURE DEL MODELLO ---
    print("\nImportanza delle feature nel modello Random Forest:")
    for feat, imp in zip(features, models['Random Forest'].feature_importances_):
        print(f"{feat}: {imp:.3f}")
        # --- GRAFICO AGGREGATO PROBABILITA' PODIO PER PILOTA (SCENARI) ---
        try:
            df_scenari = pd.read_csv('output_podio_italia_vincitori2020_2025/previsioni_scenari.csv')
            df_agg = df_scenari.groupby('Rider')[['Prob_podio_RF','Prob_podio_LogReg']].mean().reset_index()
            plt.figure(figsize=(8,5))
            bar_width = 0.35
            x = np.arange(len(df_agg['Rider']))
            bars_rf = plt.bar(x - bar_width/2, df_agg['Prob_podio_RF'], bar_width, label='Random Forest', color='royalblue')
            bars_lr = plt.bar(x + bar_width/2, df_agg['Prob_podio_LogReg'], bar_width, label='Logistic Regression', color='orange')
            plt.xticks(x, df_agg['Rider'], rotation=45)
            plt.ylabel('Probabilità media podio')
            plt.title('Confronto probabilità podio media per pilota (scenari)')
            plt.legend()
            # Annota valori percentuali sopra le barre
            max_val = max(df_agg['Prob_podio_RF'].max(), df_agg['Prob_podio_LogReg'].max())
            plt.ylim(0, min(1.0, max_val + 0.15))
            for rect in list(bars_rf) + list(bars_lr):
                height = rect.get_height()
                plt.text(
                    rect.get_x() + rect.get_width()/2.0,
                    height + 0.01,
                    f"{height*100:.1f}%",
                    ha='center', va='bottom', fontsize=9, clip_on=False
                )
            plt.tight_layout()
            plt.savefig(f'{output_dir}/barplot_probabilita_podio_aggregato.png')
            plt.close()
            print("Barplot aggregato salvato in output_podio_italia_vincitori2020_2025/barplot_probabilita_podio_aggregato.png")
        except Exception as e:
            print(f"Errore generazione barplot aggregato: {e}")