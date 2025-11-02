"""
Validazione incrociata stratificata per i modelli di classificazione (podio) e regressione (tempo giro)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import make_scorer, f1_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

file_class = '../classifica/motogp_results_cleaned_final_normalized.csv'
# Percorso assoluto per compatibilità
file_class = '../classifica/motogp_results_cleaned_final_normalized.csv'.replace('../classifica/', 'classifica/')
df = pd.read_csv(file_class)
df = df[pd.to_numeric(df['Position'], errors='coerce').notnull()]
df['Position'] = df['Position'].astype(int)
df['Podio'] = (df['Position'] <= 3).astype(int)
features = ['Rider_normalized', 'Team', 'Grand Prix', 'Year']
X = df[features].copy()
y = df['Podio']
for col in ['Rider_normalized', 'Team', 'Grand Prix']:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
X = X.fillna(-1)

models_class = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
    'KNN (k=7)': KNeighborsClassifier(n_neighbors=7),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.08, random_state=42)
}

print('--- Validazione incrociata classificazione (podio) ---')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_means = []
f1_stds = []
labels_class = []
for name, model in models_class.items():
    # SMOTE solo sul training di ogni fold
    f1_scores = []
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        sm = SMOTE(random_state=42)
        X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        f1 = f1_score(y_te, y_pred)
        f1_scores.append(f1)
    mean = np.mean(f1_scores)
    std = np.std(f1_scores)
    f1_means.append(mean)
    f1_stds.append(std)
    labels_class.append(name)
    print(f"{name}: F1-score medio = {mean:.3f} (+/- {std:.3f})")

# Barplot F1-score medi con deviazione standard
plt.figure(figsize=(8,5))
bars = plt.bar(labels_class, f1_means, yerr=f1_stds, capsize=6, color='#377eb8')
plt.ylabel('F1-score medio (5-fold CV)')
plt.ylim(0,1)
plt.title('Cross validation - F1-score medio classificazione podio')
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.01, f"{f1_means[i]:.2f}", ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig('modello/barplot_cv_f1score_podio.png', dpi=150)
plt.close()

file_reg = '../classifica/motogp_results_cleaned_final_normalized.csv'.replace('../classifica/', 'classifica/')
df = pd.read_csv(file_reg)
df = df[pd.to_numeric(df['Time/Gap'], errors='coerce').notnull()]
df['Time/Gap'] = df['Time/Gap'].astype(float)
features_reg = ['Rider_normalized', 'Team', 'Grand Prix', 'Year']
Xr = df[features_reg].copy()
yr = df['Time/Gap']
for col in ['Rider_normalized', 'Team', 'Grand Prix']:
    Xr[col] = LabelEncoder().fit_transform(Xr[col].astype(str))
Xr = Xr.fillna(-1)

models_reg = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Linear Regression': LinearRegression(),
    'SVR': SVR(),
    'KNN (k=7)': KNeighborsRegressor(n_neighbors=7),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.08, random_state=42)
}

print('\n--- Validazione incrociata regressione (tempo giro) ---')
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mse_means = []
mse_stds = []
labels_reg = []
for name, model in models_reg.items():
    mse_scores = -cross_val_score(model, Xr, yr, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
    mean = np.mean(mse_scores)
    std = np.std(mse_scores)
    mse_means.append(mean)
    mse_stds.append(std)
    labels_reg.append(name)
    print(f"{name}: MSE medio = {mean:.3f} (+/- {std:.3f})")

# Barplot MSE medi con deviazione standard
plt.figure(figsize=(8,5))
bars = plt.bar(labels_reg, mse_means, yerr=mse_stds, capsize=6, color='#e41a1c')
plt.ylabel('MSE medio (5-fold CV)')
# Titolo richiesto
plt.title('Cross validation - MSE medio regressione tempo sul giro')

# Imposta un limite superiore per lasciare spazio alle etichette sopra le barre+errore
max_val = max([m + s for m, s in zip(mse_means, mse_stds)]) if mse_stds else max(mse_means)
plt.ylim(0, max_val * 1.10)

# Etichette posizionate sopra le barre considerando la barra d'errore, con box bianco per leggibilità
for i, bar in enumerate(bars):
    height = bar.get_height()
    err = mse_stds[i] if i < len(mse_stds) else 0
    y = height + err + max_val * 0.02
    plt.text(
        bar.get_x() + bar.get_width()/2,
        y,
        f"{mse_means[i]:.1f}",
        ha='center', va='bottom', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85)
    )

plt.tight_layout()
plt.savefig('modello/barplot_cv_mse_tempo_giro.png', dpi=150)
plt.close()
