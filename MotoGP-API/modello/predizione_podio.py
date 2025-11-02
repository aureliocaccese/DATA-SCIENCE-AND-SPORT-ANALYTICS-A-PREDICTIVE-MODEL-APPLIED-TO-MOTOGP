
# Previsione probabilità di podio (classificazione)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_curve, auc, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import learning_curve, validation_curve


# Carica il dataset e filtra solo piloti classificati (Position numerica)
file_path = '/Users/aurelio/Desktop/TESI/MotoGP-API/classifica/motogp_results_cleaned_final_normalized.csv'
df = pd.read_csv(file_path)
df = df[pd.to_numeric(df['Position'], errors='coerce').notnull()]
df['Position'] = df['Position'].astype(int)

# Target: podio (1 se posizione <=3, 0 altrimenti)
df['Podio'] = (df['Position'] <= 3).astype(int)
y = df['Podio']

# Barplot distribuzione classi (target)
import matplotlib.pyplot as plt
plt.figure(figsize=(4,4))
df['Podio'].value_counts().sort_index().plot(kind='bar', color=['#377eb8','#e41a1c'])
plt.xticks([0,1], ['No Podio','Podio'], rotation=0)
plt.ylabel('Numero campioni')
plt.title('Distribuzione classi target (Podio)')
plt.tight_layout()
plt.savefig('modello/barplot_distribuzione_classi.png', dpi=150)
plt.close()
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE


# Carica il dataset e filtra solo piloti classificati (Position numerica)
file_path = '/Users/aurelio/Desktop/TESI/MotoGP-API/classifica/motogp_results_cleaned_final_normalized.csv'
df = pd.read_csv(file_path)
df = df[pd.to_numeric(df['Position'], errors='coerce').notnull()]
df['Position'] = df['Position'].astype(int)

# Feature engineering basilare
# Consideriamo solo alcune feature di esempio (puoi aggiungere altre variabili!)
features = ['Rider_normalized', 'Team', 'Grand Prix', 'Year']
X = df[features].copy()

# Target: podio (1 se posizione <=3, 0 altrimenti)
df['Podio'] = (df['Position'] <= 3).astype(int)
y = df['Podio']

# Encoding variabili categoriche
for col in ['Rider_normalized', 'Team', 'Grand Prix']:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))


# Gestione NaN
X = X.fillna(-1)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Oversampling SMOTE
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# Confronto tra più algoritmi di classificazione

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
    'KNN (k=7)': KNeighborsClassifier(n_neighbors=7),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.08, random_state=42)
}

results = {}
roc_curves = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"{name}: Accuracy = {acc:.3f}, F1-score = {f1:.3f}")
    # ROC curve
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:,1]
    else:
        y_score = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    roc_curves[name] = (fpr, tpr, roc_auc)
    results[name] = {'accuracy': acc, 'f1': f1, 'y_pred': y_pred, 'roc_auc': roc_auc}

# Grafico comparativo delle metriche con valori sulle barre ed evidenziazione miglior modello
labels = list(results.keys())
accuracy = [results[k]['accuracy'] for k in labels]
f1 = [results[k]['f1'] for k in labels]
best_idx = f1.index(max(f1))

# Grafico Accuracy separato
plt.figure(figsize=(8,5))
bars_acc = plt.bar(labels, accuracy, color=['gold' if i==best_idx else 'royalblue' for i in range(len(labels))])
plt.ylim(0, 1.05)
plt.title('Accuracy - Confronto algoritmi classificazione podio')
plt.ylabel('Score')
plt.xticks(range(len(labels)), labels, rotation=20)
for i, bar in enumerate(bars_acc):
    y_pos = min(1.0, bar.get_height() + 0.03)
    plt.text(bar.get_x() + bar.get_width()/2, y_pos, f"{accuracy[i]:.2f}", ha='center', va='bottom', fontsize=11,
             bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))
plt.tight_layout()
plt.savefig('modello/confronto_algoritmi_classificazione_accuracy.png', dpi=150)
plt.close()

# Grafico F1 separato
plt.figure(figsize=(8,5))
bars_f1 = plt.bar(labels, f1, color=['gold' if i==best_idx else 'orange' for i in range(len(labels))])
plt.ylim(0, 1.05)
plt.title('F1-score - Confronto algoritmi classificazione podio')
plt.ylabel('Score')
plt.xticks(range(len(labels)), labels, rotation=20)
for i, bar in enumerate(bars_f1):
    y_pos = min(1.0, bar.get_height() + 0.03)
    plt.text(bar.get_x() + bar.get_width()/2, y_pos, f"{f1[i]:.2f}", ha='center', va='bottom', fontsize=11,
             bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))
plt.tight_layout()
plt.savefig('modello/confronto_algoritmi_classificazione_f1.png', dpi=150)
plt.close()

# Mantieni anche il grafico combinato per retro-compatibilità
fig, axs = plt.subplots(1, 2, figsize=(13,5))
bars1 = axs[0].bar(labels, accuracy, color=['gold' if i==best_idx else 'royalblue' for i in range(len(labels))])
axs[0].set_ylim(0,1)
axs[0].set_title('Accuracy')
axs[0].set_ylabel('Score')
for i, bar in enumerate(bars1):
    axs[0].text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.01, f"{accuracy[i]:.2f}", ha='center', va='bottom', fontsize=10)
bars2 = axs[1].bar(labels, f1, color=['gold' if i==best_idx else 'orange' for i in range(len(labels))])
axs[1].set_ylim(0,1)
axs[1].set_title('F1-score')
for i, bar in enumerate(bars2):
    axs[1].text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.01, f"{f1[i]:.2f}", ha='center', va='bottom', fontsize=10)
for ax in axs:
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20)
plt.suptitle('Confronto algoritmi classificazione podio')
plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig('modello/confronto_algoritmi_classificazione.png', dpi=150)
plt.close()

# Salva tabella metriche
df_metrics = pd.DataFrame({
    'Algoritmo': labels,
    'Accuracy': accuracy,
    'F1-score': f1,
    'ROC AUC': [results[k]['roc_auc'] for k in labels]
})
df_metrics.to_csv('modello/metriche_classificazione.csv', index=False)

# ROC curve comparativa
plt.figure(figsize=(8,7))
for name, (fpr, tpr, roc_auc) in roc_curves.items():
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC={roc_auc:.2f})')
plt.plot([0,1],[0,1],'k--',lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Confronto algoritmi')
plt.legend()
plt.tight_layout()
plt.savefig('modello/roc_curve_confronto_algoritmi.png', dpi=150)
plt.close()



# 1. Matrice di confusione per ogni algoritmo
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
for i, name in enumerate(labels):
    y_pred = results[name]['y_pred']
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Podio','Podio'])
    fig, ax = plt.subplots(figsize=(4,4))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title(f'Confusion Matrix - {name}')
    plt.tight_layout()
    fname = name.lower().replace(' ','_').replace('(','').replace(')','').replace('/','_')
    plt.savefig(f'modello/confusion_matrix_{fname}.png', dpi=150)
    plt.close()

# 2. ROC curve comparativa (già presente)
# 3. Barplot metriche (già presente)

# 5. Precision-Recall curve comparativa
from sklearn.metrics import precision_recall_curve, average_precision_score
plt.figure(figsize=(8,7))
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:,1]
    else:
        y_score = model.decision_function(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    ap = average_precision_score(y_test, y_score)
    plt.plot(recall, precision, lw=2, label=f'{name} (AP={ap:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Confronto algoritmi')
plt.legend()
plt.tight_layout()
plt.savefig('modello/precision_recall_curve_confronto_algoritmi.png', dpi=150)
plt.close()

# 6. Feature importance per modelli ad albero
tree_models = ['Random Forest', 'Gradient Boosting']
for name in tree_models:
    if name in results:
        model = models[name]
        importances = model.feature_importances_
        plt.figure(figsize=(7,4))
        plt.bar(features, importances, color='#377eb8')
        plt.ylabel('Importanza')
        plt.title(f'Feature Importance - {name}')
        plt.tight_layout()
        fname = name.lower().replace(' ','_')
        plt.savefig(f'modello/feature_importance_{fname}.png', dpi=150)
        plt.close()

# 7. Interpretabilità SHAP per Random Forest
import shap
if 'Random Forest' in models:
    explainer = shap.TreeExplainer(models['Random Forest'])
    shap_values = explainer.shap_values(X_test)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=features, show=False)
    plt.tight_layout()
    plt.savefig('modello/shap_summary_random_forest.png', dpi=150, bbox_inches='tight')
    plt.close()

# 4. Barplot percentuali TP, FP, TN, FN per ogni algoritmo
import numpy as np
for i, name in enumerate(labels):
    y_pred = results[name]['y_pred']
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    perc = np.array([tn, fp, fn, tp]) / total * 100
    plt.figure(figsize=(5,4))
    plt.bar(['TN','FP','FN','TP'], perc, color=['#377eb8','#e41a1c','#ff7f00','#4daf4a'])
    plt.ylim(0,100)
    plt.ylabel('Percentuale (%)')
    plt.title(f'Percentuali TN/FP/FN/TP - {name}')
    plt.tight_layout()
    fname = name.lower().replace(' ','_').replace('(','').replace(')','').replace('/','_')
    plt.savefig(f'modello/barplot_confusion_percent_{fname}.png', dpi=150)
    plt.close()


# Scatterplot reale vs predetto per tutti gli algoritmi (singoli e confronto)
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#984ea3']
marker = 'o'
plt.figure(figsize=(8,8))
for i, name in enumerate(labels):
    y_pred = results[name]['y_pred']
    plt.scatter(y_test, y_pred, alpha=0.25, label=name, color=colors[i%len(colors)], marker=marker, s=36, edgecolor='black', linewidth=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfetto')
plt.xlabel('Podio reale')
plt.ylabel('Podio predetto')
plt.title('Confronto scatter reale vs predetto - Tutti gli algoritmi')
plt.xlim(-0.2,1.2)
plt.ylim(-0.2,1.2)
plt.legend()
plt.tight_layout()
plt.savefig('modello/scatter_podio_reale_vs_predetto_confronto.png', dpi=150)
plt.close()

for i, name in enumerate(labels):
    y_pred = results[name]['y_pred']
    plt.figure(figsize=(7,7))
    plt.scatter(y_test, y_pred, alpha=0.3, color=colors[i%len(colors)], marker=marker, s=36, edgecolor='black', linewidth=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Podio reale')
    plt.ylabel('Podio predetto')
    plt.title(f'{name} - Podio reale vs predetto')
    plt.xlim(-0.2,1.2)
    plt.ylim(-0.2,1.2)
    plt.tight_layout()
    fname = name.lower().replace(' ','_').replace('(','').replace(')','').replace('/','_')
    plt.savefig(f'modello/scatter_podio_reale_vs_predetto_{fname}.png', dpi=150)
    plt.close()

# 8. Scatterplot reale vs predetto per tutti gli algoritmi (singoli e confronto)
# Confusion matrix per il migliore (miglior F1)
best_name = labels[best_idx]
best_pred = results[best_name]['y_pred']
cm = confusion_matrix(y_test, best_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Podio','Podio'])
fig, ax = plt.subplots(figsize=(5,5))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
plt.title(f'Confusion Matrix - {best_name}')
plt.tight_layout()
plt.savefig('modello/confusion_matrix_podio.png', dpi=150)
plt.close()

# --- 9. Learning curve per il miglior modello
train_sizes, train_scores, test_scores = learning_curve(models[best_name], X, y, cv=5, scoring='f1', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 8), shuffle=True, random_state=42)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.figure(figsize=(7,5))
plt.plot(train_sizes, train_mean, 'o-', color='royalblue', label='Training F1')
plt.plot(train_sizes, test_mean, 'o-', color='darkorange', label='Validation F1')
plt.xlabel('Numero campioni di training')
plt.ylabel('F1-score')
plt.title(f'Learning Curve - {best_name}')
plt.legend()
plt.tight_layout()
plt.savefig('modello/learning_curve_best_model.png', dpi=150)
plt.close()

# --- 10. Validation curve per iperparametro chiave (Random Forest: n_estimators)
if 'Random Forest' in models:
    param_range = [10, 30, 50, 70, 100, 150, 200, 300]
    train_scores, test_scores = validation_curve(
        RandomForestClassifier(random_state=42, class_weight='balanced'),
        X, y, param_name="n_estimators", param_range=param_range,
        cv=5, scoring="f1", n_jobs=-1)
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    plt.figure(figsize=(7,5))
    plt.plot(param_range, train_mean, 'o-', color='royalblue', label='Training F1')
    plt.plot(param_range, test_mean, 'o-', color='darkorange', label='Validation F1')
    plt.xlabel('n_estimators (Random Forest)')
    plt.ylabel('F1-score')
    plt.title('Validation Curve - Random Forest')
    plt.legend()
    plt.tight_layout()
    plt.savefig('modello/validation_curve_random_forest.png', dpi=150)
    plt.close()

# --- 11. Barplot precision, recall, specificity per ogni modello
precision = []
recall = []
specificity = []
for name in labels:
    y_pred = results[name]['y_pred']
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision.append(tp/(tp+fp) if (tp+fp)>0 else 0)
    recall.append(tp/(tp+fn) if (tp+fn)>0 else 0)
    specificity.append(tn/(tn+fp) if (tn+fp)>0 else 0)
bar_width = 0.25
index = np.arange(len(labels))
plt.figure(figsize=(10,5))
plt.bar(index, precision, bar_width, label='Precision', color='#377eb8')
plt.bar(index+bar_width, recall, bar_width, label='Recall', color='#e41a1c')
plt.bar(index+2*bar_width, specificity, bar_width, label='Specificity', color='#4daf4a')
plt.xticks(index+bar_width, labels, rotation=20)
plt.ylim(0,1)
plt.ylabel('Score')
plt.title('Precision, Recall, Specificity per modello')
plt.legend()
plt.tight_layout()
plt.savefig('modello/barplot_precision_recall_specificity.png', dpi=150)
plt.close()
