# Struttura cartella `modello/`

Questa cartella contiene tutti i risultati, grafici e metriche prodotti dai modelli di classificazione e regressione per l'analisi MotoGP.

## Sottocartelle

- **confusion_matrix/**: Matrici di confusione per ogni algoritmo (classificazione podio)
- **feature_importance/**: Grafici di importanza delle feature e interpretabilità SHAP
- **metriche/**: Barplot metriche, percentuali TN/FP/FN/TP, file CSV con metriche
- **risultati/**: File CSV con predizioni reali e stimate, confronti reali/predetti
- **roc_pr/**: Curve ROC e Precision-Recall comparative
- **scatter/**: Scatterplot reale vs predetto (solo per regressione)

## Utilizzo

- Tutti i grafici sono in formato PNG, pronti per essere inseriti in tesi o presentazioni.
- I file CSV contengono metriche e confronti utili per analisi quantitative e tabelle.
- Le sottocartelle aiutano a trovare rapidamente il tipo di risultato desiderato.

## Aggiornamento

Per riordinare nuovamente la cartella dopo nuove esecuzioni degli script, lancia:

```
python3 modello/ordina_modello.py
```

## Note
- I file possono essere rinominati o spostati liberamente per esigenze di presentazione.
- Per domande su come interpretare i risultati, consulta la documentazione o chiedi supporto.
