# README - predizione_podio_vincitori2020_2025.py

Questo script esegue la predizione della probabilità di podio per piloti MotoGP su scenari specifici (pilota, team, pista, anno, meteo), confrontando Random Forest e Logistic Regression.

## Funzionalità
- Addestramento modelli su dati storici
- Predizione podio per scenari personalizzati
- Esportazione risultati e grafici (metriche, feature importance, SHAP)
- Organizzazione output in una cartella dedicata

## Utilizzo
Esegui lo script:
```
python modello/predizione_podio_vincitori2020_2025.py
```

## Output
- CSV con risultati scenari
- Grafici metriche e interpretabilità

## Dipendenze
- pandas
- scikit-learn
- matplotlib
- seaborn
- shap (opzionale)

## Note
Assicurati che i valori di pilota, team, pista e meteo siano presenti nel dataset.
