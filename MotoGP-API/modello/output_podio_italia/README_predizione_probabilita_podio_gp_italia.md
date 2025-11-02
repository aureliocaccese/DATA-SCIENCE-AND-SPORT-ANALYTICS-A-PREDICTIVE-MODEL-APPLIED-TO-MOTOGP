# predizione_probabilita_podio_gp_italia.py

Script per la predizione e visualizzazione della probabilità di podio al GP Italia per i piloti MotoGP, con grafici ottimizzati per la tesi.

## Funzionalità
- Addestramento Random Forest su dati storici
- Predizione podio per scenari reali e futuri (2010-2030)
- Esportazione risultati in CSV
- Grafici migliorati: colori armoniosi, etichette leggibili, legenda chiara, valori sulle barre, griglia leggera
- Output organizzati in una sola cartella

## Utilizzo
Esegui:
```
python modello/output_podio_italia/predizione_probabilita_podio_gp_italia.py
```

## Output
- CSV con probabilità podio storiche e predette
- Grafici PNG ad alta risoluzione

## Dipendenze
- pandas
- scikit-learn
- matplotlib
- seaborn

## Note
Personalizza top_n piloti, periodo e parametri grafici secondo le esigenze della tesi.
