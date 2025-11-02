# Regressione Gap – Uso rapido

Questo script (`modello/predizione_tempo_giro.py`) addestra più modelli di regressione per il gap rispetto al vincitore e genera grafici/metriche.

Output principali:
- `modello/confronto_algoritmi_regressione_*.png`: barre MAE/RMSE.
- `modello/metriche_regressione.csv`: tabella con MAE e RMSE.
- `modello/scatter_*.png`: scatter reale vs predetto (singoli e multipli modelli).
- `modello/istogramma_errori_gap*.png`: istogramma avanzato per il modello migliore.
- `modello/errori/*.png`: stesso istogramma avanzato per OGNI modello (con KDE, media/mediana e riquadro metriche) + versioni semplici per residui/assoluti/%.

## Cambiare le variabili (feature)

Imposta `MOTOGP_FEATURES` come lista separata da virgole. Usa virgolette quando ci sono spazi nei nomi colonne.

Esempi (zsh):

```zsh
# Usa un sottoinsieme di variabili
MOTOGP_FEATURES="Rider_normalized,Team,Grand Prix,Year_int,Humidity_num,GroundTemp_num" \
python modello/predizione_tempo_giro.py

# Usa anche TrackConditions_cat
MOTOGP_FEATURES="Rider_normalized,Team,Grand Prix,Year_int,TrackConditions_cat,Humidity_num,GroundTemp_num" \
python modello/predizione_tempo_giro.py
```

Se non impostata, viene usato il set di default:
`Rider_normalized, Team, Grand Prix, Year_int, TrackConditions_cat, Humidity_num, GroundTemp_num`.

## Selezionare i modelli

Imposta `MOTOGP_MODELS` con le chiavi:
- `rf` Random Forest
- `lr` Linear Regression
- `svr` SVR
- `knn` KNN (k=7)
- `gbr` Gradient Boosting
- `dt` Decision Tree

Esempi:

```zsh
# Solo Random Forest e SVR
MOTOGP_MODELS="rf,svr" python modello/predizione_tempo_giro.py

# Tutti i modelli inclusi Decision Tree
MOTOGP_MODELS="rf,lr,svr,knn,gbr,dt" python modello/predizione_tempo_giro.py
```

I grafici per-modello vengono salvati in `modello/errori/` con nomi tipo:
`istogramma_errori_gap_random_forest.png`, `istogramma_errori_gap_svr.png`, ecc.
