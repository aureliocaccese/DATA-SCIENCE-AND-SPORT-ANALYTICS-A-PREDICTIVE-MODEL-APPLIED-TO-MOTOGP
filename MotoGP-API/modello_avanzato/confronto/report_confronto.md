
# REPORT CONFRONTO MODELLI BASE vs AVANZATO

## 📊 RISULTATI PERFORMANCE

### Modello Base (5 features):
- ROC AUC: 0.782 ± 0.094
- Features: Rider, Team, Event, Year, Conditions
- Campioni: 4,041

### Modello Avanzato (19 features):
- ROC AUC: 0.904 ± 0.029
- Features: 19 (incluse performance storiche, caratteristiche circuito, etc.)
- Campioni: 774

## 🚀 MIGLIORAMENTO

- **Miglioramento assoluto**: +0.122 ROC AUC
- **Miglioramento percentuale**: +15.5%

## 📈 IMPORTANZA CATEGORIE FEATURES

- **Performance Storiche**: 0.374 (37.4%)
- **Features Competitive**: 0.305 (30.5%)
- **Base Features**: 0.136 (13.6%)
- **Altri**: 0.076 (7.6%)
- **Features Temporali**: 0.069 (6.9%)
- **Caratteristiche Circuito**: 0.041 (4.1%)


## 💡 CONCLUSIONI

1. **Performance**: Il modello avanzato mostra un miglioramento significativo di 15.5%
2. **Features più importanti**: Le performance storiche dominano l'importanza
3. **Robustezza**: Entrambi i modelli mostrano bassa varianza nella cross-validation
4. **Raccomandazione**: Utilizzare il modello avanzato per predizioni più accurate

## 📁 FILE GENERATI

- confronto_modelli.csv: Risultati numerici
- confronto_modelli_visualizzazioni.png: Grafici comparativi
- report_confronto.md: Questo report

---
Generato automaticamente il: 2025-09-13 18:01:55
