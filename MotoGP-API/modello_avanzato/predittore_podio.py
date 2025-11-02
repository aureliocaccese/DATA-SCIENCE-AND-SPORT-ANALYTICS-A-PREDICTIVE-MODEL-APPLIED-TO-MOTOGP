import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

class PredittorePodioMotoGP:
    """
    Predittore di podio MotoGP con features avanzate
    """
    
    def __init__(self):
        self.modelo = None
        self.label_encoders = {}
        self.feature_columns = [
            'Rider_normalized', 'Team', 'Event', 'Year', 'Conditions',
            'Avg_Position_Last3', 'Podiums_Last5', 'Points_Streak',
            'length_km', 'width_m', 'right corners', 'left corners',
            'Race_Number_Season', 'Rider_Experience',
            'Season_Points_So_Far', 'Championship_Position',
            'Team_Avg_Position', 'Difficult_Conditions',
            'Rider_Circuit_Avg'
        ]
        self.categorical_features = ['Rider_normalized', 'Team', 'Event', 'Conditions']
        
    def addestra_modello(self, dataset_path='classifica/motogp_results_expanded_features.csv'):
        """
        Addestra il modello sui dati storici
        """
        print("🏁 Addestrando modello predittivo MotoGP...")
        
        # Carica dati
        df = pd.read_csv(dataset_path)
        df = df[pd.to_numeric(df['Position'], errors='coerce').notnull()]
        df['Position'] = df['Position'].astype(int)
        df['Podio'] = (df['Position'] <= 3).astype(int)
        
        # Prepara dataset pulito
        df_clean = df.dropna(subset=self.feature_columns + ['Podio'])
        
        # Split temporale (training fino al 2020)
        mask_train = (df_clean['Year'] >= 2002) & (df_clean['Year'] <= 2020)
        df_train = df_clean[mask_train]
        
        # Prepara features
        X_train = df_train[self.feature_columns].copy()
        y_train = df_train['Podio']
        
        # Encoding variabili categoriche
        for col in self.categorical_features:
            self.label_encoders[col] = LabelEncoder()
            X_train[col] = self.label_encoders[col].fit_transform(X_train[col].astype(str))
        
        X_train = X_train.fillna(-1)
        
        # Addestra modello
        self.modelo = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        )
        
        self.modelo.fit(X_train, y_train)
        
        print(f"✅ Modello addestrato su {len(X_train):,} campioni")
        print(f"📊 Feature Importance Top 5:")
        
        # Mostra feature importance
        importance_df = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': self.modelo.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        for i, row in importance_df.head().iterrows():
            print(f"   {row['Feature']}: {row['Importance']:.3f}")
        
        return self
    
    def predici_podio(self, rider, team, event, year=2024, conditions='Dry', 
                     stats_storiche=None, stats_circuito=None):
        """
        Predice la probabilità di podio per un pilota specifico
        
        Args:
            rider: Nome pilota (es. "Francesco Bagnaia")
            team: Team (es. "Ducati Lenovo Team")
            event: Evento/GP (es. "Catalunya")
            year: Anno (default 2024)
            conditions: Condizioni meteo ('Dry', 'Wet', 'Mixed')
            stats_storiche: Dict con statistiche storiche del pilota
            stats_circuito: Dict con caratteristiche del circuito
        """
        
        if self.modelo is None:
            raise ValueError("Modello non addestrato! Usa addestra_modello() prima.")
        
        # Crea input base
        input_data = {
            'Rider_normalized': rider,
            'Team': team,
            'Event': event,
            'Year': year,
            'Conditions': conditions
        }
        
        # Inizializza tutti i valori di default per tutte le features
        defaults = {
            'Avg_Position_Last3': 6.0,
            'Podiums_Last5': 1,
            'Points_Streak': 3,
            'length_km': 4.2,
            'width_m': 12.0,
            'right corners': 6,
            'left corners': 7,
            'Race_Number_Season': 10,
            'Rider_Experience': max(1, year - 2015),
            'Season_Points_So_Far': 50,
            'Championship_Position': 8,
            'Team_Avg_Position': 5.5,
            'Difficult_Conditions': 1 if conditions != 'Dry' else 0,
            'Rider_Circuit_Avg': 7.5
        }
        
        # Aggiungi tutti i defaults
        input_data.update(defaults)
        
        # Sovrascrivi con statistiche storiche se fornite
        if stats_storiche:
            for key, value in stats_storiche.items():
                if key in input_data:
                    input_data[key] = value
        
        # Sovrascrivi con caratteristiche circuito se fornite
        if stats_circuito:
            for key, value in stats_circuito.items():
                if key in input_data:
                    input_data[key] = value
        
        # Converti in DataFrame
        df_input = pd.DataFrame([input_data])
        
        # Prepara per predizione
        X_input = df_input[self.feature_columns].copy()
        
        # Encoding
        for col in self.categorical_features:
            if col in self.label_encoders:
                try:
                    X_input[col] = self.label_encoders[col].transform(X_input[col].astype(str))
                except ValueError:
                    # Valore non visto in training, usa il più frequente
                    X_input[col] = 0
        
        X_input = X_input.fillna(-1)
        
        # Predizione
        prob_podio = self.modelo.predict_proba(X_input)[0][1]
        predizione_binaria = self.modelo.predict(X_input)[0]
        
        return {
            'rider': rider,
            'team': team,
            'event': event,
            'probabilita_podio': prob_podio,
            'predizione_podio': bool(predizione_binaria),
            'confidenza': 'Alta' if prob_podio > 0.7 or prob_podio < 0.3 else 'Media'
        }
    
    def analizza_griglia(self, piloti_info, event, year=2024, conditions='Dry'):
        """
        Analizza una griglia completa di piloti per un GP
        
        Args:
            piloti_info: Lista di dict con info piloti
            event: Nome dell'evento
            year: Anno
            conditions: Condizioni meteo
        """
        
        risultati = []
        
        print(f"🏁 Analizzando griglia per {event} {year} (Condizioni: {conditions})")
        print("-" * 70)
        
        for i, pilota in enumerate(piloti_info):
            pred = self.predici_podio(
                rider=pilota['rider'],
                team=pilota['team'],
                event=event,
                year=year,
                conditions=conditions,
                stats_storiche=pilota.get('stats_storiche'),
                stats_circuito=pilota.get('stats_circuito')
            )
            
            risultati.append(pred)
            
            # Mostra risultato
            emoji_podio = "🏆" if pred['predizione_podio'] else "❌"
            print(f"{i+1:2d}. {emoji_podio} {pred['rider']:<20} "
                  f"({pred['team']:<20}) - "
                  f"Prob: {pred['probabilita_podio']:.1%} "
                  f"({pred['confidenza']})")
        
        # Ordina per probabilità
        risultati_ordinati = sorted(risultati, key=lambda x: x['probabilita_podio'], reverse=True)
        
        print("\n🥇 TOP 3 PREDIZIONI PODIO:")
        print("-" * 50)
        for i, pred in enumerate(risultati_ordinati[:3]):
            medaglia = ["🥇", "🥈", "🥉"][i]
            print(f"{medaglia} {pred['rider']} - {pred['probabilita_podio']:.1%}")
        
        return risultati_ordinati
    
    def salva_modello(self, percorso='modello_avanzato/modello_podio_motogp.pkl'):
        """
        Salva il modello addestrato
        """
        model_data = {
            'modelo': self.modelo,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'categorical_features': self.categorical_features,
            'addestrato_il': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, percorso)
        print(f"💾 Modello salvato in: {percorso}")
    
    def carica_modello(self, percorso='modello_avanzato/modello_podio_motogp.pkl'):
        """
        Carica un modello precedentemente addestrato
        """
        model_data = joblib.load(percorso)
        
        self.modelo = model_data['modelo']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.categorical_features = model_data['categorical_features']
        
        print(f"📂 Modello caricato da: {percorso}")
        print(f"🕐 Addestrato il: {model_data.get('addestrato_il', 'Data sconosciuta')}")

def esempio_utilizzo():
    """
    Esempio di utilizzo del predittore
    """
    
    print("🏁 ESEMPIO UTILIZZO PREDITTORE PODIO MOTOGP")
    print("=" * 60)
    
    # Crea e addestra predittore
    predittore = PredittorePodioMotoGP()
    predittore.addestra_modello()
    
    # Esempio predizione singola
    print("\n🔮 PREDIZIONE SINGOLA - Francesco Bagnaia")
    pred_bagnaia = predittore.predici_podio(
        rider="Francesco Bagnaia",
        team="Ducati Lenovo Team",
        event="Catalunya",
        year=2024,
        conditions="Dry",
        stats_storiche={
            'Avg_Position_Last3': 2.0,
            'Podiums_Last5': 4,
            'Points_Streak': 8,
            'Rider_Experience': 7,
            'Season_Points_So_Far': 120,
            'Championship_Position': 1,
            'Rider_Circuit_Avg': 3.2
        }
    )
    
    print(f"Pilota: {pred_bagnaia['rider']}")
    print(f"Probabilità podio: {pred_bagnaia['probabilita_podio']:.1%}")
    print(f"Predizione: {'PODIO' if pred_bagnaia['predizione_podio'] else 'NO PODIO'}")
    print(f"Confidenza: {pred_bagnaia['confidenza']}")
    
    # Esempio griglia completa
    print("\n🏁 ANALISI GRIGLIA COMPLETA - GP Catalunya 2024")
    
    griglia_catalunya = [
        {
            'rider': 'Francesco Bagnaia',
            'team': 'Ducati Lenovo Team',
            'stats_storiche': {'Avg_Position_Last3': 2.0, 'Podiums_Last5': 4, 'Points_Streak': 8}
        },
        {
            'rider': 'Jorge Martin',
            'team': 'Prima Pramac Racing',
            'stats_storiche': {'Avg_Position_Last3': 3.5, 'Podiums_Last5': 3, 'Points_Streak': 6}
        },
        {
            'rider': 'Marc Marquez',
            'team': 'Gresini Racing MotoGP',
            'stats_storiche': {'Avg_Position_Last3': 4.0, 'Podiums_Last5': 2, 'Points_Streak': 4}
        },
        {
            'rider': 'Enea Bastianini',
            'team': 'Ducati Lenovo Team',
            'stats_storiche': {'Avg_Position_Last3': 5.0, 'Podiums_Last5': 2, 'Points_Streak': 3}
        },
        {
            'rider': 'Pedro Acosta',
            'team': 'Red Bull GASGAS Tech3',
            'stats_storiche': {'Avg_Position_Last3': 8.0, 'Podiums_Last5': 1, 'Points_Streak': 2}
        }
    ]
    
    risultati = predittore.analizza_griglia(griglia_catalunya, "Catalunya", 2024, "Dry")
    
    # Salva modello
    predittore.salva_modello()
    
    print("\n✅ ESEMPIO COMPLETATO!")

if __name__ == "__main__":
    esempio_utilizzo()
