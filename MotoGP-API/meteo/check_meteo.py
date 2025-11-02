import pandas as pd

# Carica il dataset
file_path = "/Users/aurelio/Desktop/TESI/MotoGP-API/motogp_weather_data.csv"
df = pd.read_csv(file_path)

# Elenco degli eventi attesi
events = ['hun', 'aut', 'cze', 'ger', 'ned', 'ita', 'ara', 'gbr', 'fra', 'esp', 'qat', 'usa', 'arg', 'tha']

# 1. Verifica la presenza di tutte le gare per ogni anno
missing_events = []
for year in range(2002, 2026):  # Dal 2002 al 2025
    year_events = df[df['Year'] == year]['Event'].unique()
    missing = set(events) - set(year_events)
    if missing:
        missing_events.append((year, missing))

if missing_events:
    print("Eventi mancanti per anno:")
    for year, missing in missing_events:
        print(f"Anno {year}: {missing}")
else:
    print("Tutti gli eventi sono presenti per ogni anno.")

# 2. Verifica la completezza dei dati meteo
missing_weather = df[df[['Conditions', 'Temperature', 'Track Conditions', 'Humidity', 'Ground Temp']].isnull().any(axis=1)]
if not missing_weather.empty:
    print(f"Righe con dati meteo mancanti: {len(missing_weather)}")
    print(missing_weather)
else:
    print("Tutti i dati meteo sono completi.")

# 3. Verifica i titoli delle gare
missing_titles = df[df['Race Title'].isnull()]
if not missing_titles.empty:
    print(f"Righe con titoli di gara mancanti: {len(missing_titles)}")
    print(missing_titles)
else:
    print("Tutti i titoli delle gare sono presenti.")