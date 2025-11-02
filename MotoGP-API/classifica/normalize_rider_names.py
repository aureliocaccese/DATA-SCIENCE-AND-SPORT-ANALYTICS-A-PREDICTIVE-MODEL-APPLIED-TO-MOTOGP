import pandas as pd
import re

# Carica il file CSV originale
file_path = 'classifica/motogp_results_cleaned_final.csv'
df = pd.read_csv(file_path)

def normalizza_rider(rider):
    if not isinstance(rider, str):
        return ''
    # Prendi la parola alfabetica più lunga (cognome)
    words = re.findall(r'[A-Za-z]+', rider)
    cognome = max(words, key=len).capitalize() if words else ''
    # Mapping manuale per i casi più noti
    mapping = {
        # MARQUEZ
        'marquez': 'Marquez', 'marquezm': 'Marquez', 'marquezmarc': 'Marquez', 'marquez93': 'Marquez', 'marquezmm': 'Marquez',
        'marquezalex': 'Marquez', 'alexmarquez': 'Marquez', 'marquezalex': 'Marquez',
        'marquezalex73': 'Marquez', 'marquezalexlcr': 'Marquez', 'marquezgresini': 'Marquez', 'marquezbk8': 'Marquez',
        # ROSSI
        'rossi': 'Rossi', 'rossifiat': 'Rossi', 'rossimovistar': 'Rossi', 'rossiyamaha': 'Rossi', 'rossi46': 'Rossi', 'rossivalentino': 'Rossi',
        # LORENZO
        'lorenzo': 'Lorenzo', 'lorenzorge': 'Lorenzo', 'lorenzo99': 'Lorenzo',
        # PEDROSA
        'pedrosa': 'Pedrosa', 'pedrosadani': 'Pedrosa', 'pedrosa26': 'Pedrosa',
        # STONER
        'stoner': 'Stoner', 'stonercasey': 'Stoner', 'stoner27': 'Stoner',
        # DOVIZIOSO
        'dovizioso': 'Dovizioso', 'doviziosoa': 'Dovizioso', 'doviziosoandrea': 'Dovizioso', 'dovizioso04': 'Dovizioso',
        # BAGNAIA
        'bagnaia': 'Bagnaia', 'bagnaiafrancesco': 'Bagnaia', 'bagnaia63': 'Bagnaia', 'bagnaia1': 'Bagnaia',
        # QUARTARARO
        'quartararo': 'Quartararo', 'quartararo20': 'Quartararo', 'quartararofabio': 'Quartararo',
        'quartararomonster': 'Quartararo', 'quartararopetronas': 'Quartararo',
        # VIÑALES
        'vinales': 'Viñales', 'vinalesmaverick': 'Viñales', 'vinales12': 'Viñales', 'viñales': 'Viñales',
        'alesmonster': 'Viñales', 'alesaprilia': 'Viñales', 'alesred': 'Viñales',
        # ESPARGARO
        'espargaro': 'Espargaro', 'espargaroaleix': 'Espargaro', 'espargaropol': 'Espargaro',
        'espargaroaprilia': 'Espargaro', 'espargarored': 'Espargaro', 'espargarorepsol': 'Espargaro',
        'espargarogasgas': 'Espargaro', 'espargarohonda': 'Espargaro',
        # BINDER
        'binder': 'Binder', 'binderbrad': 'Binder', 'binder33': 'Binder', 'binderred': 'Binder', 'binderwithu': 'Binder',
        # MARTIN
        'martin': 'Martin', 'martinjorge': 'Martin', 'martin89': 'Martin', 'martinpramac': 'Martin', 'martinprima': 'Martin', 'martinaprilia': 'Martin',
        # ZARCO
        'zarco': 'Zarco', 'zarcojohann': 'Zarco', 'zarco5': 'Zarco', 'zarcopramac': 'Zarco', 'zarcoesponsorama': 'Zarco', 'zarcocastrol': 'Zarco', 'zarcoprima': 'Zarco',
        # MIR
        'mir': 'Mir', 'mirjoan': 'Mir', 'mir36': 'Mir', 'mirteam': 'Mir', 'mirrepsol': 'Mir', 'mirhonda': 'Mir',
        # BASTIANINI
        'bastianini': 'Bastianini', 'bastianinienea': 'Bastianini', 'bastianini23': 'Bastianini', 'bastianiniavintia': 'Bastianini', 'bastianinigresini': 'Bastianini', 'bastianiniducati': 'Bastianini', 'bastianinired': 'Bastianini',
        # BEZZECCHI
        'bezzecchi': 'Bezzecchi', 'bezzecchimarco': 'Bezzecchi', 'bezzecchi72': 'Bezzecchi', 'bezzecchisky': 'Bezzecchi', 'bezzecchimooney': 'Bezzecchi', 'bezzecchipertamina': 'Bezzecchi', 'bezzecchiaprilia': 'Bezzecchi',
        # MORBIDELLI
        'morbidelli': 'Morbidelli', 'morbidellifranco': 'Morbidelli', 'morbidelli21': 'Morbidelli', 'morbidellipetronas': 'Morbidelli', 'morbidellimonster': 'Morbidelli', 'morbidelliprima': 'Morbidelli', 'morbidellipertamina': 'Morbidelli',
        # OLIVEIRA
        'oliveira': 'Oliveira', 'oliveiramiguel': 'Oliveira', 'oliveira88': 'Oliveira', 'oliveirared': 'Oliveira', 'oliveiracryptodata': 'Oliveira', 'oliveiratrackhouse': 'Oliveira', 'oliveiraprima': 'Oliveira',
        # MILLER
        'miller': 'Miller', 'millerjack': 'Miller', 'miller43': 'Miller', 'millerpramac': 'Miller', 'millerducati': 'Miller', 'millerred': 'Miller', 'millerprima': 'Miller',
        # MARINI
        'marini': 'Marini', 'mariniluca': 'Marini', 'marini10': 'Marini', 'marinisky': 'Marini', 'marinimooney': 'Marini', 'marinirepsol': 'Marini', 'marinihonda': 'Marini',
        # ACOSTA
        'acosta': 'Acosta', 'acostapedro': 'Acosta', 'acosta37': 'Acosta', 'acostared': 'Acosta',
        # RINS
        'rins': 'Rins', 'rinsalex': 'Rins', 'rins42': 'Rins', 'rinsteam': 'Rins', 'rinslcr': 'Rins', 'rinsmonster': 'Rins',
        # SAVADORI
        'savadori': 'Savadori', 'savadoriaprilia': 'Savadori', 'savadoricryptodata': 'Savadori',
        # LECOUNA
        'lecuona': 'Lecuona', 'lecuonared': 'Lecuona', 'lecuonatech': 'Lecuona', 'lecuonalcr': 'Lecuona', 'lecuonarepsol': 'Lecuona',
        # CRUTCHLOW
        'crutchlow': 'Crutchlow', 'crutchlowlcr': 'Crutchlow', 'crutchlowmonster': 'Crutchlow', 'crutchlowwithu': 'Crutchlow',
        # NAKAGAMI
        'nakagami': 'Nakagami', 'nakagamilcr': 'Nakagami', 'nakagamiidemitsu': 'Nakagami', 'nakagamihonda': 'Nakagami',
        # PIRRO
        'pirro': 'Pirro', 'pirropramac': 'Pirro', 'pirroaruba': 'Pirro', 'pirropertamina': 'Pirro',
        # FERNANDEZ
        'fernandez': 'Fernandez', 'fernandeztech': 'Fernandez', 'fernandezgasgas': 'Fernandez', 'fernandezcryptodata': 'Fernandez', 'fernandezprima': 'Fernandez', 'fernandezred': 'Fernandez', 'fernandeztrackhouse': 'Fernandez', 'fernandezyamaha': 'Fernandez',
        # GARDNER
        'gardner': 'Gardner', 'gardnertech': 'Gardner', 'gardnermonster': 'Gardner',
        # GIANNANTONIO
        'giannantonio': 'Giannantonio', 'giannantoniobeta': 'Giannantonio', 'giannantoniogresini': 'Giannantonio', 'giannantoniopertamina': 'Giannantonio',
        # SMITH
        'smithaprilia': 'Smith',
        # DIXON
        'dixonpetronas': 'Dixon',
        # BENDSNEYDER
        'bendsneydernts': 'Bendsneyder',
        # BEZZECCHI (già sopra)
        # PONS
        'ponsfederal': 'Pons',
        # MANZI
        'manzimv': 'Manzi',
        # CORSI
        'corsimv': 'Corsi',
        # IZDIHAR
        'izdiharidemitsu': 'Izdihar',
        # RAFFIN
        'raffinnts': 'Raffin',
        # PORTA
        'portaitaltrans': 'Porta',
        # CHANTRA
        'chantraidemitsu': 'Chantra',
        # OGURA
        'oguratrackhouse': 'Ogura',
        # ALDEGUER
        'aldeguerbk': 'Aldeguer',
        # FOLGER
        'folgergasgas': 'Folger',
        # ALTRI
        'bradlrepsol': 'Bradl', 'bradlhrc': 'Bradl',
    }
    cognome_lower = cognome.lower()
    for key in mapping:
        if cognome_lower.startswith(key.lower()):
            return mapping[key]
    return cognome

# Applica la normalizzazione
if 'Rider' in df.columns:
    df['Rider_normalized'] = df['Rider'].apply(normalizza_rider)

# Salva il nuovo file
output_path = 'classifica/motogp_results_cleaned_final_normalized.csv'
df.to_csv(output_path, index=False)
print(f"File normalizzato salvato in: {output_path}")
