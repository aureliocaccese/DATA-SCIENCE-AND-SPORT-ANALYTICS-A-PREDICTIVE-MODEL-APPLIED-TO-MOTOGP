from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import time

# Configura il driver di Selenium con Service
service = Service("/Users/aurelio/Desktop/TESI/MotoGP-API/chromedriver")
driver = webdriver.Chrome(service=service)

# Lista per salvare tutti i dati
all_data = []

# Elenco degli eventi (formattati per l'URL)
events = ['hun', 'aut', 'cze', 'ger', 'ned', 'ita', 'ara', 'gbr', 'fra', 'esp', 'qat', 'usa', 'arg', 'tha']
#hun - Hungary (Ungheria)
#aut - Austria
#cze - Czech Republic (Repubblica Ceca)
#ger - Germany (Germania)
#ned - Netherlands (Paesi Bassi)
#ita - Italy (Italia)
#ara - Aragon (Aragona, Spagna)
###gbr - Great Britain (Gran Bretagna)
###fra - France (Francia)
###esp - Spain (Spagna)
##qat - Qatar
#usa - United States (Stati Uniti)
#arg - Argentina
#tha - Thailand (Thailandia)

# Itera sugli anni dal 2002 ad oggi
for year in range(2002, 2026):  # Dal 2002 al 2025
    for event in events:
        # URL della classifica per l'anno e l'evento corrente
        event_url = f"https://www.motogp.com/en/gp-results/{year}/{event}/motogp/rac/classification"
        print(f"Caricamento dati per l'anno {year}, evento {event}: {event_url}")
        driver.get(event_url)

        # Attendi che la pagina venga caricata
        try:
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "table")))
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')

            # Trova il titolo della gara
            race_title = soup.find('h1', class_='event-title').text.strip() if soup.find('h1', class_='event-title') else f"Event {event}"

            # Trova la tabella
            table = soup.find('table')
            if table:
                for row in table.find_all('tr')[1:]:  # Salta l'intestazione
                    cols = row.find_all('td')
                    row_data = [col.text.strip() for col in cols]
                    row_data.append(race_title)  # Aggiungi il titolo della gara
                    row_data.append(year)  # Aggiungi l'anno
                    row_data.append(event)  # Aggiungi il codice dell'evento
                    all_data.append(row_data)
        except Exception as e:
            print(f"Errore durante lo scraping dell'evento {event} per l'anno {year}: {e}")

# Chiudi il driver
driver.quit()

# Salva i dati in un file CSV
columns = ['Position', 'Points', 'Race Number', 'Rider', 'Team', 'Time/Gap', 'Grand Prix', 'Year', 'Event']
output_path = "/Users/aurelio/Desktop/TESI/MotoGP-API/motogp_results_.csv"
df = pd.DataFrame(all_data, columns=columns)
df.to_csv(output_path, index=False)

print(f"Dati di tutte le gare dal 2002 al 2025 salvati in: {output_path}")