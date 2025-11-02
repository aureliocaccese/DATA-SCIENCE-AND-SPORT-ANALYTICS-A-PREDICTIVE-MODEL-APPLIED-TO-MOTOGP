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

            # Trova le informazioni meteo
            weather_section = soup.find('div', class_='results-table__weather')
            if weather_section:
                conditions = weather_section.find('span', class_='results-table__weather-header--cell').text.strip() if weather_section.find('span', class_='results-table__weather-header--cell') else "N/A"
                temperature = weather_section.find('span', class_='results-table__weather-cell').text.strip() if weather_section.find('span', class_='results-table__weather-cell') else "N/A"
                track_conditions = weather_section.find('span', text='Track conditions').find_next('span').text.strip() if weather_section.find('span', text='Track conditions') else "N/A"
                humidity = weather_section.find('span', text='Humidity').find_next('span').text.strip() if weather_section.find('span', text='Humidity') else "N/A"
                ground_temp = weather_section.find('span', text='Ground').find_next('span').text.strip() if weather_section.find('span', text='Ground') else "N/A"
            else:
                conditions, temperature, track_conditions, humidity, ground_temp = "N/A", "N/A", "N/A", "N/A", "N/A"

            # Salva i dati
            all_data.append({
                'Year': year,
                'Event': event,
                'Race Title': race_title,
                'Conditions': conditions,
                'Temperature': temperature,
                'Track Conditions': track_conditions,
                'Humidity': humidity,
                'Ground Temp': ground_temp
            })
        except Exception as e:
            print(f"Errore durante lo scraping dell'evento {event} per l'anno {year}: {e}")

# Chiudi il driver
driver.quit()

# Salva i dati in un file CSV
output_path = "/Users/aurelio/Desktop/TESI/MotoGP-API/motogp_weather_data.csv"
df = pd.DataFrame(all_data)
df.to_csv(output_path, index=False)

print(f"Dati di meteo e titoli di gara salvati in: {output_path}")