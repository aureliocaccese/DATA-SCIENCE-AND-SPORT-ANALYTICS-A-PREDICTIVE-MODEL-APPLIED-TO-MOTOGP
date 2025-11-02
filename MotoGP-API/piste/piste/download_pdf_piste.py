import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Configura il driver di Selenium
options = webdriver.ChromeOptions()
service = Service("/Users/aurelio/Desktop/TESI/MotoGP-API/chromedriver")
driver = webdriver.Chrome(service=service, options=options)

download_dir = "/Users/aurelio/Desktop/TESI/MotoGP-API/piste/pdf"
os.makedirs(download_dir, exist_ok=True)

events = ['hun', 'aut', 'cze', 'ger', 'ned', 'ita', 'ara', 'gbr', 'fra', 'esp', 'qat', 'usa', 'arg', 'tha']
base_url = "https://www.motogp.com/en/gp-results/2025/{event}/motogp/rac/classification"

for event in events:
    try:
        url = base_url.format(event=event)
        print(f"Accedendo alla pagina: {url}")
        driver.get(url)

        # Cerca il link "Circuit Information"
        try:
            circuit_info_link = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.XPATH, "//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'circuit information')]"))
            )
            pdf_url = circuit_info_link.get_attribute("href")
            print(f"Trovato il link del PDF: {pdf_url}")

            # Scarica il PDF con requests
            response = requests.get(pdf_url)
            if response.status_code == 200:
                filename = f"{event}_circuit_information.pdf"
                filepath = os.path.join(download_dir, filename)
                with open(filepath, "wb") as f:
                    f.write(response.content)
                print(f"PDF salvato: {filepath}")
            else:
                print(f"Errore nel download del PDF per {event}: status {response.status_code}")
        except Exception as e:
            print(f"Link 'Circuit Information' non trovato per l'evento {event}: {e}")

    except Exception as e:
        print(f"Errore durante l'elaborazione dell'evento {event}: {e}")

driver.quit()
print(f"Tutti i PDF disponibili sono stati scaricati nella directory: {download_dir}")