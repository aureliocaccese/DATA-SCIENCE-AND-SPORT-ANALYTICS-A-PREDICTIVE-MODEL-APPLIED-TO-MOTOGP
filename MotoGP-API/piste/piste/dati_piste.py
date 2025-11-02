import os
import pdfplumber
import pandas as pd
import re

pdf_dir = "/Users/aurelio/Desktop/TESI/MotoGP-API/piste/piste/pdf"
output_csv = "/Users/aurelio/Desktop/TESI/MotoGP-API/piste/piste/dati_piste.csv"

wanted = [
    "name",
    "length",
    "width",
    "right corners",
    "left corners",
    "longest straight"
]

def normalize(s):
    return str(s).strip().lower().replace("  ", " ")

rows = []

for filename in os.listdir(pdf_dir):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_dir, filename)
        data = {"file": filename}
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
                for w in wanted:
                    # Regex per trovare la voce e il valore dopo i due punti
                    match = re.search(rf"{w}[\s:]*([^\n]+)", text, re.IGNORECASE)
                    if match:
                        data[w] = match.group(1).strip()
                    else:
                        data[w] = ""
            rows.append(data)
        except Exception as e:
            print(f"Errore con {filename}: {e}")

# Salva in CSV
df = pd.DataFrame(rows)
df.to_csv(output_csv, index=False)
print(f"Dati salvati in {output_csv}.")