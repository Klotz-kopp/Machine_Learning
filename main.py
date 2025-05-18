import dataframe_to_db
import model_test
from auswertung import Auswertung
from utils import zeit_messen  # <-- WICHTIG
import logging

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
    filename='ML_Pipeline.log',  # Hier den gewünschten Dateinamen angeben
    filemode='a'  # Optional: 'w' für überschreiben, 'a' für anhängen (Standard ist 'a')
)

@zeit_messen
def main():
    try:
        dataframe_to_db.main()
        model_test.main()

        auswertung = Auswertung()  # <-- Instanz erzeugen

        for gruppierung in ["modellname", "datenname"]:
            auswertung.speichere_gruppierte_ergebnisse_csv(gruppierung)
            auswertung.plot_beste_scores(gruppierung)
            auswertung.plot_schnellste_durchlaeufe(gruppierung)
            auswertung.ranking_plot(gruppierung)
            auswertung.generiere_html_report(gruppierung)
    except Exception as e:
        logging.error(f"Ein Fehler ist aufgetreten: {e}")
        print(f"Ein unerwarteter Fehler ist aufgetreten. Bitte überprüfen Sie die Logdatei.")

# -------------------------------
# Einstiegspunkt
# -------------------------------
if __name__ == "__main__":
    main()
